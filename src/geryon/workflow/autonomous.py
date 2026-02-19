"""Autonomous workflow using LangGraph for tool-based exploration."""

from datetime import UTC, datetime
import json
import re
from typing import Annotated, Literal, TypedDict
import uuid

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from geryon.db import Database
from geryon.engine import HypothesisExecutor
from geryon.labeling.labeled_store import LabeledStore
from geryon.labeling.models import LabeledHypothesis
from geryon.labeling.storage import HypothesisStore
from geryon.lang.spec import GeryonHyp
from geryon.llm.conversation_logger import SessionTracer
from geryon.llm.generator import HypothesisProposal
from geryon.llm.narrator import ResultNarrator
from geryon.llm.provider import create_provider
from geryon.tools.database import describe_table, list_tables, query_data
from geryon.workflow.session import Session, SessionConfig


class AgentState(TypedDict):
    """State for LangGraph agent."""

    messages: Annotated[list[BaseMessage], "Messages in conversation"]


class AutonomousWorkflow:
    """LangGraph-based autonomous hypothesis generation.

    Uses tool calling to explore database before proposing hypotheses.
    """

    def __init__(self, config: SessionConfig):
        """Initialize autonomous workflow.

        Parameters
        ----------
        config : SessionConfig
            Session configuration
        """
        self.config = config
        self.session = Session(config)

        self.db = Database(config.parquet_dir)
        self.executor = HypothesisExecutor(self.db)
        provider_kwargs: dict = {"model": config.model}
        if config.provider_type == "aws_bedrock":
            provider_kwargs["region"] = config.aws_region
            provider_kwargs["profile"] = config.aws_profile
        self.provider = create_provider(config.provider_type, **provider_kwargs)

        self.store = HypothesisStore(config.storage_dir)
        self.labeled_store = LabeledStore(config.labeled_dir)

        self.llm_logger = None
        if config.enable_llm_logging:
            self.llm_logger = SessionTracer(
                storage_dir=config.storage_dir,
                session_id=config.session_id,
                model=f"{config.provider_type}/{config.model}",
            )

        # Create LangChain-compatible tools
        self.tools = self._create_tools()

        # Create LangChain LLM
        self.llm = self._create_llm()

        # Build LangGraph workflow
        self.graph = self._build_graph()

    def _create_tools(self):
        """Create LangChain tool wrappers for database exploration."""

        @tool
        def list_tables_tool() -> str:
            """List all available tables in the database."""
            print("[TOOL] list_tables_tool called")
            result = list_tables(self.db)
            print(f"[TOOL] list_tables_tool done, {len(result)} chars")
            return result

        @tool
        def describe_table_tool(table_name: str) -> str:
            """Get schema (columns, types, sample values) for a specific table.

            Use this to understand what data is available in a table before
            querying it.
            """
            print(f"[TOOL] describe_table_tool({table_name}) called")
            result = describe_table(self.db, table_name)
            print(f"[TOOL] describe_table_tool done, {len(result)} chars")
            return result

        @tool
        def query_data_tool(sql: str) -> str:
            """Run SELECT query to explore data (max 100 rows).

            Use this to examine actual data values, check distributions,
            or validate that certain values exist before proposing a hypothesis.
            """
            print(f"[TOOL] query_data_tool called: {sql[:100]}...")
            result = query_data(self.db, sql)
            print(f"[TOOL] query_data_tool done, {len(result)} chars")
            return result

        return [list_tables_tool, describe_table_tool, query_data_tool]

    def _create_llm(self):
        """Create LangChain model from provider config."""
        if self.config.provider_type == "openai":
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": self.config.model,
                "temperature": 0.8,
                "max_tokens": 16384,
            }
            if self.config.base_url:
                kwargs["openai_api_base"] = self.config.base_url
            if self.config.api_key:
                kwargs["openai_api_key"] = self.config.api_key

            return ChatOpenAI(**kwargs)  # type: ignore[arg-type]
        elif self.config.provider_type == "anthropic":
            from langchain_anthropic import ChatAnthropic

            kwargs = {
                "model_name": self.config.model,
                "temperature": 0.8,
                "max_tokens": 16384,
                "timeout": 300.0,
                "stop": None,
            }
            if self.config.api_key:
                kwargs["anthropic_api_key"] = self.config.api_key

            return ChatAnthropic(**kwargs)  # type: ignore[arg-type]
        elif self.config.provider_type == "aws_bedrock":
            from botocore.config import Config as BotoConfig
            from langchain_aws import ChatBedrock

            boto_config = BotoConfig(
                read_timeout=300,
                connect_timeout=30,
                retries={"max_attempts": 2},
            )
            kwargs = {
                "model_id": self.config.model,
                "model_kwargs": {"temperature": 0.8, "max_tokens": 16384},
                "config": boto_config,
            }
            if self.config.aws_region:
                kwargs["region_name"] = self.config.aws_region
            if self.config.aws_profile:
                kwargs["credentials_profile_name"] = self.config.aws_profile
            # When using inference profile ARN, provider must be specified
            if "arn:" in self.config.model or "anthropic" in self.config.model.lower():
                kwargs["provider"] = "anthropic"

            return ChatBedrock(**kwargs)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown provider type: {self.config.provider_type}")

    def _build_graph(self):
        """Build LangGraph workflow using ReAct agent."""
        from langgraph.prebuilt import create_react_agent

        # Use ReAct agent which handles reasoning better with local models
        return create_react_agent(self.llm, self.tools)

    def _agent_node(self, state: AgentState) -> dict:
        """Agent node: calls LLM with tools."""
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Invoke LLM
        response = llm_with_tools.invoke(state["messages"])

        # Return updated state
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Decide whether to continue or end."""
        last_message = state["messages"][-1]

        # If LLM made tool calls, continue to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"  → LLM requesting {len(last_message.tool_calls)} tool call(s):")
            for tc in last_message.tool_calls:
                print(f"    - {tc['name']}({list(tc.get('args', {}).keys())})")
            return "continue"

        # Otherwise, end
        print("  → LLM finished (no more tool calls)")
        return "end"

    def run_iteration(
        self, n_proposals: int | None = None, iteration: int | None = None
    ) -> list[LabeledHypothesis]:
        """Run one iteration: generate hypotheses via tool exploration.

        Parameters
        ----------
        n_proposals : int, optional
            Number of proposals to generate (default from config)
        iteration : int, optional
            Iteration number (1-indexed) for tracking

        Returns
        -------
        list[LabeledHypothesis]
            Generated and executed hypotheses
        """
        if n_proposals is None:
            n_proposals = self.config.num_proposals_per_iteration

        # Load labeled hypotheses (from prior annotation) and same-session hypotheses
        labeled = self.labeled_store.load_all()
        session_previous = self.store.load()
        previous_context = self._format_previous_hypotheses(labeled, session_previous)

        print(f"Generating {n_proposals} hypothesis(es)...")
        print(f"Using model: {self.config.provider_type}/{self.config.model}")
        print()

        # Create initial prompt - enforce multi-step validation
        prompt_text = f"""You are helping generate cancer research hypotheses.
You MUST explore the database using tools before proposing ANY hypotheses.

{previous_context}

SCIENTIFIC DISCOVERY GOALS:

Your primary goal is to discover NEW, unpublished relationships in this cancer
genomics data. While confirming known findings is useful as sanity checks
(especially when expected results DON'T replicate — that is itself a discovery),
prioritize:

- Unexpected gene-outcome associations not well-established in literature
- Interactions between clinical features and molecular markers
- Subgroup effects that might not be obvious to an oncologist
- Comparisons that test non-obvious biological hypotheses

If a well-known association fails to replicate, flag it — that is valuable.
But spend most proposals on genuinely exploratory comparisons.

HYPOTHESIS REFINEMENT:

When you see a previous hypothesis with uncontrolled=2 or
uncontrolled=3, consider proposing a refined version that controls
for the noted confounder (e.g., restrict both cohorts to a single
cancer type or age range).

To mark a proposal as a refinement, set "refines_hypothesis" to
the 8-character ID shown in [brackets] next to the parent. This
is OPTIONAL — only use it when genuinely building on a prior result.

Example:
  Previous: [a3f1c9e2] KRAS mut vs WT [uncontrolled=3] — confounded by age
  Refinement: restrict both cohorts to age >= 60,
  set "refines_hypothesis": "a3f1c9e2"

REQUIRED EXPLORATION STEPS (DO NOT SKIP ANY):

Step 1: Call list_tables_tool() to see all available tables

Step 2: For EACH table you want to use in a hypothesis:
   - Call describe_table_tool(table_name) to see its columns
   - Note the EXACT column names, data types, and sample values
   - Do NOT assume column names - verify them first!

Step 3: If you need to verify specific filter values beyond what describe_table
   showed, call query_data_tool(sql).
   Example: SELECT DISTINCT "CANCER_TYPE" FROM clinical_patient LIMIT 20
   describe_table now shows top values with counts, so you can often skip
   this step — but DO query if the column is high-cardinality or the value
   you need wasn't in the top values shown.

Step 4: Generate {n_proposals} hypothesis(es) using ONLY columns AND values
   you verified in Steps 2-3

CRITICAL VALIDATION:
- Your proposals will be checked against the actual database schema
- If you reference a table or column that doesn't exist, that proposal will be REJECTED
- If you use a filter value that doesn't exist in the data, the cohort
  will be EMPTY and the hypothesis will fail
- Using real, verified column names AND values is MANDATORY

Output format:
{{
  "proposals": [
    {{
      "cohort_a_description": "Patients with KRAS mutation",
      "cohort_b_description": "Patients without KRAS mutation",
      "outcome_description": "Overall survival",
      "rationale": "...",
      "refines_hypothesis": null,
      "geryon_spec": {{
        "version": 1,
        "query": {{
          "operation": "compare_cohorts",
          "cohort_a": {{
            "operation": "select_cohort",
            "filters": [
              {{"table": "gene_matrix", "column": "KRAS", "operator": "==", "value": 1}}
            ]
          }},
          "cohort_b": {{
            "operation": "select_cohort",
            "filters": [
              {{"table": "gene_matrix", "column": "KRAS", "operator": "==", "value": 0}}
            ]
          }},
          "outcome": {{
            "outcome_type": "overall_survival",
            "time_column": "OS_MONTHS",
            "event_column": "OS_STATUS",
            "table": "clinical_patient"
          }},
          "method": "hazard_ratio_cox"
        }}
      }}
    }}
  ]
}}

ADDITIONAL OUTCOME EXAMPLES:

Time to next treatment (use with hazard_ratio_cox):
  "outcome": {{
    "outcome_type": "time_to_next_treatment",
    "agent": "Pembrolizumab",
    "gap_days": 90,
    "table": "timeline_treatment"
  }}

Survival from treatment (use with hazard_ratio_cox):
  "outcome": {{
    "outcome_type": "survival_from_treatment",
    "agent": "Carboplatin",
    "table": "timeline_treatment"
  }}

Metastatic burden (use with wilcoxon_rank_sum):
  "outcome": {{
    "outcome_type": "metastatic_burden",
    "landmark_days": 0,
    "window_days": 30,
    "table": "timeline_tumor_sites"
  }}

IMPORTANT:
- operator must be one of: ==, !=, >, <, >=, <=, in
- method: "hazard_ratio_cox" or "wilcoxon_rank_sum"
- outcome_type: "overall_survival",
  "time_to_next_treatment", "survival_from_treatment",
  or "metastatic_burden"
- hazard_ratio_cox works with: overall_survival,
  time_to_next_treatment, survival_from_treatment
- wilcoxon_rank_sum works with: metastatic_burden
- cohort_a and cohort_b must have "filters"
  as an ARRAY of filter objects"""
        initial_message = HumanMessage(content=prompt_text)

        # Run LangGraph
        try:
            result = self.graph.invoke({"messages": [initial_message]})

            if self.llm_logger:
                self.llm_logger.log_raw_messages(result["messages"])

            # Emit compact tool-call events from the conversation
            if self.llm_logger:
                for msg in result["messages"]:
                    if getattr(msg, "type", None) == "tool":
                        tool_name = getattr(msg, "name", "unknown")
                        tool_args = self._find_tool_args(
                            result["messages"],
                            getattr(msg, "tool_call_id", None),
                        )
                        self.llm_logger.log_tool_call(
                            tool_name, tool_args, msg.content or ""
                        )

            # Extract proposals from final message
            final_message = result["messages"][-1]

            # Check for truncation
            response_meta = getattr(final_message, "response_metadata", {}) or {}
            stop_reason = response_meta.get(
                "stop_reason", response_meta.get("finish_reason", "")
            )
            if stop_reason in ("max_tokens", "length"):
                print(
                    "\u26a0 LLM response was TRUNCATED (hit max_tokens limit). "
                    "Proposals may be incomplete."
                )

            # Log the conversation for debugging
            print(f"LLM completed after {len(result['messages'])} messages")
            print(f"Final response type: {type(final_message).__name__}")
            print("Final response preview (first 500 chars):")
            print(f"  {str(final_message.content)[:500]}")
            print()

            proposals = self._extract_proposals(final_message.content)

            if not proposals:
                print("⚠ WARNING: No valid proposals extracted from LLM response")
                print(f"  Raw response length: {len(final_message.content)} chars")
                print("  Response content:")
                print(f"  {final_message.content[:1000]}")
                print("  Continuing to next iteration...")
                return []

            # Resolve short IDs to full UUIDs
            short_to_full: dict[str, str] = {}
            for hyp in labeled + session_previous:
                sid = hyp.hypothesis_id[:8]
                short_to_full[sid] = hyp.hypothesis_id
            for proposal in proposals:
                ref = proposal.refines_hypothesis
                if ref:
                    full_id = short_to_full.get(ref)
                    if full_id:
                        proposal.refines_hypothesis = full_id
                    else:
                        print(
                            f"  ⚠ Could not resolve refines_hypothesis "
                            f"'{ref}', clearing"
                        )
                        proposal.refines_hypothesis = None

            print(f"Generated {len(proposals)} valid proposals. Executing all...")
            print()

            # Execute and narrate
            labeled_hypotheses = []
            narrator = ResultNarrator(self.provider, logger=self.llm_logger)

            for i, proposal in enumerate(proposals, 1):
                a_desc = proposal.cohort_a_description
                b_desc = proposal.cohort_b_description
                print(f"  [{i}/{len(proposals)}] Executing: {a_desc} vs {b_desc}")

                if self.llm_logger:
                    self.llm_logger.log_proposal(idx=i, proposal=proposal)

                try:
                    start_time = datetime.now(UTC)
                    result = self.executor.execute(proposal.geryon_spec)
                    execution_time = (datetime.now(UTC) - start_time).total_seconds()

                    if self.llm_logger:
                        self.llm_logger.log_execution(
                            idx=i, result=result, time_s=execution_time
                        )

                    print(f"    Executed in {execution_time:.1f}s. Narrating...")

                    narrative = narrator.narrate(
                        spec=proposal.geryon_spec,
                        result=result,
                        proposal_rationale=proposal.rationale,
                        idx=i,
                    )

                    labeled_hyp = LabeledHypothesis(
                        hypothesis_id=str(uuid.uuid4()),
                        session_id=self.session.session_id,
                        proposal=proposal,
                        spec=proposal.geryon_spec,
                        result=result,
                        execution_time_seconds=execution_time,
                        narrative=narrative,
                        llm_model=f"{self.config.provider_type}/{self.config.model}",
                        iteration=iteration,
                    )

                    self.store.save(labeled_hyp)
                    labeled_hypotheses.append(labeled_hyp)

                    print(f"    Saved. Summary: {narrative.summary}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

            return labeled_hypotheses

        except Exception as e:
            print("⚠ WARNING: Hypothesis generation failed")
            print(f"  Error: {type(e).__name__}: {str(e)}")

            # Show more details for debugging
            import traceback

            print("  Full traceback:")
            traceback.print_exc()

            return []

    def run_full_session(self) -> list[LabeledHypothesis]:
        """Run full session with multiple iterations.

        Returns
        -------
        list[LabeledHypothesis]
            All hypotheses generated in session
        """
        all_hypotheses = []

        print(f"Starting autonomous session {self.session.session_id}")
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"Proposals per iteration: {self.config.num_proposals_per_iteration}")
        print()

        for i in range(self.config.max_iterations):
            print(f"=== Iteration {i+1}/{self.config.max_iterations} ===")

            hypotheses = self.run_iteration(iteration=i + 1)

            # Check if we got any hypotheses this iteration
            if not hypotheses:
                print(f"⚠ No hypotheses generated in iteration {i+1}, continuing...")
                print()
                continue

            if self.config.critic_cycles > 0:
                from geryon.llm.critic import HypothesisCritic

                critic = HypothesisCritic(self.provider, logger=self.llm_logger)
                for cycle in range(self.config.critic_cycles):
                    print(f"  Critic cycle {cycle + 1}/{self.config.critic_cycles}...")
                    hypotheses = critic.rate(hypotheses)

                all_session = self.store.load()
                hyp_by_id = {h.hypothesis_id: h for h in hypotheses}
                updated = [hyp_by_id.get(h.hypothesis_id, h) for h in all_session]
                self.store.save_all(updated)

            all_hypotheses.extend(hypotheses)
            print(f"✓ Iteration {i+1} complete: {len(hypotheses)} hypotheses")
            print()

        if self.llm_logger:
            successful = sum(1 for h in all_hypotheses if h.result.success)
            failed = len(all_hypotheses) - successful
            self.llm_logger.log_session_end(
                total=len(all_hypotheses), successful=successful, failed=failed
            )

        print(f"Session complete. Generated {len(all_hypotheses)} hypotheses total.")
        return all_hypotheses

    def _extract_proposals(self, content: str) -> list[HypothesisProposal]:
        """Extract GeryonHyp proposals from LLM response.

        Parameters
        ----------
        content : str
            LLM response content (JSON or JSON wrapped in markdown)

        Returns
        -------
        list[HypothesisProposal]
            Extracted proposals
        """
        try:
            # Try to find JSON in response (might be wrapped in markdown code block)
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Assume entire content is JSON
                json_str = content

            data = json.loads(json_str)

            # Handle both single proposal and proposals array
            proposals_data = data.get(
                "proposals", [data] if "geryon_spec" in data else []
            )

            proposals = []
            for i, item in enumerate(proposals_data, 1):
                proposal = HypothesisProposal(
                    cohort_a_description=item["cohort_a_description"],
                    cohort_b_description=item["cohort_b_description"],
                    outcome_description=item["outcome_description"],
                    rationale=item["rationale"],
                    geryon_spec=GeryonHyp(**item["geryon_spec"]),
                    refines_hypothesis=item.get("refines_hypothesis"),
                )

                # Validate tables/columns exist before accepting
                validation_errors = self._validate_geryon_spec(proposal.geryon_spec)
                if validation_errors:
                    print(f"  ⚠ REJECTED proposal {i}: {'; '.join(validation_errors)}")
                    continue

                proposals.append(proposal)

            if not proposals and proposals_data:
                print("  ⚠ All proposals were rejected due to validation errors")

            return proposals

        except Exception as e:
            print("⚠ WARNING: Failed to extract proposals from LLM response")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            print(f"  Response length: {len(content)} chars")
            print("  Response preview (first 500 chars):")
            print(f"  {content[:500]}")
            if len(content) > 500:
                print("  Response preview (last 500 chars):")
                print(f"  {content[-500:]}")
            return []

    def _validate_geryon_spec(self, spec: GeryonHyp) -> list[str]:
        """Validate that tables and columns in spec actually exist.

        Parameters
        ----------
        spec : GeryonHyp
            Hypothesis specification to validate

        Returns
        -------
        list[str]
            List of validation error messages (empty if valid)
        """
        errors = []
        tables = self.db.list_tables()

        # Check cohort_a filters
        for i, filter_obj in enumerate(spec.query.cohort_a.filters):
            if filter_obj.table not in tables:
                errors.append(
                    f"Cohort A filter {i}: Table '{filter_obj.table}' does not exist"
                )
            else:
                try:
                    table_df = self.db.execute(
                        f"SELECT * FROM {filter_obj.table} LIMIT 0"
                    )
                    columns = table_df.columns.tolist()
                    if filter_obj.column not in columns:
                        errors.append(
                            f"Cohort A filter {i}: Column '{filter_obj.column}' "
                            f"not in {filter_obj.table}"
                        )
                except Exception as e:
                    errors.append(
                        f"Cohort A filter {i}: Error checking {filter_obj.table}: {e}"
                    )

        # Check cohort_b filters
        for i, filter_obj in enumerate(spec.query.cohort_b.filters):
            if filter_obj.table not in tables:
                errors.append(
                    f"Cohort B filter {i}: Table '{filter_obj.table}' does not exist"
                )
            else:
                try:
                    table_df = self.db.execute(
                        f"SELECT * FROM {filter_obj.table} LIMIT 0"
                    )
                    columns = table_df.columns.tolist()
                    if filter_obj.column not in columns:
                        errors.append(
                            f"Cohort B filter {i}: Column '{filter_obj.column}' "
                            f"not in {filter_obj.table}"
                        )
                except Exception as e:
                    errors.append(
                        f"Cohort B filter {i}: Error checking {filter_obj.table}: {e}"
                    )

        # Check outcome table/columns
        outcome = spec.query.outcome
        if outcome.table not in tables:
            errors.append(f"Outcome table '{outcome.table}' does not exist")
        elif hasattr(outcome, "time_column") and hasattr(outcome, "event_column"):
            try:
                table_df = self.db.execute(f"SELECT * FROM {outcome.table} LIMIT 0")
                columns = table_df.columns.tolist()
                if outcome.time_column not in columns:
                    errors.append(
                        f"Time column '{outcome.time_column}' not in {outcome.table}"
                    )
                if outcome.event_column not in columns:
                    errors.append(
                        f"Event column '{outcome.event_column}' not in {outcome.table}"
                    )
            except Exception as e:
                errors.append(f"Error checking {outcome.table}: {str(e)}")

        return errors

    def _format_previous_hypotheses(
        self,
        labeled: list[LabeledHypothesis],
        session_previous: list[LabeledHypothesis],
    ) -> str:
        """Format labeled and same-session hypotheses for LLM context."""
        from geryon.workflow.context import format_previous_hypotheses

        return format_previous_hypotheses(labeled, session_previous)

    @staticmethod
    def _find_tool_args(messages: list, tool_call_id: str | None) -> dict:
        """Find the args dict for a tool call by its id."""
        if not tool_call_id:
            return {}
        for msg in messages:
            for tc in getattr(msg, "tool_calls", None) or []:
                if tc.get("id") == tool_call_id:
                    return tc.get("args", {})
        return {}
