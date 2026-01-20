"""Autonomous workflow using LangGraph for tool-based exploration."""

from datetime import datetime
import json
import re
from typing import Annotated, Literal, TypedDict
import uuid

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from msk_cycl.db import Database
from msk_cycl.engine import HypothesisExecutor
from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.labeling.storage import HypothesisStore
from msk_cycl.lang.spec import CyclHyp
from msk_cycl.llm.conversation_logger import ConversationLogger
from msk_cycl.llm.generator import HypothesisProposal
from msk_cycl.llm.narrator import ResultNarrator
from msk_cycl.llm.provider import create_provider
from msk_cycl.tools.database import describe_table, list_tables, query_data
from msk_cycl.workflow.session import Session, SessionConfig


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
        self.provider = create_provider(
            config.provider_type,
            model=config.model,
        )

        self.store = HypothesisStore(config.storage_dir)

        self.llm_logger = None
        if config.enable_llm_logging:
            self.llm_logger = ConversationLogger(
                storage_dir=config.storage_dir,
                session_id=config.session_id,
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
            return list_tables(self.db)

        @tool
        def describe_table_tool(table_name: str) -> str:
            """Get schema (columns, types, sample values) for a specific table.

            Use this to understand what data is available in a table before
            querying it.
            """
            return describe_table(self.db, table_name)

        @tool
        def query_data_tool(sql: str) -> str:
            """Run SELECT query to explore data (max 100 rows).

            Use this to examine actual data values, check distributions,
            or validate that certain values exist before proposing a hypothesis.
            """
            return query_data(self.db, sql)

        return [list_tables_tool, describe_table_tool, query_data_tool]

    def _create_llm(self):
        """Create LangChain model from provider config."""
        if self.config.provider_type == "ollama":
            from langchain_ollama import ChatOllama

            # Don't use format="json" - it breaks tool calling!
            # Let the model use native tool calling instead
            return ChatOllama(
                model=self.config.model,
                base_url="http://localhost:11434",
                temperature=0.8,
            )
        elif self.config.provider_type == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.config.model,
                temperature=0.8,
            )
        elif self.config.provider_type == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model_name=self.config.model,
                temperature=0.8,
                timeout=300.0,
                stop=None,
            )
        else:
            raise ValueError(f"Unknown provider type: {self.config.provider_type}")

    def _build_graph(self):
        """Build LangGraph workflow using ReAct agent for better Ollama support."""
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

    def run_iteration(self, n_proposals: int | None = None) -> list[LabeledHypothesis]:
        """Run one iteration: generate hypotheses via tool exploration.

        Parameters
        ----------
        n_proposals : int, optional
            Number of proposals to generate (default from config)

        Returns
        -------
        list[LabeledHypothesis]
            Generated and executed hypotheses
        """
        if n_proposals is None:
            n_proposals = self.config.num_proposals_per_iteration

        # Generate 3x more proposals to allow for validation filtering
        n_to_generate = n_proposals * 3

        # Load previous hypotheses to avoid duplicates
        previous = self.store.load_session(self.session.session_id)
        previous_context = self._format_previous_hypotheses(previous)

        print(
            f"Generating {n_to_generate} hypothesis(es) "
            f"(will filter to best {n_proposals})..."
        )
        print(f"Using model: {self.config.provider_type}/{self.config.model}")
        print()

        # Create initial prompt - enforce multi-step validation
        prompt_text = f"""You are helping generate cancer research hypotheses.
You MUST explore the database using tools before proposing ANY hypotheses.

{previous_context}

REQUIRED EXPLORATION STEPS (DO NOT SKIP ANY):

Step 1: Call list_tables_tool() to see all available tables

Step 2: For EACH table you want to use in a hypothesis:
   - Call describe_table_tool(table_name) to see its columns
   - Note the EXACT column names, data types, and sample values
   - Do NOT assume column names - verify them first!

Step 3 (Optional): Call query_data_tool(sql) to check value distributions

Step 4: Generate {n_to_generate} hypothesis(es) using ONLY columns you verified

CRITICAL VALIDATION:
- Your proposals will be checked against the actual database schema
- If you reference a table or column that doesn't exist, that proposal will be REJECTED
- Using real, verified column names is MANDATORY

Output format:
{{
  "proposals": [
    {{
      "cohort_a_description": "...",
      "cohort_b_description": "...",
      "outcome_description": "Overall survival",
      "rationale": "...",
      "cycl_spec": {{
        "cohort_a": {{
          "table": "...", "column": "...", "operator": "...", "value": ...
        }},
        "cohort_b": {{
          "table": "...", "column": "...", "operator": "...", "value": ...
        }},
        "outcome": {{
          "table": "clinical_patient",
          "time_column": "OS_MONTHS",
          "event_column": "OS_STATUS"
        }}
      }}
    }}
  ]
}}"""
        initial_message = HumanMessage(content=prompt_text)

        # Run LangGraph
        try:
            result = self.graph.invoke({"messages": [initial_message]})

            # Log the full conversation without truncation
            if self.llm_logger:
                # Log each message in the conversation
                for i, msg in enumerate(result["messages"]):
                    msg_type = getattr(msg, "type", "unknown")

                    # Check if this is a tool call message
                    has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls

                    content_dict = {
                        "message_index": i,
                        "type": msg_type,
                        "content": str(msg.content),  # No truncation
                    }

                    if has_tool_calls:
                        content_dict["tool_calls"] = [
                            {
                                "name": tc.get("name"),
                                "args": tc.get("args", {}),
                                "id": tc.get("id"),
                            }
                            for tc in msg.tool_calls
                        ]

                    self.llm_logger.log_interaction(
                        interaction_type=f"autonomous_msg_{i}_{msg_type}",
                        messages=[{"role": msg_type, "content": str(content_dict)}],
                        response={
                            "content": f"Message {i} of {len(result['messages'])}",
                            "model": f"{self.config.provider_type}/{self.config.model}",
                        },
                        usage=None,
                        metadata={"total_messages": len(result["messages"])},
                    )

            # Extract proposals from final message
            final_message = result["messages"][-1]

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

            # Filter to requested number of proposals
            valid_proposals = proposals[:n_proposals]

            if len(valid_proposals) < n_proposals:
                print(f"⚠ Only {len(valid_proposals)}/{n_proposals} valid proposals")
                print(f"  (requested {n_to_generate}, got {len(proposals)} valid)")
            else:
                print(
                    f"Generated {len(proposals)} valid proposals. "
                    f"Executing top {len(valid_proposals)}..."
                )
            print()

            # Execute and narrate (same as linear workflow)
            labeled_hypotheses = []
            narrator = ResultNarrator(self.provider, logger=self.llm_logger)

            for i, proposal in enumerate(valid_proposals, 1):
                a_desc = proposal.cohort_a_description
                b_desc = proposal.cohort_b_description
                print(f"  [{i}/{len(proposals)}] Executing: {a_desc} vs {b_desc}")

                try:
                    start_time = datetime.utcnow()
                    result = self.executor.execute(proposal.cycl_spec)
                    execution_time = (datetime.utcnow() - start_time).total_seconds()

                    print(f"    Executed in {execution_time:.1f}s. Narrating...")

                    narrative = narrator.narrate(
                        spec=proposal.cycl_spec,
                        result=result,
                        proposal_rationale=proposal.rationale,
                    )

                    labeled = LabeledHypothesis(
                        hypothesis_id=str(uuid.uuid4()),
                        session_id=self.session.session_id,
                        proposal=proposal,
                        spec=proposal.cycl_spec,
                        result=result,
                        execution_time_seconds=execution_time,
                        narrative=narrative,
                        llm_model=f"{self.config.provider_type}/{self.config.model}",
                    )

                    self.store.save(labeled)
                    labeled_hypotheses.append(labeled)

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

            hypotheses = self.run_iteration()

            # Check if we got any hypotheses this iteration
            if not hypotheses:
                print(f"⚠ No hypotheses generated in iteration {i+1}, continuing...")
                print()
                continue

            all_hypotheses.extend(hypotheses)
            print(f"✓ Iteration {i+1} complete: {len(hypotheses)} hypotheses")
            print()

        print(f"Session complete. Generated {len(all_hypotheses)} hypotheses total.")
        return all_hypotheses

    def _extract_proposals(self, content: str) -> list[HypothesisProposal]:
        """Extract CyclHyp proposals from LLM response.

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
                "proposals", [data] if "cycl_spec" in data else []
            )

            proposals = []
            for i, item in enumerate(proposals_data, 1):
                proposal = HypothesisProposal(
                    cohort_a_description=item["cohort_a_description"],
                    cohort_b_description=item["cohort_b_description"],
                    outcome_description=item["outcome_description"],
                    rationale=item["rationale"],
                    cycl_spec=CyclHyp(**item["cycl_spec"]),
                )

                # Validate tables/columns exist before accepting
                validation_errors = self._validate_cycl_spec(proposal.cycl_spec)
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

    def _validate_cycl_spec(self, spec: CyclHyp) -> list[str]:
        """Validate that tables and columns in spec actually exist.

        Parameters
        ----------
        spec : CyclHyp
            Hypothesis specification to validate

        Returns
        -------
        list[str]
            List of validation error messages (empty if valid)
        """
        errors = []
        tables = self.db.list_tables()

        # Check cohort_a
        cohort_a = spec.query.cohort_a
        if cohort_a.table not in tables:
            errors.append(f"Table '{cohort_a.table}' does not exist")
        else:
            # Check column exists in table
            try:
                table_df = self.db.execute(f"SELECT * FROM {cohort_a.table} LIMIT 0")
                columns = table_df.columns.tolist()
                if cohort_a.column not in columns:
                    errors.append(f"Column '{cohort_a.column}' not in {cohort_a.table}")
            except Exception as e:
                errors.append(f"Error checking {cohort_a.table}: {str(e)}")

        # Check cohort_b
        cohort_b = spec.query.cohort_b
        if cohort_b.table not in tables:
            errors.append(f"Table '{cohort_b.table}' does not exist")
        else:
            try:
                table_df = self.db.execute(f"SELECT * FROM {cohort_b.table} LIMIT 0")
                columns = table_df.columns.tolist()
                if cohort_b.column not in columns:
                    errors.append(f"Column '{cohort_b.column}' not in {cohort_b.table}")
            except Exception as e:
                errors.append(f"Error checking {cohort_b.table}: {str(e)}")

        # Check outcome table/columns
        outcome = spec.query.outcome
        if outcome.table not in tables:
            errors.append(f"Outcome table '{outcome.table}' does not exist")
        else:
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

    def _format_previous_hypotheses(self, previous: list[LabeledHypothesis]) -> str:
        """Format previous hypotheses for context."""
        if not previous:
            return "**No previous hypotheses yet.**"

        lines = ["**Previously tested hypotheses (avoid duplicates)**:"]
        for i, hyp in enumerate(previous[:10], 1):
            a_desc = hyp.proposal.cohort_a_description
            b_desc = hyp.proposal.cohort_b_description
            desc = f"{a_desc} vs {b_desc}"
            lines.append(f"{i}. {desc}")

        if len(previous) > 10:
            lines.append(f"... and {len(previous) - 10} more")

        return "\n".join(lines)
