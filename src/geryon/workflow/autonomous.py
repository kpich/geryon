"""Autonomous workflow using LangGraph for tool-based exploration."""

import contextlib
import json
from pathlib import Path
import threading
from typing import Annotated, TypedDict
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
from geryon.tools.derived import create_derived_view
from geryon.tools.volcano import scan_groupby
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
        self._derived_views_path = Path(config.storage_dir) / "derived_views.json"
        self._views_lock = threading.Lock()
        self._replay_derived_views()
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

        self._ranking_context: str | None = None
        self._ranking_top_ids: set[str] = set()

        # Create LangChain-compatible tools
        self.tools = self._create_tools()

        # Create LangChain LLM
        self.llm = self._create_llm()

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

        @tool
        def scan_groupby_tool(
            group_table: str,
            group_column: str,
            outcome_spec: str,
            method: str = "hazard_ratio_cox",
            min_group_size: int = 10,
            top_n: int = 20,
        ) -> str:
            """Scan all unique values of a categorical column, comparing each group vs
            the rest of patients for a clinical outcome. Returns the top hits ranked by
            significance with effect sizes — the data for a volcano plot.

            Args:
                group_table: Table containing the grouping variable (e.g.
                    'mutations_extended', 'clinical_patient')
                group_column: Column to scan (e.g. 'Hugo_Symbol', 'CANCER_TYPE')
                outcome_spec: JSON outcome descriptor, e.g.
                    '{"outcome_type": "overall_survival"}' or
                    '{"outcome_type": "time_to_next_treatment", "agent": "Sotorasib"}'
                method: 'hazard_ratio_cox' (time-to-event) or 'wilcoxon_rank_sum'
                    (continuous)
                min_group_size: Minimum patients per group after outcome filtering
                    (default 10)
                top_n: Number of top hits to return (default 20)
            """
            print(f"[TOOL] scan_groupby_tool({group_table}.{group_column}) called")
            result = scan_groupby(
                self.db,
                self.executor,
                group_table,
                group_column,
                outcome_spec,
                method=method,
                min_group_size=min_group_size,
                top_n=top_n,
                cache_dir=self.db.parquet_dir,
            )
            print(f"[TOOL] scan_groupby_tool done, {len(result)} chars")
            return result

        @tool
        def create_derived_view_tool(name: str, sql: str) -> str:
            """Create a persistent SQL view for complex derived concepts not expressible
            as simple column filters — e.g. treatment regimens, drug sequencing, or any
            multi-step join/aggregation.

            The view is registered as 'derived_{name}', saved to disk, and automatically
            restored in future sessions. Available immediately to query_data_tool,
            scan_groupby_tool, and as a filter table in hypotheses.

            IMPORTANT: Always include PATIENT_ID in the SELECT so the view can be used
            in cohort filters.

            Args:
                name: Short identifier (auto-prefixed with 'derived_')
                sql: SELECT or WITH (CTE) query defining the view
            """
            print(f"[TOOL] create_derived_view_tool({name!r}) called")
            result = create_derived_view(self.db, name, sql)
            if not result.startswith("ERROR"):
                safe_name = name if name.startswith("derived_") else f"derived_{name}"
                self._save_derived_view(safe_name, sql)
            print(f"[TOOL] create_derived_view_tool done: {result[:80]}")
            return result

        return [
            list_tables_tool,
            describe_table_tool,
            query_data_tool,
            scan_groupby_tool,
            create_derived_view_tool,
        ]

    def _replay_derived_views(self) -> None:
        """Recreate any derived views saved from prior sessions."""
        if not self._derived_views_path.exists():
            return
        try:
            defs: dict[str, str] = json.loads(self._derived_views_path.read_text())
            for name, sql in defs.items():
                try:
                    self.db.create_view(name, sql)
                except Exception as e:
                    print(f"  ⚠ Could not replay derived view '{name}': {e}")
        except Exception as e:
            print(f"  ⚠ Could not load derived_views.json: {e}")

    def _save_derived_view(self, name: str, sql: str) -> None:
        """Append or update a derived view definition on disk."""
        lock = getattr(self, "_views_lock", None) or threading.Lock()
        with lock:
            defs: dict[str, str] = {}
            if self._derived_views_path.exists():
                with contextlib.suppress(Exception):
                    defs = json.loads(self._derived_views_path.read_text())
            defs[name] = sql
            self._derived_views_path.write_text(json.dumps(defs, indent=2))

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

    def _make_submit_tool(self, iteration: int, submitted: list):
        """Create a submit_hypothesis_tool bound to this iteration's submitted list."""

        @tool
        def submit_hypothesis_tool(
            cohort_a_description: str,
            cohort_b_description: str,
            outcome_description: str,
            rationale: str,
            geryon_spec_json: str,
            refines_hypothesis: str | None = None,
        ) -> str:
            """Submit a hypothesis for immediate execution and storage.

            Call this when you have a well-supported hypothesis. You can call it
            multiple times during exploration. The tool executes the comparison
            immediately and returns the result so you can keep exploring.

            geryon_spec_json: JSON string with the full GeryonHyp spec.
            Returns: HR, p-value, and save confirmation, or an error message.
            """
            try:
                spec_dict = json.loads(geryon_spec_json)
                spec = GeryonHyp(**spec_dict)
            except Exception as e:
                return f"ERROR parsing spec: {e}"

            errors = self._validate_geryon_spec(spec)
            if errors:
                return f"REJECTED: {'; '.join(errors)}"

            proposal = HypothesisProposal(
                cohort_a_description=cohort_a_description,
                cohort_b_description=cohort_b_description,
                outcome_description=outcome_description,
                rationale=rationale,
                geryon_spec=spec,
                refines_hypothesis=refines_hypothesis,
            )

            try:
                result = self.executor.execute(spec)
                narrator = ResultNarrator(self.provider, logger=self.llm_logger)
                narrative = narrator.narrate(
                    spec=spec,
                    result=result,
                    proposal_rationale=rationale,
                    idx=len(submitted) + 1,
                )
                labeled = LabeledHypothesis(
                    hypothesis_id=str(uuid.uuid4()),
                    session_id=self.session.session_id,
                    proposal=proposal,
                    spec=spec,
                    result=result,
                    execution_time_seconds=0.0,
                    narrative=narrative,
                    llm_model=f"{self.config.provider_type}/{self.config.model}",
                    iteration=iteration,
                )
                self.store.save(labeled)
                submitted.append(labeled)
                if self.llm_logger:
                    self.llm_logger.log_proposal(idx=len(submitted), proposal=proposal)
                    self.llm_logger.log_execution(
                        idx=len(submitted), result=result, time_s=0.0
                    )
            except Exception as e:
                return f"ERROR executing hypothesis: {e}"

            hr = result.hazard_ratio
            p = result.p_value
            return (
                f"✓ Saved [{labeled.hypothesis_id[:8]}]: "
                f"HR={hr:.3f}, p={p:.3e}. {narrative.summary}"
            )

        return submit_hypothesis_tool

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
        from geryon.workflow.context import load_prior_hypotheses

        prior = load_prior_hypotheses(self.config.output_dir, self.config.session_id)
        session_previous = prior + self.store.load()
        prev_ctx = self._format_previous_hypotheses(labeled, session_previous)
        previous_context = prev_ctx.text
        if self._ranking_context:
            previous_context = previous_context + "\n\n" + self._ranking_context

        if self.llm_logger and iteration is not None:
            self.llm_logger.log_iteration_start(
                iteration=iteration,
                previous_ids=prev_ctx.rated_ids + prev_ctx.unrated_session_ids,
                n_rated=len(prev_ctx.rated_ids),
                n_unrated_session=len(prev_ctx.unrated_session_ids),
            )

        print(f"Generating {n_proposals} hypothesis(es)...")
        print(f"Using model: {self.config.provider_type}/{self.config.model}")
        print()

        # Build messages: static system prompt + dynamic user message
        from geryon.llm.prompts import PROPOSAL_SYSTEM_PROMPT
        from geryon.llm.schema_context import load_schema_context

        schema_ctx = load_schema_context(self.config.parquet_dir, self.db)
        system_content = PROPOSAL_SYSTEM_PROMPT
        if schema_ctx:
            system_content = system_content + "\n\n" + schema_ctx
        system_message = SystemMessage(content=system_content)
        user_text = (
            f"{previous_context}\n\n"
            f"Generate {n_proposals} hypothesis(es). "
            f"Follow the exploration steps described above before proposing."
        )
        user_message = HumanMessage(content=user_text)

        # Run LangGraph
        submitted: list[LabeledHypothesis] = []
        try:
            from langgraph.prebuilt import ToolNode, create_react_agent

            submit_tool = self._make_submit_tool(iteration or 0, submitted)
            tool_node = ToolNode(self.tools + [submit_tool], handle_tool_errors=True)
            graph = create_react_agent(self.llm, tool_node)

            if self.llm_logger:
                self.llm_logger._write(event="graph_invoke_start")

            result = graph.invoke(
                {"messages": [system_message, user_message]},
                config={"recursion_limit": 100},
            )

            if self.llm_logger:
                self.llm_logger._write(
                    event="graph_invoke_end",
                    n_messages=len(result["messages"]),
                    n_submitted=len(submitted),
                )

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

            print(
                f"LLM completed after {len(result['messages'])} messages, "
                f"{len(submitted)} hypotheses submitted"
            )

            if not submitted:
                print("⚠ No hypotheses submitted in this iteration.")

            return submitted

        except Exception as e:
            print("⚠ WARNING: Hypothesis generation failed")
            print(f"  Error: {type(e).__name__}: {str(e)}")

            import traceback

            print("  Full traceback:")
            traceback.print_exc()

            if submitted:
                print(
                    f"  Returning {len(submitted)} hypotheses submitted before failure."
                )
            return submitted

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
                try:
                    from geryon.llm.critic import HypothesisCritic

                    critic = HypothesisCritic(self.provider, logger=self.llm_logger)
                    for cycle in range(self.config.critic_cycles):
                        print(
                            f"  Critic cycle {cycle + 1}/{self.config.critic_cycles}..."
                        )
                        hypotheses = critic.rate(hypotheses)

                    all_session = self.store.load()
                    hyp_by_id = {h.hypothesis_id: h for h in hypotheses}
                    updated = [hyp_by_id.get(h.hypothesis_id, h) for h in all_session]
                    self.store.save_all(updated)
                except Exception as e:
                    print(
                        f"  ⚠ Critic failed: {type(e).__name__}: {e}"
                        " — continuing without ratings"
                    )

            all_hypotheses.extend(hypotheses)

            if self.config.rank_after_critic and self.config.critic_cycles > 0:
                try:
                    from geryon.llm.ranker import HypothesisRanker
                    from geryon.workflow.context import MAX_RATED

                    labeled_rated = [
                        h
                        for h in self.labeled_store.load_all()
                        if not h.effective_rating.is_pending
                    ]
                    session_rated = [
                        h for h in all_hypotheses if not h.effective_rating.is_pending
                    ]
                    # human-rated first, critic-rated second; most recent first per tier
                    seen: set[str] = set()
                    pool: list[LabeledHypothesis] = []
                    for h in list(reversed(labeled_rated)) + list(
                        reversed(session_rated)
                    ):
                        if h.hypothesis_id not in seen:
                            seen.add(h.hypothesis_id)
                            pool.append(h)
                    pool = pool[:MAX_RATED]
                    if pool:
                        ranker = HypothesisRanker(self.provider, logger=self.llm_logger)
                        rank_result = ranker.rank(
                            pool, priority_ids=self._ranking_top_ids
                        )
                        self._ranking_top_ids = set(
                            rank_result.top_general + rank_result.top_refinement
                        )
                        hyps_by_id = {h.hypothesis_id: h for h in pool}
                        self._ranking_context = ranker.format_for_prompt(
                            rank_result, hyps_by_id
                        )
                        print(
                            f"  Ranker: {len(rank_result.top_general)} general, "
                            f"{len(rank_result.top_refinement)} refinement candidates"
                        )
                except Exception as e:
                    print(
                        f"  ⚠ Ranker failed: {type(e).__name__}: {e}"
                        " — continuing without ranking"
                    )

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
        # ProgressionFromTreatment uses two tables (treatment_table/progression_table)
        # rather than a single `table` field; skip the single-table check for it.
        table = getattr(outcome, "table", None)
        time_col = getattr(outcome, "time_column", None)
        event_col = getattr(outcome, "event_column", None)
        if table is None:
            pass
        elif table not in tables:
            errors.append(f"Outcome table '{table}' does not exist")
        elif time_col is not None and event_col is not None:
            try:
                table_df = self.db.execute(f"SELECT * FROM {table} LIMIT 0")
                columns = table_df.columns.tolist()
                if time_col not in columns:
                    errors.append(f"Time column '{time_col}' not in {table}")
                if event_col not in columns:
                    errors.append(f"Event column '{event_col}' not in {table}")
            except Exception as e:
                errors.append(f"Error checking {table}: {str(e)}")

        return errors

    def _format_previous_hypotheses(
        self,
        labeled: list[LabeledHypothesis],
        session_previous: list[LabeledHypothesis],
    ):
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
