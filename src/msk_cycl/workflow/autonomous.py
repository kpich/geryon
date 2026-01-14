"""Autonomous workflow using LangGraph for tool-based exploration."""

from datetime import datetime
import json
import re
from typing import Annotated, Literal, TypedDict
import uuid

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

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
            from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                model=self.config.model,
                base_url="http://localhost:11434",
                format="json",  # Force JSON output for Ollama
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
                model=self.config.model,
                temperature=0.8,
            )
        else:
            raise ValueError(f"Unknown provider type: {self.config.provider_type}")

    def _build_graph(self):
        """Build LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

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
            return "continue"

        # Otherwise, end
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

        # Load previous hypotheses to avoid duplicates
        previous = self.store.load_session(self.session.session_id)
        previous_context = self._format_previous_hypotheses(previous)

        print(f"Generating {n_proposals} hypothesis(es) with autonomous exploration...")
        print()

        # Create initial prompt
        prompt_text = f"""Propose {n_proposals} novel hypothesis(es) for comparing \
patient cohorts on survival.

{previous_context}

**Process**:
1. Use list_tables_tool to see available data
2. Use describe_table_tool to understand key tables (e.g., clinical_patient, CNA)
3. Use query_data_tool to explore interesting features
4. Propose hypothesis(es) using ONLY tables/columns you've verified exist

**Output format** (JSON):
{{
  "proposals": [
    {{
      "cohort_a_description": "Patients with KRAS amplification",
      "cohort_b_description": "Patients without KRAS amplification",
      "outcome_description": "Overall survival",
      "rationale": "KRAS amplifications are common in cancer and may affect prognosis",
      "cycl_spec": {{
        "cohort_a": {{"table": "CNA", "column": "KRAS", "operator": ">", "value": 0}},
        "cohort_b": {{"table": "CNA", "column": "KRAS", "operator": "==", "value": 0}},
        "outcome": {{
          "table": "clinical_patient",
          "time_column": "OS_MONTHS",
          "event_column": "OS_STATUS"
        }}
      }}
    }}
  ]
}}

Start by exploring the database to understand what data is available."""
        initial_message = HumanMessage(content=prompt_text)

        # Run LangGraph
        try:
            result = self.graph.invoke({"messages": [initial_message]})

            # Extract proposals from final message
            final_message = result["messages"][-1]
            proposals = self._extract_proposals(final_message.content)

            if not proposals:
                print("⚠ WARNING: No valid proposals extracted from LLM response")
                print("  Continuing to next iteration...")
                return []

            print(f"Generated {len(proposals)} proposals. Executing...")
            print()

            # Execute and narrate (same as linear workflow)
            labeled_hypotheses = []
            narrator = ResultNarrator(self.provider, logger=self.llm_logger)

            for i, proposal in enumerate(proposals, 1):
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
            for item in proposals_data:
                proposals.append(
                    HypothesisProposal(
                        cohort_a_description=item["cohort_a_description"],
                        cohort_b_description=item["cohort_b_description"],
                        outcome_description=item["outcome_description"],
                        rationale=item["rationale"],
                        cycl_spec=CyclHyp(**item["cycl_spec"]),
                    )
                )

            return proposals

        except Exception as e:
            print("⚠ WARNING: Failed to extract proposals from LLM response")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            print(f"  Response preview: {content[:200]}...")
            return []

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
