"""Linear workflow for hypothesis generation: propose → execute → narrate → store."""

from datetime import datetime
import uuid

from msk_cycl.db import Database
from msk_cycl.engine import HypothesisExecutor
from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.labeling.storage import HypothesisStore
from msk_cycl.llm.conversation_logger import ConversationLogger
from msk_cycl.llm.generator import HypothesisGenerator
from msk_cycl.llm.narrator import ResultNarrator
from msk_cycl.llm.provider import create_provider
from msk_cycl.llm.schema import discover_schema
from msk_cycl.workflow.session import Session, SessionConfig


class LinearWorkflow:
    """Simple linear workflow: propose → execute → narrate → store.

    This is v0 - a human-driven batch processing workflow. Future v1 will add
    MCP-based autonomous iteration where the LLM can call tools and iterate.
    """

    def __init__(self, config: SessionConfig):
        """Initialize workflow.

        Parameters
        ----------
        config : SessionConfig
            Session configuration
        """
        self.config = config
        self.session = Session(config)

        # Initialize components
        self.db = Database(config.parquet_dir)
        self.executor = HypothesisExecutor(self.db)
        self.provider = create_provider(
            config.provider_type,
            model=config.model,
        )

        # Schema discovery with progress bar
        print("Discovering database schema...")
        self.schema = discover_schema(self.db, show_progress=True)

        self.store = HypothesisStore(config.storage_dir)

        # Initialize LLM conversation logger
        self.llm_logger = None
        if config.enable_llm_logging:
            self.llm_logger = ConversationLogger(
                storage_dir=config.storage_dir,
                session_id=config.session_id,
            )

    def run_iteration(self, n_proposals: int | None = None) -> list[LabeledHypothesis]:
        """Run one iteration: generate N hypotheses, execute, narrate, store.

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

        print(f"Generating {n_proposals} hypotheses...")

        # 1. Generate proposals
        generator = HypothesisGenerator(
            provider=self.provider,
            schema=self.schema,
            previous_hypotheses=previous,
            logger=self.llm_logger,
        )
        proposals = generator.propose(n=n_proposals)

        print(f"Generated {len(proposals)} proposals. Executing...")

        # 2. Execute and narrate each proposal
        labeled_hypotheses = []
        narrator = ResultNarrator(self.provider, logger=self.llm_logger)

        for i, proposal in enumerate(proposals, 1):
            a_desc = proposal.cohort_a_description
            b_desc = proposal.cohort_b_description
            print(f"  [{i}/{len(proposals)}] Executing: {a_desc} vs {b_desc}")

            try:
                # Execute
                start_time = datetime.utcnow()
                result = self.executor.execute(proposal.cycl_spec)
                execution_time = (datetime.utcnow() - start_time).total_seconds()

                print(f"    Executed in {execution_time:.1f}s. Narrating...")

                # Narrate
                narrative = narrator.narrate(
                    spec=proposal.cycl_spec,
                    result=result,
                    proposal_rationale=proposal.rationale,
                )

                # Create labeled hypothesis (pending human review)
                labeled = LabeledHypothesis(
                    hypothesis_id=str(uuid.uuid4()),
                    session_id=self.session.session_id,
                    proposal=proposal,
                    spec=proposal.cycl_spec,
                    result=result,
                    execution_time_seconds=execution_time,
                    narrative=narrative,
                    llm_model=self.provider.model_id(),
                )

                # Save to storage
                self.store.save(labeled)
                labeled_hypotheses.append(labeled)

                print(f"    Saved. Summary: {narrative.summary}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        return labeled_hypotheses

    def run_full_session(self) -> list[LabeledHypothesis]:
        """Run full session with multiple iterations.

        Returns
        -------
        list[LabeledHypothesis]
            All hypotheses generated in session
        """
        all_hypotheses = []

        print(f"Starting session {self.session.session_id}")
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"Proposals per iteration: {self.config.num_proposals_per_iteration}")
        print()

        for i in range(self.config.max_iterations):
            print(f"=== Iteration {i+1}/{self.config.max_iterations} ===")

            hypotheses = self.run_iteration()
            all_hypotheses.extend(hypotheses)

            # Early stopping if no new hypotheses generated
            if len(hypotheses) == 0:
                print("No new hypotheses generated. Stopping early.")
                break

            print()

        print(f"Session complete. Generated {len(all_hypotheses)} hypotheses total.")
        return all_hypotheses
