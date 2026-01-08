"""Autonomous workflow using MCP (Model Context Protocol).

STUB FOR FUTURE IMPLEMENTATION

This module documents the design for MCP-based autonomous hypothesis generation.
The execution engine (HypothesisExecutor) is already MCP-ready as a pure function.

## MCP Integration Strategy

The current execution engine is designed for MCP:
- Input: CyclHyp (structured Pydantic spec)
- Output: ComparisonResult (structured results)
- Pure function: no side effects, no mutations

### MCP Tools to Expose

1. **execute_hypothesis(spec: CyclHyp) -> ComparisonResult**
   - Executes a hypothesis and returns results
   - LLM can call this tool with generated specs

2. **get_schema() -> DatabaseSchema**
   - Returns available tables and columns
   - LLM can query database structure

3. **get_previous_hypotheses(limit: int) -> list[LabeledHypothesis]**
   - Returns previously tested hypotheses
   - LLM can avoid duplicates and build on findings

### LLM Capabilities with MCP

With these tools, the LLM can autonomously:
- Query database schema on demand
- Propose hypotheses
- Execute via tool call
- Analyze results
- Iterate based on findings (e.g., refine cohort definitions)
- Build hypothesis graphs/trees
- Follow up on interesting results

### Example Autonomous Flow

```
1. LLM: get_schema() → sees available data
2. LLM: Proposes hypothesis comparing KRAS mutant vs wildtype
3. LLM: execute_hypothesis(spec) → gets results (HR=1.23, p=0.045)
4. LLM: Analyzes - significant result, worth exploring
5. LLM: Proposes refined hypothesis (KRAS + specific cancer type)
6. LLM: execute_hypothesis(refined_spec) → gets refined results
7. LLM: Compares results, generates narrative
8. Repeat...
```

### Difference from LinearWorkflow

- **LinearWorkflow** (current): Human-driven batch processing
  - LLM proposes N hypotheses upfront
  - All executed in batch
  - Human reviews results offline

- **AutonomousWorkflow** (future): LLM-driven exploration
  - LLM iteratively proposes and executes
  - Adjusts strategy based on intermediate results
  - Builds hypothesis trees
  - More sample-efficient (focuses on promising directions)

### Implementation Notes

When implementing:
1. Create MCP server exposing HypothesisExecutor methods
2. Provide LLM with tool schemas
3. Add conversation loop with tool call handling
4. Track hypothesis DAG (not just linear sequence)
5. Add termination criteria (budget, convergence, etc.)
"""

from msk_cycl.workflow.session import SessionConfig


class AutonomousWorkflow:
    """MCP-based autonomous hypothesis generation (NOT YET IMPLEMENTED)."""

    def __init__(self, config: SessionConfig):
        """Initialize autonomous workflow.

        Parameters
        ----------
        config : SessionConfig
            Session configuration

        Raises
        ------
        NotImplementedError
            Always - this is a stub
        """
        raise NotImplementedError(
            "Autonomous MCP workflow not yet implemented. "
            "See module docstring for design notes. "
            "Use LinearWorkflow for current functionality."
        )
