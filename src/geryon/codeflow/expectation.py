"""What effect an analysis was expected to report, asked without showing its result.

A null is interesting only where an effect was expected, and an effect only where
none was. Recording the expectation lets nulls and effects sit on one scale: how far
the result landed from it (see ``plot.holdout_table.surprise``).

The expectation must be given blind, or hindsight pulls it toward the result. The
generator's title and description usually state the result, so two calls are made:
the first restates the analysis as a neutral question, and the second sees only that
question and the effect measure. The prompts are fixed here rather than in the prompt
set because this is a measurement: expectations are comparable only if every session,
and any backfill of old hypotheses, asks the same way.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from geryon.codeflow.chat import (
    MessageUsage,
    parse_reply,
    reply_text,
    sum_message_usage,
)
from geryon.codeflow.models import Expectation

QUESTION_PROMPT = """\
You restate a completed analysis as the research question it answers, so that \
someone can be asked what result they expect without being told the result.

Name the cohort, the comparison (exposure and reference group, or the predictor \
and its units), the outcome or endpoint with its time zero, any adjustment, and the \
effect measure with the direction of its contrast (e.g. "hazard ratio, mutated vs \
wild-type"). Keep the specifics that define the analysis.

Leave out everything that reveals or hints at the result: its direction, size, \
significance or interval, and wording such as "predicts", "worse", "protective", \
"no effect", "concordant" or "resolves". If the text reports several results, ask the \
question for the one the effect measure names.

Return ONLY JSON: {"question": "..."}"""

EXPECT_PROMPT = """\
You are an oncologist and cancer-genomics researcher. You are given a research \
question that an analysis of the MSK-IMPACT cohort answers: real-world patients at \
one cancer center, with tumor sequencing, treatment and outcome records. You have not \
seen the analysis or its result.

Say what value of the effect measure you expect the analysis to report, from what you \
know of the biology, the clinical literature and how observational analyses like this \
tend to come out. Give your best guess and an 80% interval: you would be surprised if \
the reported value fell outside it.

Return ONLY JSON, on the scale of the effect measure: \
{"effect": <number>, "lower": <number>, "upper": <number>}"""


class _Question(BaseModel):
    question: str


class _Guess(BaseModel):
    effect: float
    lower: float
    upper: float


def elicit_expectation(
    llm: BaseChatModel, *, title: str, description: str, effect_size_type: str | None
) -> tuple[Expectation, MessageUsage]:
    """The expected effect, with the token usage of both calls."""
    measure = effect_size_type or "(unspecified)"
    q_reply = llm.invoke(
        [
            SystemMessage(content=QUESTION_PROMPT),
            HumanMessage(
                content=f"# TITLE\n{title}\n\n# DESCRIPTION\n{description}\n\n"
                f"# EFFECT MEASURE\n{measure}"
            ),
        ]
    )
    question = parse_reply(
        reply_text(q_reply, "expectation question"), _Question, "expectation question"
    ).question

    e_reply = llm.invoke(
        [
            SystemMessage(content=EXPECT_PROMPT),
            HumanMessage(
                content=f"# QUESTION\n{question}\n\n# EFFECT MEASURE\n{measure}"
            ),
        ]
    )
    guess = parse_reply(reply_text(e_reply, "expectation"), _Guess, "expectation")
    try:
        expectation = Expectation(
            question=question,
            effect=guess.effect,
            lower=guess.lower,
            upper=guess.upper,
            model=e_reply.response_metadata.get("model_name"),
        )
    except ValueError as e:
        raise ValueError(f"expectation returned a disordered interval: {guess}") from e
    return expectation, sum_message_usage([q_reply, e_reply])
