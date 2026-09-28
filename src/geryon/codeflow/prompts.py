"""System prompts for the code-first loop.

The prose lives in a *prompt set*: a directory holding one markdown file per template
in :data:`TEMPLATE_FILES`. ``prompt_sets/default/`` ships with the package; any other
set is a directory given by path (copy the default and edit it). The session resolves
its set once, and the texts are stored in its ``config.json``, so every hypothesis can
be traced to the exact prompts that produced it.
"""

from pathlib import Path

from pydantic import BaseModel

DEFAULT_PROMPT_SET = "default"
_BUILTIN_SETS_DIR = Path(__file__).parent / "prompt_sets"

# In generator_user.md; replaced by the prior-hypothesis list. A template without it
# shows the generator no prior hypotheses (get_script can still fetch them).
PREVIOUS_HYPOTHESES_PLACEHOLDER = "{previous_hypotheses}"

TEMPLATE_FILES = {
    "generator": "generator.md",
    "generator_user": "generator_user.md",
    "critic": "critic.md",
    "narrator": "narrator.md",
}


class PromptSet(BaseModel):
    """The resolved prompt texts a session runs with."""

    name: str
    generator: str
    generator_user: str
    critic: str
    narrator: str

    @property
    def shows_previous_hypotheses(self) -> bool:
        return PREVIOUS_HYPOTHESES_PLACEHOLDER in self.generator_user

    def render_generator_user(self, previous_hypotheses: str) -> str:
        return self.generator_user.replace(
            PREVIOUS_HYPOTHESES_PLACEHOLDER, previous_hypotheses
        )


def load_prompt_set(spec: str | Path | None = None) -> PromptSet:
    """Load ``default`` (or None) from the package, or any other set by directory path.

    A custom set must contain every template: filling gaps from the default would make
    a session's prompts depend on a set the config doesn't name.
    """
    if spec is None or str(spec) == DEFAULT_PROMPT_SET:
        directory, name = _BUILTIN_SETS_DIR / DEFAULT_PROMPT_SET, DEFAULT_PROMPT_SET
    else:
        directory = Path(spec).expanduser().resolve()
        name = str(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Prompt set {spec!r}: no directory at {directory}")
    missing = [f for f in TEMPLATE_FILES.values() if not (directory / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Prompt set {directory} is missing {', '.join(missing)}. Copy "
            f"{_BUILTIN_SETS_DIR / DEFAULT_PROMPT_SET} and edit it."
        )
    texts = {k: (directory / f).read_text() for k, f in TEMPLATE_FILES.items()}
    return PromptSet(name=name, **texts)


DATA_FACTS_BLOCK = """

# Verified facts about this data

Each fact was recorded by an earlier agent together with a script that asserts it
against this data version, and the script passed. Rely on them instead of re-deriving
them. They describe the data, not results about it. If you show one is wrong, record the correction with
`record_data_fact(..., supersedes=<id>)`.

{facts}
"""

FOCUS_BLOCK = """

# Research focus

This session is a focused line of investigation, not free exploration. What follows
constrains what counts as a good hypothesis here; treat it as binding.

{focus}
{note}"""

CRITIC_FOCUS_NOTE = """
Judge `novelty` relative to this focus: a hypothesis that is only newly *possible*
under this focus counts as novel, not as a well-known result.
"""


def with_data_facts(base: str, facts: str | None) -> str:
    """Append the verified data facts; unchanged when there are none.

    Goes before the focus, which is the more specific of the two.
    """
    if not facts or not facts.strip():
        return base
    return base + DATA_FACTS_BLOCK.format(facts=facts.strip())


def with_focus(base: str, focus: str | None, note: str = "") -> str:
    """Append a session's focus to a system prompt; unchanged when there is no focus.

    Goes on the system message rather than the user message because that is where the
    cache breakpoint sits (see geryon.llm.caching) and the focus is session-static,
    while the user message already varies per iteration with the prior-hypothesis list.
    """
    if not focus or not focus.strip():
        return base
    return base + FOCUS_BLOCK.format(focus=focus.strip(), note=note)
