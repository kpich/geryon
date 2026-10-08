"""Write the current figure as a PDF and an SVG beside it.

The PDF is the figure for papers. The SVG is what ``index`` puts on the overview
page, since a browser shows SVG as an image at its own size but shows PDF in a
viewer that crops it to a fixed frame.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def save(output: Path) -> None:
    for path in (output, output.with_suffix(".svg")):
        plt.savefig(path, bbox_inches="tight", transparent=True)
