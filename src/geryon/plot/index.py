"""Write an index.html that shows every plot a run produced, grouped by subdir.

Each plot is shown from its SVG at its own size and links to its PDF. Only SVGs
modified at or after ``--since`` are listed, so plots a run no longer makes
(renamed modules, hand-made figures beside them) don't show up as current.
"""

import argparse
from collections import defaultdict
import html
from pathlib import Path

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Geryon plots</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 16px; background: #fafafa; }}
h2 {{ margin-top: 32px; border-bottom: 1px solid #ccc; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
figure {{ margin: 0; max-width: 100%; background: white; border: 1px solid #ddd;
  padding: 8px; box-sizing: border-box; }}
figcaption {{ font-family: monospace; font-size: 13px; margin-bottom: 4px; }}
figure img {{ display: block; max-width: 100%; height: auto; }}
</style></head><body>
<h1>Geryon plots</h1>
{sections}
</body></html>
"""


def _figure(svg: Path) -> str:
    src = html.escape(svg.as_posix())
    pdf = html.escape(svg.with_suffix(".pdf").as_posix())
    return (
        f'<figure><figcaption><a href="{pdf}">{html.escape(svg.stem)}</a>'
        f'</figcaption><img src="{src}" alt="{html.escape(svg.stem)}"></figure>'
    )


def build(output_dir: Path, since: float) -> str:
    groups: dict[str, list[Path]] = defaultdict(list)
    for svg in sorted(output_dir.rglob("*.svg")):
        if svg.stat().st_mtime >= since:
            groups[str(svg.parent.relative_to(output_dir))].append(svg)
    if not groups:
        raise SystemExit(f"no plots under {output_dir} newer than {since}")
    sections = []
    for group, svgs in sorted(groups.items()):
        figures = "".join(_figure(svg.relative_to(output_dir)) for svg in svgs)
        sections.append(
            f'<h2>{html.escape(group)}</h2><div class="grid">{figures}</div>'
        )
    return _PAGE.format(sections="\n".join(sections))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--since", required=True, type=float, help="Unix time")
    args = parser.parse_args()
    index = args.output_dir / "index.html"
    index.write_text(build(args.output_dir, args.since))
    print(index)


if __name__ == "__main__":
    main()
