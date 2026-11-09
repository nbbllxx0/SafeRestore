"""Mark bibliography entries added or revised since TIM-26-11920 in the change colour.

BibTeX regenerates main.bbl on every run, so run this after each bibtex
pass and before the final pdflatex passes:

    pdflatex main; bibtex main; python scripts/colour_new_refs.py main.bbl
    pdflatex main; pdflatex main

Each new entry is preceded by \\chgblock and every other entry by
\\normalcolor, so an entry's label and text share one colour and the colour
cannot leak into the next entry.  The script is idempotent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NEW_KEYS = {
    "berger1982iut",
    "cao2026m2ad",
    "gorthi2010fringe",
    "holm1979",
    "wang2026lec",
    "hao2026racer",
    "ren2022defectvision",
    "wiens2003fallback",
    "zuo2018phaseshift",
    "zuo2022dloptical",
}

MARK = r"(?:\\chgblock |\\normalcolor )?"
ITEM = re.compile(rf"^{MARK}\\bibitem\{{([^}}]+)\}}\n{MARK}", re.M)


def main(path: str) -> None:
    bbl = Path(path)
    text = bbl.read_text(encoding="utf-8")
    seen: set[str] = set()

    def mark(m: re.Match) -> str:
        key = m.group(1)
        seen.add(key)
        prefix = "\\chgblock " if key in NEW_KEYS else "\\normalcolor "
        return f"{prefix}\\bibitem{{{key}}}\n"

    text = ITEM.sub(mark, text)
    if not seen:
        raise SystemExit(f"no \\bibitem entries found in {bbl}")
    bbl.write_text(text, encoding="utf-8")
    coloured = sorted(NEW_KEYS & seen)
    print(f"{bbl}: coloured {len(coloured)} new of {len(seen)} entries; "
          f"not cited: {sorted(NEW_KEYS - seen)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main.bbl")
