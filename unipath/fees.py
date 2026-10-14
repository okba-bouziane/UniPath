"""Parse the fee tables out of the university markdown files."""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from unipath import config


@dataclass
class FeeRow:
    university: str
    programme: str
    tuition_gbp: int
    deposit_gbp: int
    source: str


def _university_name(text: str, fallback: str) -> str:
    """'# Northbridge University — International ...' -> 'Northbridge University'."""
    match = re.search(r"^#\s+(.+?)(?:\s+[—-]\s+.*)?$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def _parse_money(cell: str) -> int:
    return int(re.sub(r"[^\d]", "", cell) or 0)


def _fees_section(text: str) -> str:
    match = re.search(r"^##\s+Fees.*?$(.*?)(?=^##\s|\Z)", text, flags=re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


def load_fee_table(data_dir: Path = config.DATA_DIR) -> list[FeeRow]:
    """Return one FeeRow per programme for every file that has a '## Fees' table."""
    rows: list[FeeRow] = []
    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        section = _fees_section(text)
        if not section:
            continue
        name = _university_name(text, path.stem)
        for line in section.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3 or not re.search(r"\d", cells[1]):
                continue  # header, separator or non-table line
            rows.append(
                FeeRow(name, cells[0], _parse_money(cells[1]), _parse_money(cells[2]), path.name)
            )
    return rows


def _key(name: str) -> str:
    """Short lowercase key used for matching, e.g. 'University of Eastmere' -> 'eastmere'."""
    stop = {"university", "of", "the", "london"}
    words = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in stop]
    return " ".join(words)


def match_universities(queries: list[str], known: list[str]) -> tuple[list[str], list[str]]:
    """Map user-typed names to known university names. Returns (matched, unknown)."""
    keys = {_key(name): name for name in known}
    matched, unknown = [], []
    for query in queries:
        q = _key(query)
        hit = next((name for key, name in keys.items() if q and (key in q or q in key)), None)
        if hit is None:
            close = difflib.get_close_matches(q, list(keys), n=1, cutoff=0.75)
            hit = keys[close[0]] if close else None
        if hit is None:
            unknown.append(query)
        elif hit not in matched:
            matched.append(hit)
    return matched, unknown
