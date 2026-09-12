"""Map word positions in the cleaned filing back to printed page numbers.

The cleaned text keeps the filing's printed page numbers as standalone lines,
but so do hundreds of table figures. Genuine page footers are the ones that
ascend through the document, so we recover them as the longest strictly
increasing subsequence of candidate numbers and discard the rest.
"""

import bisect
from pathlib import Path

from config import CLEANED_TEXT

MAX_PAGE = 300


def _candidate_markers(lines):
    """Every standalone number that could plausibly be a page footer."""
    markers = []

    for line_index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.isdigit() and 1 <= int(stripped) <= MAX_PAGE:
            markers.append((line_index, int(stripped)))

    return markers


def _longest_increasing(markers):
    """Keep only the markers that ascend, i.e. the real page footers."""
    pages = [page for _, page in markers]

    tails = []
    tail_positions = []
    previous = [-1] * len(pages)

    for position, page in enumerate(pages):
        slot = bisect.bisect_left(tails, page)

        if slot == len(tails):
            tails.append(page)
            tail_positions.append(position)
        else:
            tails[slot] = page
            tail_positions[slot] = position

        previous[position] = tail_positions[slot - 1] if slot > 0 else -1

    sequence = []
    position = tail_positions[-1]

    while position != -1:
        sequence.append(markers[position])
        position = previous[position]

    sequence.reverse()

    return sequence


def build_page_map(text):
    """Return (word_offsets, pages): the word index at which each page ends.

    A chunk covering words [start, end) sits on the pages whose offsets span
    that range, so callers can label chunks by bisecting word_offsets.
    """
    lines = text.splitlines()
    footers = _longest_increasing(_candidate_markers(lines))

    # Convert line positions into word positions, since chunking works in words.
    words_before_line = []
    running_total = 0

    for line in lines:
        words_before_line.append(running_total)
        running_total += len(line.split())

    word_offsets = [words_before_line[line_index] for line_index, _ in footers]
    pages = [page for _, page in footers]

    return word_offsets, pages


def pages_for_span(word_offsets, pages, start, end):
    """Printed pages covered by the word range [start, end)."""
    if not pages:
        return []

    first = bisect.bisect_left(word_offsets, start)
    last = bisect.bisect_left(word_offsets, max(end - 1, start))

    first = min(first, len(pages) - 1)
    last = min(last, len(pages) - 1)

    # last >= first always, so this slice is never empty.
    return sorted(set(pages[first:last + 1]))


def load_page_map(path=CLEANED_TEXT):
    return build_page_map(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    word_offsets, pages = load_page_map()

    print(f"Recovered {len(pages)} printed page markers.")
    print(f"Page range: {pages[0]} to {pages[-1]}")
