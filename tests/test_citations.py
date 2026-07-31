"""Citation stream parsing.

The fiddliest logic in the codebase: a character-level state machine that has
to strip [[Source: ...]] blocks from a token stream where a single citation can
be split across arbitrary token boundaries.
"""

from __future__ import annotations

import pytest

from core.citations import CitationStreamParser, consolidate_citations


def run(tokens):
    """Feed tokens through the parser; return (visible_text, sources, rendered)."""
    parser = CitationStreamParser()
    visible = "".join(parser.feed(t) for t in tokens)
    visible += parser.finish()
    return visible, parser.as_dict(), parser.formatted()


def test_plain_text_passes_through_unchanged():
    visible, sources, _ = run(["Revenue grew by 12 percent."])
    assert visible == "Revenue grew by 12 percent."
    assert sources == {}


def test_citation_is_stripped_and_collected():
    visible, sources, rendered = run(["Revenue was $5.2M. [[Source: report.pdf | Page: 3]]"])
    assert visible == "Revenue was $5.2M. "
    assert sources == {"report.pdf": [3]}
    assert rendered == "[Source: report.pdf | Page: 3]"


def test_citation_split_across_token_boundaries():
    """The case that actually happens in production — models emit tiny fragments."""
    tokens = ["Revenue was ", "$5.2M", ". ", "[", "[Sou", "rce: rep", "ort.pdf | ", "Page: 1,3", "]]"]
    visible, sources, rendered = run(tokens)
    assert visible == "Revenue was $5.2M. "
    assert sources == {"report.pdf": [1, 3]}
    assert rendered == "[Source: report.pdf | Page: 1,3]"


def test_citation_split_one_character_per_token():
    """Worst case: every character arrives separately."""
    text = "Total was 9. [[Source: a.pdf | Page: 2]]"
    visible, sources, _ = run(list(text))
    assert visible == "Total was 9. "
    assert sources == {"a.pdf": [2]}


def test_double_brackets_that_are_not_citations_survive_as_text():
    visible, sources, _ = run(["See [[not a citation]] here."])
    assert visible == "See [[not a citation]] here."
    assert sources == {}


def test_single_brackets_are_untouched():
    visible, _, _ = run(["The array[0] and [note] stay put."])
    assert visible == "The array[0] and [note] stay put."


def test_unterminated_citation_is_dropped_but_answer_is_kept():
    """A model that opens a citation and stops must not eat the answer."""
    visible, sources, _ = run(["Answer text. [[Source: x.pdf | Pag"])
    assert visible == "Answer text. "
    assert sources == {}


def test_trailing_single_bracket_is_flushed():
    visible, _, _ = run(["ends with ["])
    assert visible == "ends with ["


def test_pages_merge_and_deduplicate_across_citations():
    visible, sources, rendered = run([
        "A[[Source: a.pdf | Page: 2]]"
        "B[[Source: a.pdf | Page: 1,2]]"
        "C[[Source: b.pdf | Page: 9]]"
    ])
    assert visible == "ABC"
    assert sources == {"a.pdf": [1, 2], "b.pdf": [9]}
    assert rendered == "[Source: a.pdf | Page: 1,2] [Source: b.pdf | Page: 9]"


def test_non_numeric_pages_are_ignored():
    _, sources, _ = run(["x[[Source: a.pdf | Page: 1, ,3]]"])
    assert sources == {"a.pdf": [1, 3]}


def test_no_citations_renders_empty_string():
    _, _, rendered = run(["Nothing to cite."])
    assert rendered == ""


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 8, 13, 50])
def test_output_is_identical_regardless_of_chunking(chunk_size):
    """Tokenisation must not change the result — the whole point of buffering."""
    text = (
        "Revenue rose to $5,200,000 [[Source: fy23.pdf | Page: 12]] while costs "
        "fell [[Source: fy23.pdf | Page: 14]]. An array[3] is not a citation."
    )
    tokens = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    visible, sources, _ = run(tokens)

    assert "[[Source:" not in visible
    assert "array[3]" in visible
    assert sources == {"fy23.pdf": [12, 14]}


def test_consolidate_citations_moves_them_to_one_trailing_line():
    result = consolidate_citations(
        "Rev was 5M [[Source: r.pdf | Page: 3]] and up [[Source: r.pdf | Page: 4]]."
    )
    assert "[[Source:" not in result
    assert result.endswith("[Source: r.pdf | Page: 3,4]")
