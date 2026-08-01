"""Triage agent logic.

The model call is never made here. What is tested is everything around it: the
reporting decision, the reproduction convention, prompt assembly, and the
parsing that apply_fix.py depends on. Those are where a bug would either flood
the issue tracker with noise or silently drop a real finding.
"""

from __future__ import annotations

import pytest

from security import apply_fix
from security.github_api import MARKER_PREFIX, marker_for
from security.triage_agent import (
    Analysis,
    LLMError,
    ReproResult,
    _parse_json,
    _strip_fences,
    build_prompt,
    issue_body,
    issue_title,
    read_code_context,
)


def make_alert(number=7, rule_id="py/sql-injection", path="server/app.py", line=42):
    return {
        "number": number,
        "rule": {
            "id": rule_id,
            "name": "SQL injection",
            "security_severity_level": "high",
            "full_description": "User input flows into a query.",
        },
        "tool": {"name": "CodeQL"},
        "most_recent_instance": {
            "location": {"path": path, "start_line": line, "end_line": line},
            "message": {"text": "This query depends on user input."},
        },
    }


def make_analysis(**overrides):
    base = {
        "verdict": "exploitable",
        "confidence": 0.9,
        "severity": "high",
        "what": "The session token signature is not checked. Anyone can forge one.",
        "why": "The comparison uses == on a truncated digest.",
        "impact": "An attacker can impersonate any session and spend the API budget.",
        "attack_steps": ["Mint a token with any sid", "Send it as a Bearer header"],
        "repro_test": "def test_forged_token_is_accepted():\n    assert True\n",
        "repro_explanation": "Passing means a forged signature was accepted.",
        "fix_explanation": "Use hmac.compare_digest against the full signature.",
        "patch": "--- a/server/guards.py\n+++ b/server/guards.py\n@@\n-bad\n+good\n",
        "false_positive_reason": "",
    }
    base.update(overrides)
    return Analysis.from_json(base)


# ── Analysis parsing ─────────────────────────────────────────────────────────

def test_analysis_survives_a_sparse_reply():
    """A model that omits fields must not crash the run."""
    analysis = Analysis.from_json({})
    assert analysis.verdict == "needs_human"      # fails safe, not silent
    assert analysis.confidence == 0.0
    assert analysis.attack_steps == []
    assert analysis.repro_test == ""


def test_verdict_is_normalised():
    assert Analysis.from_json({"verdict": "  EXPLOITABLE "}).verdict == "exploitable"


# ── JSON extraction ──────────────────────────────────────────────────────────

def test_plain_json_is_parsed():
    assert _parse_json('{"verdict": "exploitable"}')["verdict"] == "exploitable"


def test_fenced_json_is_parsed():
    assert _parse_json('```json\n{"verdict": "low"}\n```')["verdict"] == "low"


def test_json_with_surrounding_prose_is_parsed():
    text = 'Here is my analysis:\n{"verdict": "not_exploitable"}\nHope that helps.'
    assert _parse_json(text)["verdict"] == "not_exploitable"


def test_unparseable_reply_raises_clearly():
    with pytest.raises(LLMError, match="was not JSON"):
        _parse_json("I could not analyse this.")


# ── Fence stripping ──────────────────────────────────────────────────────────

def test_python_fence_is_stripped():
    assert _strip_fences("```python\ndef f():\n    pass\n```") == "def f():\n    pass"


def test_unfenced_code_is_untouched():
    assert _strip_fences("def f():\n    pass") == "def f():\n    pass"


# ── Markers (deduplication across runs) ──────────────────────────────────────

def test_marker_is_stable_and_identifiable():
    marker = marker_for(7, "py/sql-injection")
    assert marker.startswith(MARKER_PREFIX)
    assert marker == marker_for(7, "py/sql-injection")


def test_markers_differ_per_alert():
    assert marker_for(7, "py/x") != marker_for(8, "py/x")
    assert marker_for(7, "py/x") != marker_for(7, "py/y")


def test_issue_body_embeds_the_marker_so_reruns_skip_it():
    body = issue_body(make_alert(), make_analysis(), ReproResult(True, True, "1 passed"))
    assert marker_for(7, "py/sql-injection") in body


# ── Issue rendering ──────────────────────────────────────────────────────────

def test_confirmed_finding_says_it_was_reproduced():
    body = issue_body(
        make_alert(), make_analysis(),
        ReproResult(True, True, "1 passed", "tests/security/test_alert_7.py"),
    )
    assert "**Reproduced.**" in body
    assert "tests/security/test_alert_7.py" in body
    assert "approved-fix" in body                      # tells the reader how to approve


def test_unreproduced_finding_is_labelled_unverified():
    body = issue_body(make_alert(), make_analysis(), ReproResult(True, False, "1 failed"))
    assert "**Not reproduced.**" in body
    assert "unverified" in body


def test_untested_finding_says_so():
    body = issue_body(
        make_alert(),
        make_analysis(repro_test=""),
        ReproResult(False, False, "No reproduction test was produced"),
    )
    assert "**Not tested.**" in body


def test_body_covers_what_why_impact_and_fix():
    """The four things the issue is supposed to answer."""
    body = issue_body(make_alert(), make_analysis(), ReproResult(True, True, "ok"))
    assert "## What is wrong" in body
    assert "## Why it is there" in body
    assert "## What it lets someone do" in body
    assert "## How to fix it" in body


def test_patch_is_rendered_as_an_applyable_diff_block():
    body = issue_body(make_alert(), make_analysis(), ReproResult(True, True, "ok"))
    assert "```diff" in body
    # apply_fix.py must be able to read back exactly what was written.
    assert apply_fix.extract_patch(body) is not None


def test_missing_patch_is_stated_plainly():
    body = issue_body(make_alert(), make_analysis(patch=""), ReproResult(True, True, "ok"))
    assert "by hand" in body
    assert apply_fix.extract_patch(body) is None


def test_title_is_short_and_names_the_file():
    title = issue_title(make_alert(), make_analysis())
    assert title.startswith("[security]")
    assert "app.py" in title
    assert len(title) < 140


# ── Round-trip: what triage writes, apply_fix reads ──────────────────────────

def test_apply_fix_can_extract_everything_it_needs():
    """A mismatch here would break the approval flow silently."""
    body = issue_body(
        make_alert(number=31),
        make_analysis(),
        ReproResult(True, True, "1 passed", "tests/security/test_alert_31.py"),
    )

    assert apply_fix.extract_alert_number(body) == 31
    assert apply_fix.extract_repro_path(body) == "tests/security/test_alert_31.py"
    patch = apply_fix.extract_patch(body)
    assert patch is not None and patch.startswith("--- a/")


def test_extractors_return_none_on_an_unrelated_issue():
    body = "Someone filed this by hand. No patch, no marker."
    assert apply_fix.extract_patch(body) is None
    assert apply_fix.extract_alert_number(body) is None
    assert apply_fix.extract_repro_path(body) is None


# ── Prompt assembly ──────────────────────────────────────────────────────────

def test_prompt_includes_the_threat_model_and_the_code():
    prompt = build_prompt(make_alert(path="server/guards.py", line=60))
    # Without the threat model the agent cannot tell a real finding from a
    # pattern match, which is the whole reason it exists.
    assert "Threat model" in prompt
    assert "OpenAI API key" in prompt
    assert "server/guards.py" in prompt
    assert "py/sql-injection" in prompt


def test_code_context_marks_the_flagged_line():
    context = read_code_context(make_alert(path="server/guards.py", line=60))
    assert ">>" in context
    assert "server/guards.py" in context


def test_missing_file_is_reported_not_crashed():
    context = read_code_context(make_alert(path="does/not/exist.py"))
    assert "not found" in context


def test_alert_without_a_location_is_handled():
    alert = make_alert()
    alert["most_recent_instance"] = {}
    assert "no file" in read_code_context(alert)
    build_prompt(alert)   # must not raise


# ── Reproduction convention ──────────────────────────────────────────────────

def test_no_test_means_not_confirmed():
    """An empty repro_test must never be treated as confirmation."""
    from security.triage_agent import reproduce

    result = reproduce(999, make_analysis(repro_test=""))
    assert result.ran is False
    assert result.confirmed is False


def test_system_prompt_states_the_inversion_convention():
    """The convention only works if the model is told about it.

    A test that asserted absence would invert the meaning of every verdict.
    """
    from security.triage_agent import SYSTEM_PROMPT

    assert "presence" in SYSTEM_PROMPT.lower()
    assert "pass" in SYSTEM_PROMPT.lower()
    assert "offline" in SYSTEM_PROMPT.lower()      # generated tests must not call out


def test_system_prompt_tells_the_model_to_check_deployment():
    """Stops it reporting findings in tests/ or the uninstalled OCR extras."""
    from security.triage_agent import SYSTEM_PROMPT

    assert "OCR extra" in SYSTEM_PROMPT
    assert "tests/" in SYSTEM_PROMPT
