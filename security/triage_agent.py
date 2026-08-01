"""Automated triage of code scanning alerts.

Runs on a schedule. For each new open alert it:

  1. reads the flagged code plus surrounding context,
  2. asks a model whether the finding is genuinely exploitable *in this
     application*, given SECURITY_CONTEXT.md,
  3. if the model says yes and can express it as a test, writes that test and
     **runs it** — a finding that cannot be demonstrated is not reported,
  4. opens one GitHub issue per confirmed finding: what it is, why it is there,
     the reproduction output, and a proposed patch.

It never modifies main and never opens a pull request. Fixing is a separate,
human-approved step (see security/apply_fix.py).

Step 3 is the filter that matters: scanners report patterns, not exploits.
Requiring a passing reproduction discards findings that are true in general but
not true here.

    python -m security.triage_agent            # normal run
    python -m security.triage_agent --dry-run  # analyse and print, file nothing

The reproduction test is model-written code run in the CI runner. The workflow
grants this job only security-events:read, contents:read and issues:write, and
the run has a hard timeout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess  # noqa: S404  # nosec B404 - runs pytest on a generated test, by design
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from security.github_api import GitHub, GitHubError, marker_for

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("triage")

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Model access
# ─────────────────────────────────────────────────────────────────────────────
# gpt-4o rather than the mini model: a wrong "not exploitable" leaves a real hole
# open. Override with TRIAGE_MODEL.

DEFAULT_MODEL = os.environ.get("TRIAGE_MODEL", "gpt-4o")


class LLMError(RuntimeError):
    pass


def complete_json(
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 3000,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Send a prompt and parse a JSON object out of the reply.

    Temperature 0 so the same alert on unchanged code reaches the same verdict.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMError("The openai package is not installed.") from exc

    try:
        response = OpenAI(api_key=api_key).chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:
        raise LLMError(f"Model call failed: {exc}") from exc

    if response.usage:
        logger.info("Model %s: %d in / %d out tokens", model,
                    response.usage.prompt_tokens, response.usage.completion_tokens)

    return _parse_json((response.choices[0].message.content or "").strip())


def _parse_json(content: str) -> Dict[str, Any]:
    """Parse the reply, tolerating a fenced code block or surrounding prose."""
    candidates = [content]

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))

    braced = re.search(r"\{.*\}", content, re.DOTALL)
    if braced:
        candidates.append(braced.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise LLMError(f"Reply was not JSON: {content[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# Triage
# ─────────────────────────────────────────────────────────────────────────────

# Lines of code either side of the flagged line to include in the prompt. Enough
# for the model to see the enclosing function and its callers' expectations.
CONTEXT_LINES = 45

# Ceiling on alerts analysed per run, so a misconfigured scanner producing 400
# findings cannot turn into 400 model calls and 400 issues.
MAX_ALERTS_PER_RUN = int(os.environ.get("MAX_ALERTS_PER_RUN", "8"))

# A reproduction test gets this long before being treated as inconclusive.
REPRO_TIMEOUT_SECONDS = 120

REPRO_DIR = REPO_ROOT / "tests" / "security"

LABELS = {
    "security-triage": ("d93f0b", "Filed by the automated security triage agent"),
    "awaiting-approval": ("fbca04", "Add 'approved-fix' to let the agent open a fix PR"),
    "approved-fix": ("0e8a16", "Approved — the agent may open a pull request with the fix"),
}


SYSTEM_PROMPT = """You are a security engineer triaging static-analysis findings for a specific \
application. You have the threat model and the actual source code.

Your job is to decide whether a finding is genuinely exploitable IN THIS APPLICATION, not whether \
the flagged pattern is dangerous in general. Scanners are pattern matchers with no knowledge of \
context; you have context, so use it.

Be rigorous in both directions. A false "exploitable" wastes the maintainer's time and erodes \
trust in this whole process. A false "not_exploitable" leaves a real hole open. When the evidence \
genuinely does not settle it, say needs_human.

Before concluding, check specifically:
- Does untrusted input actually reach the flagged code? Trace it. If the value is a module-level \
constant, a hard-coded string, or set only by the operator via an environment variable, it is not \
attacker-controlled.
- Is there already a guard on the path? The threat model lists the existing defences. A finding \
that a guard already handles is not exploitable.
- Is the code even deployed? Files under tests/ and docs/, and evaluate.py, do not run on the \
server. Only packages in requirements.txt are installed in the image — the optional OCR extra \
(docling) and evaluation extra (ragas, datasets) are not, so a CVE in torch is not part of the \
deployed attack surface.
- Does the threat model already list this as known and accepted?

Reply with a single JSON object:

{
  "verdict": "exploitable" | "not_exploitable" | "needs_human",
  "confidence": 0.0-1.0,
  "severity": "critical" | "high" | "medium" | "low",
  "what": "Plain-language description of the flaw. No jargon. 1-3 sentences.",
  "why": "Why this code ended up this way, and the precise mechanism by which it fails.",
  "impact": "What an attacker actually achieves, referencing what this application protects.",
  "attack_steps": ["Concrete ordered steps an attacker takes"],
  "repro_test": "A complete pytest file, or empty string if it cannot be tested",
  "repro_explanation": "What the test demonstrates and why passing means vulnerable",
  "fix_explanation": "What to change and why that closes it",
  "patch": "A unified diff against the repository root, or empty string",
  "false_positive_reason": "Only when verdict is not_exploitable: why the scanner was wrong"
}

Rules for "repro_test":
- A complete, runnable pytest file including imports.
- It must FAIL if the vulnerability is absent and PASS if it is present. Assert the *presence* of \
the flaw. This inversion is deliberate: the harness treats a pass as confirmation.
- Name the test function so it reads as an assertion about the flaw, e.g. \
test_session_token_accepts_forged_signature.
- It must be self-contained and offline. No network, no real API keys, no live services. Stub what \
you need. The runner has the repository root importable and pytest available.
- Add a docstring stating what a pass proves.
- If the flaw cannot be honestly demonstrated in a unit test (a container hardening issue, a \
dependency CVE with no reachable call path), return an empty string rather than inventing a test \
that merely asserts the code looks a certain way. An empty string is a valid, useful answer.

Rules for "patch":
- Unified diff, paths relative to the repository root, with enough context to apply.
- Minimal. Fix the flaw; do not reformat or refactor around it.
- Never weaken an existing guard to make a test pass.
"""


@dataclass
class Analysis:
    verdict: str
    confidence: float
    severity: str
    what: str
    why: str
    impact: str
    attack_steps: List[str]
    repro_test: str
    repro_explanation: str
    fix_explanation: str
    patch: str
    false_positive_reason: str

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "Analysis":
        return cls(
            verdict=str(raw.get("verdict", "needs_human")).strip().lower(),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            severity=str(raw.get("severity", "medium")).strip().lower(),
            what=str(raw.get("what", "")).strip(),
            why=str(raw.get("why", "")).strip(),
            impact=str(raw.get("impact", "")).strip(),
            attack_steps=[str(s) for s in (raw.get("attack_steps") or [])],
            repro_test=str(raw.get("repro_test", "") or ""),
            repro_explanation=str(raw.get("repro_explanation", "")).strip(),
            fix_explanation=str(raw.get("fix_explanation", "")).strip(),
            patch=str(raw.get("patch", "") or ""),
            false_positive_reason=str(raw.get("false_positive_reason", "")).strip(),
        )


@dataclass
class ReproResult:
    ran: bool
    confirmed: bool
    output: str
    test_path: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Gathering context
# ─────────────────────────────────────────────────────────────────────────────

def alert_location(alert: Dict[str, Any]) -> Dict[str, Any]:
    return (alert.get("most_recent_instance") or {}).get("location") or {}


def read_code_context(alert: Dict[str, Any]) -> str:
    """The flagged file around the flagged line, with line numbers."""
    location = alert_location(alert)
    path = location.get("path")
    if not path:
        return "(the alert names no file)"

    target = REPO_ROOT / path
    if not target.is_file():
        return f"(file not found in the checkout: {path})"

    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read {path}: {exc})"

    start_line = int(location.get("start_line") or 1)
    lo = max(0, start_line - 1 - CONTEXT_LINES)
    hi = min(len(lines), start_line - 1 + CONTEXT_LINES)

    out = [f"File: {path}  (flagged at line {start_line})", ""]
    for index in range(lo, hi):
        marker = ">>" if index == start_line - 1 else "  "
        out.append(f"{marker} {index + 1:5d}  {lines[index]}")
    return "\n".join(out)


def read_threat_model() -> str:
    path = REPO_ROOT / "SECURITY_CONTEXT.md"
    if not path.is_file():
        return "(SECURITY_CONTEXT.md is missing — judge conservatively.)"
    return path.read_text(encoding="utf-8", errors="replace")


def build_prompt(alert: Dict[str, Any]) -> str:
    rule = alert.get("rule") or {}
    tool = (alert.get("tool") or {}).get("name", "unknown")
    location = alert_location(alert)

    return f"""# Threat model

{read_threat_model()}

---

# Scanner finding

Tool:          {tool}
Rule:          {rule.get('id', '?')}
Name:          {rule.get('name', '?')}
Severity:      {rule.get('security_severity_level') or rule.get('severity') or '?'}
Alert number:  {alert.get('number')}
File:          {location.get('path', '?')}
Lines:         {location.get('start_line', '?')}-{location.get('end_line', '?')}

Scanner description:
{rule.get('full_description') or rule.get('description') or '(none)'}

Message on this instance:
{((alert.get('most_recent_instance') or {}).get('message') or {}).get('text', '(none)')}

---

# Code

```
{read_code_context(alert)}
```

---

Decide whether this is genuinely exploitable in this application. Reply with the JSON object \
described in your instructions."""


# ─────────────────────────────────────────────────────────────────────────────
# Reproduction
# ─────────────────────────────────────────────────────────────────────────────

def reproduce(alert_number: int, analysis: Analysis) -> ReproResult:
    """Write the model's test and run it.

    The test asserts the vulnerability is *present*, so a pass confirms the flaw
    and a failure means the scanner or the model was wrong.
    """
    if not analysis.repro_test.strip():
        return ReproResult(
            ran=False,
            confirmed=False,
            output="No reproduction test was produced — the finding could not be demonstrated "
                   "as a unit test.",
        )

    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    init_file = REPRO_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    test_path = REPRO_DIR / f"test_alert_{alert_number}.py"
    header = (
        f'"""Reproduction for code scanning alert #{alert_number}.\n\n'
        f"Written by security/triage_agent.py. A PASS means the vulnerability is\n"
        f"present. Once the fix lands, this assertion gets inverted so the test\n"
        f"guards against regression.\n\n"
        f"{analysis.repro_explanation}\n"
        f'"""\n\n'
    )
    test_path.write_text(header + _strip_fences(analysis.repro_test), encoding="utf-8")

    logger.info(f"Running reproduction: {test_path.relative_to(REPO_ROOT)}")

    environment = dict(os.environ)
    # Obvious placeholders. The generated test must not reach a real service, and
    # if it tries, it should fail on credentials rather than spend anything.
    environment.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
    environment.setdefault("INVITE_CODE", "test-invite-code-1234")
    environment.setdefault("SESSION_SECRET", "test-session-secret-value-32-chars")
    environment["LOG_FILE"] = ""
    environment["PYTHONPATH"] = str(REPO_ROOT)

    try:
        completed = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, shell=False
            [sys.executable, "-m", "pytest", str(test_path), "-v", "--no-header",
             "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
            timeout=REPRO_TIMEOUT_SECONDS,
            cwd=REPO_ROOT,
            env=environment,
            shell=False,
        )
        output = (completed.stdout + completed.stderr)[-6000:]
        # Exit code 0 means every assertion held, i.e. the flaw is present.
        confirmed = completed.returncode == 0
        logger.info(f"Reproduction {'CONFIRMED' if confirmed else 'did not confirm'} "
                    f"(pytest exit {completed.returncode})")
        return ReproResult(True, confirmed, output, str(test_path.relative_to(REPO_ROOT)))

    except subprocess.TimeoutExpired:
        return ReproResult(
            ran=True,
            confirmed=False,
            output=f"The reproduction test exceeded {REPRO_TIMEOUT_SECONDS}s and was stopped.",
            test_path=str(test_path.relative_to(REPO_ROOT)),
        )
    except Exception as exc:
        return ReproResult(True, False, f"Could not run the reproduction: {exc}")


def _strip_fences(text: str) -> str:
    """Remove a ```python fence if the model wrapped its file in one."""
    match = re.match(r"^\s*```(?:python)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    return match.group(1) if match else text


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def issue_body(alert: Dict[str, Any], analysis: Analysis, repro: ReproResult) -> str:
    rule = alert.get("rule") or {}
    location = alert_location(alert)
    tool = (alert.get("tool") or {}).get("name", "unknown")
    path = location.get("path", "?")
    line = location.get("start_line", "?")

    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(analysis.attack_steps, 1)) or "_Not given._"

    if repro.confirmed:
        verification = (
            f"**Reproduced.** `{repro.test_path}` was written and run, and it passes — "
            "which under this harness's convention means the flaw is present.\n\n"
            f"<details><summary>Test output</summary>\n\n```\n{repro.output}\n```\n\n</details>"
        )
    elif repro.ran:
        verification = (
            "**Not reproduced.** A test was generated but did not confirm the flaw, so treat "
            f"this as unverified.\n\n<details><summary>Test output</summary>\n\n"
            f"```\n{repro.output}\n```\n\n</details>"
        )
    else:
        verification = f"**Not tested.** {repro.output}"

    patch_section = (
        f"```diff\n{_strip_fences(analysis.patch)}\n```"
        if analysis.patch.strip()
        else "_The agent did not propose a patch. This one needs fixing by hand._"
    )

    return f"""{marker_for(int(alert.get('number', 0)), str(rule.get('id', '?')))}
## What is wrong

{analysis.what}

## Why it is there

{analysis.why}

## What it lets someone do

{analysis.impact}

**How an attacker gets there:**

{steps}

## Verification

{verification}

## How to fix it

{analysis.fix_explanation}

{patch_section}

---

### Approving this fix

Add the **`approved-fix`** label and the agent will open a pull request on a new
branch: it applies the patch above, inverts the reproduction test so it asserts
the flaw is *gone*, and runs the full suite. Nothing is pushed to `main`, and the
pull request still needs your review.

If this is wrong, close the issue — the agent will not raise it again.

<sub>Alert #{alert.get('number')} · {tool} · `{rule.get('id', '?')}` · `{path}:{line}` ·
verdict `{analysis.verdict}` · confidence {analysis.confidence:.0%} ·
severity {analysis.severity}</sub>
"""


def issue_title(alert: Dict[str, Any], analysis: Analysis) -> str:
    location = alert_location(alert)
    filename = Path(location.get("path", "?")).name
    summary = analysis.what.split(".")[0].strip()[:90] or "Security finding"
    return f"[security] {summary} ({filename})"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def process(alert: Dict[str, Any], github: GitHub, dry_run: bool) -> str:
    number = int(alert.get("number", 0))
    rule_id = str((alert.get("rule") or {}).get("id", "?"))
    logger.info(f"── Alert #{number}: {rule_id}")

    try:
        raw = complete_json(SYSTEM_PROMPT, build_prompt(alert))
    except LLMError as exc:
        logger.error(f"Analysis failed for #{number}: {exc}")
        return "error"

    analysis = Analysis.from_json(raw)
    logger.info(f"   verdict={analysis.verdict} confidence={analysis.confidence:.0%} "
                f"severity={analysis.severity}")

    if analysis.verdict == "not_exploitable":
        logger.info(f"   dismissed: {analysis.false_positive_reason[:140]}")
        return "not_exploitable"

    repro = reproduce(number, analysis)

    # Report only what was demonstrated, or what a human needs to judge.
    should_report = repro.confirmed or analysis.verdict == "needs_human" or (
        not repro.ran and analysis.confidence >= 0.75
        and analysis.severity in ("critical", "high")
    )

    if not should_report:
        logger.info("   not reported: could not be demonstrated and not high-confidence severe")
        return "unconfirmed"

    if dry_run:
        print("\n" + "=" * 72)
        print(issue_title(alert, analysis))
        print("=" * 72)
        print(issue_body(alert, analysis, repro))
        return "dry-run"

    labels = ["security-triage", "awaiting-approval"]
    if analysis.severity in ("critical", "high"):
        labels.append("priority")

    assignees = [a for a in [os.environ.get("TRIAGE_ASSIGNEE", "").strip()] if a]

    issue = github.create_issue(
        title=issue_title(alert, analysis),
        body=issue_body(alert, analysis, repro),
        labels=labels,
        assignees=assignees or None,
    )
    logger.info(f"   filed issue #{issue.get('number')}")
    return "reported"


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage open code scanning alerts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse and print, but file nothing.")
    parser.add_argument("--limit", type=int, default=MAX_ALERTS_PER_RUN,
                        help="Maximum alerts to analyse this run.")
    args = parser.parse_args()

    try:
        github = GitHub()
    except GitHubError as exc:
        logger.error(str(exc))
        return 1

    if not args.dry_run:
        github.ensure_labels(LABELS)

    try:
        alerts = github.open_alerts()
    except GitHubError as exc:
        logger.error(f"Could not fetch alerts: {exc}")
        return 1

    if not alerts:
        logger.info("No open code scanning alerts. Nothing to do.")
        return 0

    logger.info(f"{len(alerts)} open alert(s)")

    # Issues are the state store, so there is no state file to race on.
    try:
        already = github.existing_triage_markers()
    except GitHubError as exc:
        logger.warning(f"Could not list existing issues, may duplicate: {exc}")
        already = set()

    fresh = [
        a for a in alerts
        if marker_for(int(a.get("number", 0)), str((a.get("rule") or {}).get("id", "?")))
        not in already
    ]

    skipped = len(alerts) - len(fresh)
    if skipped:
        logger.info(f"{skipped} already triaged, skipping")

    if not fresh:
        logger.info("Nothing new.")
        return 0

    # Worst first, so a truncated run still covers what matters.
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "warning": 3, "note": 4, "error": 1}
    fresh.sort(key=lambda a: order.get(
        str((a.get("rule") or {}).get("security_severity_level")
            or (a.get("rule") or {}).get("severity") or "medium").lower(), 5
    ))

    batch = fresh[: args.limit]
    if len(fresh) > len(batch):
        logger.info(f"Analysing the {len(batch)} most severe; {len(fresh) - len(batch)} "
                    "will wait for the next run")

    tally: Dict[str, int] = {}
    for alert in batch:
        outcome = process(alert, github, args.dry_run)
        tally[outcome] = tally.get(outcome, 0) + 1

    logger.info("── Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("## Security triage\n\n")
            fh.write(f"- Open alerts: {len(alerts)}\n")
            fh.write(f"- Already triaged: {skipped}\n")
            fh.write(f"- Analysed: {len(batch)}\n")
            for key, value in sorted(tally.items()):
                fh.write(f"- {key}: {value}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
