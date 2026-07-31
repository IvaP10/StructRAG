"""Apply an approved fix and open a pull request.

Runs only when a human adds the `approved-fix` label to a triage issue. That
label is the entire approval gate: until it is present, nothing here executes.

What it does:
  1. reads the patch out of the issue body,
  2. applies it on a fresh branch,
  3. inverts the reproduction test so it now asserts the flaw is GONE,
  4. runs the full test suite,
  5. opens a pull request linked back to the issue.

It never pushes to main, and the pull request still needs review.

Step 3 turns the reproduction into a regression guard: without it, the fix has
no test that would catch a revert.

    python -m security.apply_fix --issue 42
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess  # noqa: S404  # nosec B404 - drives git and pytest with fixed argv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from security.github_api import GitHub, GitHubError

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s", stream=sys.stdout)
logger = logging.getLogger("apply-fix")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_TIMEOUT_SECONDS = 600


def run(argv: List[str], check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    logger.info("$ " + " ".join(argv))
    completed = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, shell=False
        argv, capture_output=True, text=True, cwd=REPO_ROOT,
        timeout=timeout, shell=False,
    )
    if completed.stdout.strip():
        logger.info(completed.stdout.strip()[-2000:])
    if completed.stderr.strip():
        logger.info(completed.stderr.strip()[-2000:])
    if check and completed.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed with exit {completed.returncode}")
    return completed


# ─────────────────────────────────────────────────────────────────────────────
# Parsing the issue
# ─────────────────────────────────────────────────────────────────────────────

def extract_patch(body: str) -> Optional[str]:
    """Pull the ```diff block out of the issue body."""
    match = re.search(r"```diff\s*\n(.*?)\n```", body, re.DOTALL)
    if not match:
        return None
    patch = match.group(1).strip()
    return patch + "\n" if patch else None


def extract_alert_number(body: str) -> Optional[int]:
    match = re.search(r"<!--\s*structrag-triage:alert-(\d+):", body)
    return int(match.group(1)) if match else None


def extract_repro_path(body: str) -> Optional[str]:
    """Find the reproduction test the triage run committed, if any."""
    match = re.search(r"`(tests/security/test_alert_\d+\.py)`", body)
    return match.group(1) if match else None


# ─────────────────────────────────────────────────────────────────────────────
# Applying
# ─────────────────────────────────────────────────────────────────────────────

def apply_patch(patch: str) -> Tuple[bool, str]:
    """Apply a unified diff, trying progressively looser strategies.

    The diff was written against a snapshot, so exact context can drift while
    the intent stays clear. Hence the three-way and fuzzy fallbacks.
    """
    patch_file = REPO_ROOT / ".triage-patch.diff"
    patch_file.write_text(patch, encoding="utf-8")

    strategies = [
        (["git", "apply", "--verbose", str(patch_file)], "exact"),
        (["git", "apply", "--3way", "--verbose", str(patch_file)], "three-way merge"),
        (["git", "apply", "--reject", "--whitespace=fix", str(patch_file)], "fuzzy"),
    ]

    try:
        for argv, label in strategies:
            result = run(argv, check=False)
            if result.returncode == 0:
                logger.info(f"Patch applied ({label}).")
                return True, label
            logger.warning(f"{label} strategy failed.")
        return False, "every strategy failed"
    finally:
        patch_file.unlink(missing_ok=True)
        # --reject leaves .rej files behind; they must not reach the PR.
        for reject in REPO_ROOT.rglob("*.rej"):
            reject.unlink(missing_ok=True)


def invert_repro_test(path: str) -> bool:
    """Flip the reproduction test into a regression test.

    The test asserts the flaw is present, so a correct fix makes it fail. Marking
    it xfail(strict=True) keeps it: the suite goes red if it ever passes again.
    """
    test_path = REPO_ROOT / path
    if not test_path.is_file():
        logger.warning(f"No reproduction test at {path}; nothing to invert.")
        return False

    source = test_path.read_text(encoding="utf-8")

    if "xfail" in source:
        logger.info("Reproduction test already inverted.")
        return True

    banner = (
        "\n# ── Inverted by security/apply_fix.py ──────────────────────────────\n"
        "# This test asserted the vulnerability was PRESENT. With the fix applied it\n"
        "# must now fail. strict=True means the suite goes red if it ever passes\n"
        "# again, which would mean the fix has been reverted.\n"
        "import pytest as _pytest\n\n"
    )

    def add_marker(match: re.Match) -> str:
        return (
            '@_pytest.mark.xfail(strict=True, reason="fixed; passing again means regressed")\n'
            f"{match.group(0)}"
        )

    modified = re.sub(r"^def (test_\w+)", add_marker, source, flags=re.MULTILINE)

    if modified == source:
        logger.warning("Found no test function to invert.")
        return False

    # Insert the import after the module docstring so it stays valid.
    docstring = re.match(r'^\s*(?:"""|\'\'\')(?:.|\n)*?(?:"""|\'\'\')\s*\n', modified)
    if docstring:
        cut = docstring.end()
        modified = modified[:cut] + banner + modified[cut:]
    else:
        modified = banner + modified

    test_path.write_text(modified, encoding="utf-8")
    logger.info(f"Inverted {path}.")
    return True


def run_test_suite() -> Tuple[bool, str]:
    environment = dict(os.environ)
    environment.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
    environment.setdefault("INVITE_CODE", "test-invite-code-1234")
    environment.setdefault("SESSION_SECRET", "test-session-secret-value-32-chars")
    environment["LOG_FILE"] = ""
    environment["PYTHONPATH"] = str(REPO_ROOT)

    try:
        completed = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, shell=False
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
            capture_output=True, text=True, cwd=REPO_ROOT,
            timeout=TEST_TIMEOUT_SECONDS, env=environment, shell=False,
        )
        return completed.returncode == 0, (completed.stdout + completed.stderr)[-6000:]
    except subprocess.TimeoutExpired:
        return False, f"The test suite exceeded {TEST_TIMEOUT_SECONDS}s."


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Apply an approved security fix.")
    parser.add_argument("--issue", type=int, required=True, help="Triage issue number.")
    args = parser.parse_args()

    try:
        github = GitHub()
    except GitHubError as exc:
        logger.error(str(exc))
        return 1

    issue = github.get_issue(args.issue)
    body = issue.get("body") or ""
    labels = {label["name"] for label in issue.get("labels", [])}

    # Checked here as well as in the workflow trigger, so a manual run cannot
    # skip the approval gate.
    if "approved-fix" not in labels:
        logger.error(f"Issue #{args.issue} is not labelled approved-fix. Refusing.")
        return 1

    patch = extract_patch(body)
    if not patch:
        github.comment(
            args.issue,
            "I could not find a `diff` block in this issue, so there is no patch to apply. "
            "This one needs fixing by hand.",
        )
        logger.error("No patch in the issue body.")
        return 1

    alert_number = extract_alert_number(body) or args.issue
    branch = f"security/fix-alert-{alert_number}-issue-{args.issue}"

    run(["git", "config", "user.name", "structrag-security-agent"])
    run(["git", "config", "user.email", "noreply@github.com"])
    run(["git", "checkout", "-b", branch])

    applied, strategy = apply_patch(patch)
    if not applied:
        github.comment(
            args.issue,
            "**The patch would not apply.**\n\n"
            "The code has probably changed since this issue was filed. The description and "
            "reproduction above are still valid — the diff just needs rebasing by hand.\n\n"
            "I have left `main` untouched.",
        )
        github.remove_label(args.issue, "approved-fix")
        logger.error("Patch did not apply.")
        return 1

    repro_path = extract_repro_path(body)
    inverted = invert_repro_test(repro_path) if repro_path else False

    passed, output = run_test_suite()

    if not passed:
        github.comment(
            args.issue,
            "**The fix applied, but the test suite fails.**\n\n"
            "Not opening a pull request — a fix that breaks something else is not a fix.\n\n"
            f"<details><summary>Test output</summary>\n\n```\n{output}\n```\n\n</details>",
        )
        github.remove_label(args.issue, "approved-fix")
        logger.error("Test suite failed after the patch.")
        return 1

    run(["git", "add", "-A"])
    run([
        "git", "commit", "-m",
        f"security: fix code scanning alert #{alert_number}\n\n"
        f"Applies the reviewed patch from issue #{args.issue}.\n"
        f"{'Reproduction test inverted to guard against regression.' if inverted else ''}\n\n"
        f"Closes #{args.issue}",
    ])
    run(["git", "push", "origin", branch])

    pull_request = github.create_pull_request(
        title=f"security: fix alert #{alert_number} ({issue.get('title', '')[:70]})",
        head=branch,
        base=os.environ.get("GITHUB_BASE_BRANCH", "main"),
        body=f"""Closes #{args.issue}

Applies the patch you approved on that issue.

| | |
|---|---|
| Patch applied via | {strategy} |
| Reproduction test | {"inverted to guard against regression" if inverted else "none was available"} |
| Test suite | passing |

<details><summary>Test output</summary>

```
{output}
```

</details>

---

This branch was produced by `security/apply_fix.py` after you added the
`approved-fix` label. It still needs your review before merging — read the diff.
""",
    )

    github.comment(
        args.issue,
        f"Opened {pull_request.get('html_url')} with the fix. The suite passes. "
        f"{'The reproduction test now guards against regression. ' if inverted else ''}"
        "Please review the diff before merging.",
    )

    logger.info(f"Opened PR: {pull_request.get('html_url')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
