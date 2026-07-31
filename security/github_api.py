"""Minimal GitHub REST client over urllib.

Only the handful of calls the triage workflow needs. Deliberately stdlib-only:
this runs with a token that can write issues, so the fewer third-party packages
in that process, the smaller the supply-chain surface.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


class GitHub:
    def __init__(self, token: Optional[str] = None, repo: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        # GITHUB_REPOSITORY is "owner/name" and is always set inside Actions.
        self.repo = repo or os.environ.get("GITHUB_REPOSITORY", "")

        if not self.token:
            raise GitHubError("GITHUB_TOKEN is not set.")
        if "/" not in self.repo:
            raise GitHubError(f"GITHUB_REPOSITORY looks wrong: {self.repo!r}")

    # ── transport ────────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{API_ROOT}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        # Explicit scheme and host check. Every caller passes a literal path, so
        # this should be unreachable — but urlopen honours file:// and custom
        # schemes, and an assertion here is cheaper than trusting that no future
        # caller ever builds a path from alert data.
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "api.github.com":
            raise GitHubError(f"Refusing to request a non-GitHub URL: {url!r}")

        data = json.dumps(body).encode() if body is not None else None

        request = urllib.request.Request(url, data=data, method=method)  # noqa: S310  # nosec B310 - scheme and host asserted above
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        request.add_header("User-Agent", "structrag-triage-agent")
        if data is not None:
            request.add_header("Content-Type", "application/json")

        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310  # nosec B310 - scheme and host asserted above
                    raw = response.read().decode()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:400]

                # 403 with a rate-limit header, or 429, is worth waiting out.
                if exc.code in (403, 429) and attempt < 2:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"GitHub {exc.code}; retrying in {wait}s")
                    time.sleep(wait)
                    last_error = exc
                    continue

                raise GitHubError(f"{method} {path} -> {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise GitHubError(f"{method} {path} failed: {exc}") from exc

        raise GitHubError(f"{method} {path} failed after retries: {last_error}")

    # ── code scanning ────────────────────────────────────────────────────────

    def open_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Open code scanning alerts, newest first.

        Every scanner in this repo uploads SARIF, so CodeQL, Trivy, Semgrep,
        Bandit, pip-audit and Gitleaks findings all arrive through this one
        endpoint.
        """
        try:
            alerts = self._request(
                "GET",
                f"/repos/{self.repo}/code-scanning/alerts",
                params={"state": "open", "per_page": min(limit, 100)},
            )
        except GitHubError as exc:
            # 404 here usually means code scanning has never run, which is a
            # normal state on a fresh repo rather than a failure.
            if "404" in str(exc):
                logger.warning("No code scanning alerts endpoint yet — has a scan run?")
                return []
            raise
        return alerts or []

    # ── issues ───────────────────────────────────────────────────────────────

    def search_issues(self, query: str) -> List[Dict[str, Any]]:
        result = self._request(
            "GET",
            "/search/issues",
            params={"q": f"repo:{self.repo} {query}", "per_page": 100},
        )
        return (result or {}).get("items", [])

    def existing_triage_markers(self) -> set[str]:
        """Markers for alerts already triaged.

        Used instead of a committed state file so concurrent or re-run
        workflows cannot race each other or leave the state out of sync with
        reality. The issues *are* the state.
        """
        markers: set[str] = set()
        for issue in self.search_issues("label:security-triage"):
            body = issue.get("body") or ""
            for line in body.splitlines():
                if line.startswith(MARKER_PREFIX):
                    markers.add(line.strip())
        return markers

    def create_issue(
        self,
        title: str,
        body: str,
        labels: List[str],
        assignees: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"title": title, "body": body, "labels": labels}
        if assignees:
            payload["assignees"] = assignees
        return self._request("POST", f"/repos/{self.repo}/issues", body=payload)

    def comment(self, issue_number: int, body: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/repos/{self.repo}/issues/{issue_number}/comments",
            body={"body": body},
        )

    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        return self._request("GET", f"/repos/{self.repo}/issues/{issue_number}")

    def add_labels(self, issue_number: int, labels: List[str]) -> Any:
        return self._request(
            "POST",
            f"/repos/{self.repo}/issues/{issue_number}/labels",
            body={"labels": labels},
        )

    def remove_label(self, issue_number: int, label: str) -> None:
        try:
            self._request(
                "DELETE",
                f"/repos/{self.repo}/issues/{issue_number}/labels/{urllib.parse.quote(label)}",
            )
        except GitHubError as exc:
            if "404" not in str(exc):
                raise

    def ensure_labels(self, labels: Dict[str, tuple[str, str]]) -> None:
        """Create labels if missing. {name: (colour, description)}."""
        for name, (colour, description) in labels.items():
            try:
                self._request(
                    "POST",
                    f"/repos/{self.repo}/labels",
                    body={"name": name, "color": colour, "description": description},
                )
                logger.info(f"Created label '{name}'")
            except GitHubError as exc:
                # 422 means it already exists, which is the expected case.
                if "422" not in str(exc):
                    logger.warning(f"Could not create label '{name}': {exc}")

    def create_pull_request(
        self, title: str, head: str, base: str, body: str
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/repos/{self.repo}/pulls",
            body={"title": title, "head": head, "base": base, "body": body},
        )


# Machine-readable marker embedded in every issue the agent files, so a rerun
# recognises its own prior work.
MARKER_PREFIX = "<!-- structrag-triage:alert-"


def marker_for(alert_number: int, rule_id: str) -> str:
    return f"{MARKER_PREFIX}{alert_number}:{rule_id} -->"
