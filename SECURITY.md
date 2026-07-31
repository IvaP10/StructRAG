# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's private reporting instead: go to the
[Security tab](https://github.com/IvaP10/StructRAG/security/advisories/new) and
open a draft advisory. That keeps the details between us until there is a fix.

Useful to include, if you have it:

- what you did and what happened
- which part is affected — the hosted API, the static frontend, or the CLI
- what an attacker gets out of it
- anything that reproduces it

I maintain this in my spare time, so expect a first reply within about a week.

## Scope

| Part | In scope? |
|---|---|
| The hosted API (`server/`, `core/`, root modules) | **Yes** |
| The static frontend (`docs/`) | **Yes** |
| The CLI (`main.py`) | Yes, though it only ever runs on a developer's own machine |
| `evaluate.py`, `tests/` | No — neither is deployed |
| Render or GitHub Pages themselves | No — report those to the platform |

Availability is explicitly **not** in scope. This is a personal demo on a small
budget; it is *designed* to be rate-limited, budget-capped, and asleep when idle.
Being able to make it slow or unavailable is expected behaviour, not a finding.

What I care about most, in order:

1. Anything that exposes or lets someone spend the OpenAI API key.
2. Anything that reads another visitor's uploaded documents.
3. Code execution or arbitrary file access in the container.
4. Session token forgery or prediction.
5. Reaching an OpenAI call while bypassing the spend cap, rate limits, or the
   relevance gate.

[SECURITY_CONTEXT.md](SECURITY_CONTEXT.md) has the full threat model, including
what is already known and accepted.

## How this project is monitored

Most scanners report into the Security tab as code scanning alerts. Trivy and
pip-audit report into their workflow run summary instead — pip-audit because it
has no SARIF output format, Trivy because a Debian-based Python image carries a
couple of hundred base-layer CVEs that no change to this repo can fix, and as
alerts they bury the findings that are actually actionable. Both are still
scanned on every run and both still gate: Trivy fails the build outright if a
secret is baked into an image layer. No scanner opens pull requests.

| Workflow | Tool | Looks for | Reports to |
|---|---|---|---|
| [codeql.yml](.github/workflows/codeql.yml) | CodeQL | Dataflow bugs — injection, traversal, SSRF | Security tab |
| [security-scan.yml](.github/workflows/security-scan.yml) | Gitleaks | Secrets, across the whole git history | Security tab |
| [security-scan.yml](.github/workflows/security-scan.yml) | Semgrep, Bandit | OWASP Top 10, Python security lint | Security tab |
| [security-scan.yml](.github/workflows/security-scan.yml) | Trivy | Dependency CVEs, image CVEs, Dockerfile misconfig | Run summary + artifact |
| [security-scan.yml](.github/workflows/security-scan.yml) | pip-audit | Python dependency CVEs, second opinion to Trivy | Run summary |
| [zap.yml](.github/workflows/zap.yml) | OWASP ZAP | Runtime issues against the live API | Security tab |

Every two days, [vuln-triage.yml](.github/workflows/vuln-triage.yml) reads the
open alerts, works out which are genuinely exploitable given the threat model,
**writes and runs a test to prove it**, and files an issue for each confirmed
finding. Findings it cannot demonstrate are not reported — that filter is the
point.

Fixes are applied only after a human adds the `approved-fix` label, which
triggers [vuln-fix-apply.yml](.github/workflows/vuln-fix-apply.yml). Nothing
automated ever pushes to `main`.

## If a key is ever committed

Rotate first, then clean history. In that order — the moment a secret is pushed
it should be considered public, because it is trivially harvested and anyone who
already cloned the repo has it.

1. Revoke the key at the provider and issue a new one.
2. Update the Render environment variable and your local `key.env`.
3. Only then rewrite history (`git filter-repo`) if you want it gone.

Step 3 without step 1 accomplishes nothing.

`.gitignore` covers `key.env`, and [.pre-commit-config.yaml](.pre-commit-config.yaml)
runs Gitleaks before every commit, so this should not come up. Install the hooks:

```bash
pip install pre-commit && pre-commit install
```
