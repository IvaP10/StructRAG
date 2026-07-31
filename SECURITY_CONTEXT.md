# Threat model

This file is read by the automated triage agent (`security/triage_agent.py`) and
included in the prompt it sends for every scanner alert.

Its purpose is to let the agent judge whether a finding is *actually
exploitable here* rather than restating what the scanner said. A scanner knows
that `subprocess.run` was called; it does not know whether the input is a
hard-coded constant or a filename from an anonymous upload. That difference is
what this document supplies.

Keep it current. A stale threat model produces confident, wrong triage.

---

## What this application is

A hybrid RAG service over user-supplied PDFs, in two deployed parts:

| Part | Where | Holds secrets? | Reachable by |
|---|---|---|---|
| Static frontend (`docs/`) | GitHub Pages | **No** | Anyone on the internet |
| API (`server/`, `core/`, root modules) | Render (Docker) | **Yes** | Anyone with the invite code |
| CLI (`main.py`) | A developer's own machine | Local `key.env` | Only that developer |

## What is worth protecting, in order

1. **The OpenAI API key.** It is on a personal account with real money behind
   it. Theft, or abuse that runs up spend, is the worst realistic outcome.
2. **The service being used for something other than document Q&A.** Turning
   the demo into a free general-purpose LLM is the abuse this design most
   expects. Not catastrophic, but it drains the budget and it is what the
   layered guards exist to stop.
3. **One visitor reading another visitor's documents.** Uploads are private to
   a session. Cross-session leakage is a genuine confidentiality breach.
4. **Integrity of the container.** Remote code execution in the deployed image
   would expose the key, so it collapses into (1).

Explicitly *not* protected: availability. This is a personal demo. It is
allowed to be slow, asleep, rate-limited, or budget-exhausted, and a denial-of-
service finding is low severity here.

## Trust boundaries

Untrusted, attacker-controlled input:

- **Uploaded PDF bytes.** The sharpest edge in the system. Parsed by PyMuPDF and
  pdfplumber — large native parsers with a real CVE history. Validated by
  `server/guards.validate_pdf_bytes` before reaching them: magic bytes, size,
  page count, encryption, and rejection of active content (`/JavaScript`,
  `/Launch`, `/OpenAction`, `/EmbeddedFile`, …).
- **PDF *text* content.** Extracted text is fed to an LLM as context, so a
  document can attempt prompt injection. Treated as data: delimited and
  neutralised in `generator._neutralize_delimiters`, with the system prompt
  instructing that context is never instructions.
- **Query strings.** Length-capped, control characters stripped, screened by
  regex, then by a classifier.
- **Filenames.** Echoed back into the page, so sanitised by
  `guards.safe_filename` against both path traversal and HTML injection.
- **`X-Forwarded-For`.** Spoofable by design. Used only for cost-shaping rate
  limits, never as identity. Findings that say "IP can be spoofed" are correct
  but already accounted for.
- **The session token.** Signed with HMAC-SHA256 and compared using
  `hmac.compare_digest`. Forgery requires `SESSION_SECRET`.

Trusted:

- Environment variables and host-managed secrets.
- The repository's own source.

## Defences already in place

Attacks are meant to die at the cheapest layer that can stop them:

1. `DISABLED` kill switch — every endpoint 503s.
2. CORS allowlist — explicit origins, never `*`.
3. Daily spend ledger — hard 503 at `DAILY_USD_CAP`. **The limit that actually
   bounds the loss.**
4. Invite code, constant-time compared, exchanged for an expiring signed token.
5. Sliding-window rate limits per session and per IP.
6. Input validation — query length, PDF structure.
7. **Relevance gate** (`retriever._absolute_relevance_gate`) — if no chunk
   clears an absolute dense-cosine floor, the request is refused with **no LLM
   call at all**. The primary and cheapest anti-abuse measure.
8. Intent classifier — runs only after retrieval succeeds.
9. Hardened system prompt with delimited, neutralised context.
10. Output token cap.
11. Session-scoped retrieval — every Qdrant query filters on `session_id`.
12. Container runs as uid 1000 with the application directory read-only;
    only `/data` is writable.

## Judging severity here

Treat as **high** — reaches the key, the budget, or another user's data:

- Any path from a request to arbitrary code execution or file read in the container.
- Anything that leaks `OPENAI_API_KEY`, `SESSION_SECRET`, or `INVITE_CODE`,
  including into logs or error responses.
- Session token forgery, or session id prediction.
- A route that reaches an OpenAI call while bypassing the spend ledger, the rate
  limiter, or the relevance gate.
- Retrieval that returns chunks from another `session_id`.
- Anything that lets a request bypass `validate_pdf_bytes`.
- A secret committed to git history.

Treat as **low, or not a finding**:

- Denial of service, resource exhaustion, slow endpoints. Availability is not a goal.
- `X-Forwarded-For` spoofing. Known and accepted; it gates cost, not identity.
- Missing security headers on JSON API responses, where a browser is not the client.
- Findings in `tests/`. Test code is not deployed.
- Findings in `evaluate.py`, `docs/`, or the OCR/eval extras. Not part of the
  serving path — `evaluate.py` is developer-run, and `docs/` holds no secrets.
- Hard-coded strings in tests that look like credentials but are obvious
  placeholders (`sk-test-not-a-real-key`).
- CVEs in the optional OCR (`docling`) or evaluation (`ragas`, `datasets`)
  packages. Neither is installed in the image; check `requirements.txt` and the
  Dockerfile before concluding a package actually ships.
- "Unvalidated input" where the value is a module-level constant.

## Known and accepted

- Rate-limit and session state are in-process, so a restart clears them.
  Accepted: the spend ledger is the real bound, and it also resets — worst case
  is one extra day's cap.
- Single uvicorn worker. Required, since session state is in memory. A second
  worker would answer queries without seeing a visitor's uploads.
- The container filesystem is ephemeral. Intentional: uploads should not
  outlive the session.
- No OCR in the image, so scanned PDFs yield no text. A functional limitation,
  surfaced to the user, not a security issue.
- Anyone holding the invite code can spend budget up to the daily cap. That is
  the accepted cost of a shareable demo.
