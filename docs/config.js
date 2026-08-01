// Frontend configuration.
//
// Everything here is PUBLIC — this file is served to every visitor, so it holds
// no API key, no invite code, and no other secret. Those live as environment
// secrets on the backend.

window.STRUCTRAG_CONFIG = {
  // Must match the connect-src entry in index.html — app.js checks this at
  // startup and says so loudly if the two drift. See DEPLOY.md.
  //
  // Hardcoded because it has to be: GitHub Pages serves docs/ straight from the
  // branch with no build step, so there is nothing to substitute a variable at
  // deploy time. It is a public URL, not a secret.
  API_BASE: "https://structrag.onrender.com",

  // How often to poll an upload's progress, in milliseconds.
  POLL_INTERVAL_MS: 1500,

  // Give up on an upload after this long. Slightly above the backend's own
  // PARSE_TIMEOUT_SECONDS so the server's clearer error message usually wins.
  UPLOAD_TIMEOUT_MS: 240000,
};
