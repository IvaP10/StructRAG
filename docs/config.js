// Frontend configuration.
//
// Everything here is PUBLIC — this file is served to every visitor, so it holds
// no API key, no invite code, and no other secret. Those live as secrets on the
// backend Space.

window.STRUCTRAG_CONFIG = {
  // Must match the connect-src entry in index.html. See DEPLOY.md.
  API_BASE: "https://ivap10-structrag.hf.space",

  // How often to poll an upload's progress, in milliseconds.
  POLL_INTERVAL_MS: 1500,

  // Give up on an upload after this long. Slightly above the backend's own
  // PARSE_TIMEOUT_SECONDS so the server's clearer error message usually wins.
  UPLOAD_TIMEOUT_MS: 240000,
};
