// Frontend configuration.
//
// Everything here is PUBLIC — this file is served to every visitor, so it holds
// no API key, no invite code, and no other secret. Those live as environment
// secrets on the backend.

// The backend URL is deliberately NOT here. It lives once, in the connect-src
// entry of the Content-Security-Policy in index.html, which is the only place
// it can be a variable-free literal — and app.js reads it back out of there.
// Repeating it in this file would just be a second string to keep in sync.

window.STRUCTRAG_CONFIG = {
  // How often to poll an upload's progress, in milliseconds.
  POLL_INTERVAL_MS: 1500,

  // Give up on an upload after this long. Slightly above the backend's own
  // PARSE_TIMEOUT_SECONDS so the server's clearer error message usually wins.
  UPLOAD_TIMEOUT_MS: 240000,
};
