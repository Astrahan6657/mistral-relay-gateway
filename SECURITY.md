# Security Policy

## Deployment Hardening

1. **Non-local / Public Deployment**:
   - Always set both `GATEWAY_KEY` (when using server-managed accounts / `UPSTREAM_API_KEY`) and a strong `ADMIN_TOKEN`.
   - Unauthenticated access to server-managed account pools is strictly limited to direct localhost connections (`127.0.0.1` / `::1` with a matching `Host` header and no `X-Forwarded-For`).
   - Always place TLS (Caddy, Nginx, or Cloudflare Tunnel) in front of the gateway when exposing it over a network.
2. **Local Console Protection**:
   - When `ADMIN_TOKEN` is unset, `/api/admin/*` enforces:
     - Direct loopback client IP + `Host` header in `{127.0.0.1, localhost, ::1, [::1]}` (mitigates DNS rebinding)
     - Same-origin check if `Origin` header is sent
     - Custom `X-MRG-Console: 1` header (forces browser CORS preflight)
3. **Secret Hygiene**:
   - `config.json`, `accounts.json`, and `.env` are gitignored and persisted with `0600` permissions.
   - Tokens (`GATEWAY_KEY`, `ADMIN_TOKEN`) are compared using `hmac.compare_digest`.
   - The fallback proxy only forwards an explicit safe header allowlist (never `Cookie` or `X-Admin-Token`).

## Reporting a Vulnerability

Please open a private security advisory on GitHub or contact the repository maintainer directly rather than posting credentials or exploit details in a public issue.
