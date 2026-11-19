from __future__ import annotations

import argparse
import os
import threading
import time
import webbrowser

import uvicorn

from .app import create_app
from .config import GatewayConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mistral-relay-gateway",
        description="Mistral / New API compatibility gateway with built-in Web Console.",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8790")), help="Bind port (default: 8790)")
    parser.add_argument("--config", default=os.environ.get("CONFIG_PATH", ""), help="Optional JSON config file path")
    parser.add_argument("--open", action="store_true", help="Open Web Console in default browser after startup")
    args = parser.parse_args()

    cfg = GatewayConfig.load(config_path=args.config or None)
    app = create_app(cfg, bind_host=args.host)

    if args.open:
        display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        url = f"http://{display_host}:{args.port}/ui"

        def _open_browser() -> None:
            time.sleep(0.8)
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
