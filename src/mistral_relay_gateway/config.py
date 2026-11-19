from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

UpstreamMode = Literal["direct_mistral", "new_api"]
ResponsesMode = Literal["auto", "native", "convert"]
MistralChatEndpoint = Literal["native", "openai_compat"]


def normalize_upstream_base(raw: str) -> tuple[str, MistralChatEndpoint | None]:
    """Strip trailing `/v1` or `/openai/v1` so routes never double-join `/v1/v1`."""
    val = raw.strip().rstrip("/")
    if val.endswith("/openai/v1"):
        return val[: -len("/openai/v1")], "openai_compat"
    if val.endswith("/openai"):
        return val[: -len("/openai")], "openai_compat"
    if val.endswith("/v1"):
        return val[: -len("/v1")], None
    return val, None

DEFAULT_MISTRAL_PREFIXES = [
    "mistral-",
    "codestral-",
    "devstral-",
    "pixtral-",
    "ministral-",
    "magistral-",
    "deepseek-v4",
    "zai-glm-",
    "glm-5",
]

SECRET_MASK = "••••••••"


def _parse_bool(val: str | None, default: bool) -> bool:
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _parse_csv(raw: str | None, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return items


def parse_model_limits(raw: str | dict[str, Any] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    if not raw:
        return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if str(k).strip() and n > 0:
                out[str(k).strip().lower()] = n
        return out
    for item in str(raw).split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, _, val = item.partition("=")
        try:
            n = int(val.strip())
        except ValueError:
            continue
        if name.strip() and n > 0:
            out[name.strip().lower()] = n
    return out


@dataclass
class GatewayConfig:
    # Upstream routing
    upstream_mode: UpstreamMode = "direct_mistral"
    upstream_base: str = "https://api.mistral.ai"
    passthrough_auth: bool = True
    upstream_api_key: str = ""
    gateway_key: str = ""
    admin_token: str = ""

    # Mistral OpenAPI 3.1.0 compatibility
    mistral_chat_endpoint: MistralChatEndpoint = "native"
    strict_mistral_whitelist: bool = True
    responses_mode: ResponsesMode = "auto"
    normalize_content_arrays: bool = True
    rewrite_responses_reasoning: bool = True
    mistral_models: list[str] = field(default_factory=list)
    mistral_prefixes: list[str] = field(default_factory=lambda: list(DEFAULT_MISTRAL_PREFIXES))

    # Image & payload compaction
    keep_images_mistral: int = 4
    max_body_bytes_mistral: int = 3 * 1024 * 1024
    keep_images_other: int = 8
    max_body_bytes_other: int = 10 * 1024 * 1024

    # Timeouts & concurrency
    upstream_timeout: float = 300.0
    concurrency_wait_seconds: float = 120.0
    model_concurrency_limits: dict[str, int] = field(default_factory=dict)

    # Persistence path (excluded from serialized dict)
    config_path: str = ""

    def effective_responses_mode(self) -> Literal["native", "convert"]:
        if self.responses_mode in ("native", "convert"):
            return self.responses_mode
        return "native" if self.upstream_mode == "direct_mistral" else "convert"

    def is_mistral_model(self, model: str | None) -> bool:
        if self.upstream_mode == "direct_mistral":
            return True
        if not model:
            return False
        m = model.strip().lower()
        if m in self.mistral_models:
            return True
        return any(m.startswith(p) for p in self.mistral_prefixes)

    def to_dict(self, mask_secrets: bool = False) -> dict[str, Any]:
        d = asdict(self)
        d.pop("config_path", None)
        if mask_secrets:
            for k in ("upstream_api_key", "gateway_key", "admin_token"):
                d[f"has_{k}"] = bool(d.get(k))
                d[k] = SECRET_MASK if d.get(k) else ""
        return d

    def apply_patch(self, patch: dict[str, Any]) -> None:
        if "upstream_mode" in patch and patch["upstream_mode"] in ("direct_mistral", "new_api"):
            self.upstream_mode = patch["upstream_mode"]
        if "upstream_base" in patch and isinstance(patch["upstream_base"], str):
            val, inferred_ep = normalize_upstream_base(patch["upstream_base"])
            if val:
                self.upstream_base = val
            if inferred_ep and "mistral_chat_endpoint" not in patch:
                self.mistral_chat_endpoint = inferred_ep
        if "mistral_chat_endpoint" in patch and patch["mistral_chat_endpoint"] in ("native", "openai_compat"):
            self.mistral_chat_endpoint = patch["mistral_chat_endpoint"]
        if "passthrough_auth" in patch:
            self.passthrough_auth = bool(patch["passthrough_auth"])

        for secret_field in ("upstream_api_key", "gateway_key", "admin_token"):
            if patch.get(f"clear_{secret_field}"):
                setattr(self, secret_field, "")
            elif secret_field in patch and isinstance(patch[secret_field], str):
                incoming = patch[secret_field].strip()
                if incoming and incoming != SECRET_MASK:
                    setattr(self, secret_field, incoming)

        if "strict_mistral_whitelist" in patch:
            self.strict_mistral_whitelist = bool(patch["strict_mistral_whitelist"])
        if "responses_mode" in patch and patch["responses_mode"] in ("auto", "native", "convert"):
            self.responses_mode = patch["responses_mode"]
        if "normalize_content_arrays" in patch:
            self.normalize_content_arrays = bool(patch["normalize_content_arrays"])
        if "rewrite_responses_reasoning" in patch:
            self.rewrite_responses_reasoning = bool(patch["rewrite_responses_reasoning"])

        if "mistral_models" in patch:
            val = patch["mistral_models"]
            if isinstance(val, list):
                self.mistral_models = [str(x).strip().lower() for x in val if str(x).strip()]
            elif isinstance(val, str):
                self.mistral_models = _parse_csv(val, [])
        if "mistral_prefixes" in patch:
            val = patch["mistral_prefixes"]
            if isinstance(val, list):
                self.mistral_prefixes = [str(x).strip().lower() for x in val if str(x).strip()]
            elif isinstance(val, str):
                self.mistral_prefixes = _parse_csv(val, [])

        for int_field, lo, hi in (
            ("keep_images_mistral", 0, 50),
            ("keep_images_other", 0, 100),
            ("max_body_bytes_mistral", 64 * 1024, 100 * 1024 * 1024),
            ("max_body_bytes_other", 64 * 1024, 100 * 1024 * 1024),
        ):
            if int_field in patch:
                try:
                    n = int(patch[int_field])
                    setattr(self, int_field, max(lo, min(hi, n)))
                except (TypeError, ValueError):
                    pass

        for float_field, lo, hi in (
            ("upstream_timeout", 5.0, 1800.0),
            ("concurrency_wait_seconds", 1.0, 600.0),
        ):
            if float_field in patch:
                try:
                    n = float(patch[float_field])
                    setattr(self, float_field, max(lo, min(hi, n)))
                except (TypeError, ValueError):
                    pass

        if "model_concurrency_limits" in patch:
            self.model_concurrency_limits = parse_model_limits(patch["model_concurrency_limits"])

    def save(self) -> bool:
        if not self.config_path:
            return False
        p = Path(self.config_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(mask_secrets=False), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(tmp, 0o600)
        tmp.replace(p)
        return True

    @classmethod
    def load(cls, config_path: str | None = None) -> "GatewayConfig":
        cfg = cls()
        resolved_path = config_path or os.environ.get("CONFIG_PATH", "")
        cfg.config_path = resolved_path

        # 1) Load env defaults first
        mode_env = os.environ.get("UPSTREAM_MODE", "").strip()
        if mode_env in ("direct_mistral", "new_api"):
            cfg.upstream_mode = mode_env  # type: ignore[assignment]
        elif os.environ.get("UPSTREAM_BASE", "").startswith("http://new-api"):
            cfg.upstream_mode = "new_api"

        if os.environ.get("UPSTREAM_BASE"):
            base_val, inferred_ep = normalize_upstream_base(os.environ["UPSTREAM_BASE"])
            if base_val:
                cfg.upstream_base = base_val
            if inferred_ep:
                cfg.mistral_chat_endpoint = inferred_ep
        elif cfg.upstream_mode == "new_api":
            cfg.upstream_base = "http://new-api:3000"

        ep_env = os.environ.get("MISTRAL_CHAT_ENDPOINT", "").strip()
        if ep_env in ("native", "openai_compat"):
            cfg.mistral_chat_endpoint = ep_env  # type: ignore[assignment]

        cfg.passthrough_auth = _parse_bool(os.environ.get("PASSTHROUGH_AUTH"), True)
        cfg.upstream_api_key = os.environ.get("UPSTREAM_API_KEY") or os.environ.get("UPSTREAM_KEY", "")
        cfg.gateway_key = os.environ.get("GATEWAY_KEY", "")
        cfg.admin_token = os.environ.get("ADMIN_TOKEN", "")
        cfg.strict_mistral_whitelist = _parse_bool(os.environ.get("STRICT_MISTRAL_WHITELIST"), True)

        resp_mode = os.environ.get("RESPONSES_MODE", "").strip()
        if resp_mode in ("auto", "native", "convert"):
            cfg.responses_mode = resp_mode  # type: ignore[assignment]

        cfg.normalize_content_arrays = _parse_bool(os.environ.get("NORMALIZE_CONTENT_ARRAYS"), True)
        cfg.rewrite_responses_reasoning = _parse_bool(os.environ.get("REWRITE_RESPONSES_REASONING"), True)
        if os.environ.get("MISTRAL_MODELS") is not None:
            cfg.mistral_models = _parse_csv(os.environ.get("MISTRAL_MODELS"), [])
        if os.environ.get("MISTRAL_PREFIXES") is not None:
            cfg.mistral_prefixes = _parse_csv(os.environ.get("MISTRAL_PREFIXES"), DEFAULT_MISTRAL_PREFIXES)

        for env_key, attr in (
            ("KEEP_IMAGES", "keep_images_mistral"),
            ("MAX_BODY_BYTES", "max_body_bytes_mistral"),
            ("KEEP_IMAGES_OTHER", "keep_images_other"),
            ("MAX_BODY_BYTES_OTHER", "max_body_bytes_other"),
        ):
            if os.environ.get(env_key):
                try:
                    setattr(cfg, attr, int(os.environ[env_key]))
                except ValueError:
                    pass

        for env_key, attr in (
            ("UPSTREAM_TIMEOUT", "upstream_timeout"),
            ("CONCURRENCY_WAIT_SECONDS", "concurrency_wait_seconds"),
        ):
            if os.environ.get(env_key):
                try:
                    setattr(cfg, attr, float(os.environ[env_key]))
                except ValueError:
                    pass

        if os.environ.get("MODEL_CONCURRENCY_LIMITS"):
            cfg.model_concurrency_limits = parse_model_limits(os.environ["MODEL_CONCURRENCY_LIMITS"])

        # 2) Overlay persisted JSON config if it exists
        if resolved_path:
            p = Path(resolved_path).expanduser()
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        cfg.apply_patch(data)
                except Exception:
                    pass

        return cfg
