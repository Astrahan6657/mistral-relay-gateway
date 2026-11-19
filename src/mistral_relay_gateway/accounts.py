from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import SECRET_MASK


@dataclass
class QuotaBucket:
    id: str
    label: str
    remaining_pct: float  # 0..100
    reset_at: str = ""  # ISO-8601 timestamp or empty
    detail: str = ""  # human summary e.g. "$18.4 / $20.0"
    sub_tags: list[str] = field(default_factory=list)
    icon: str = "spark"  # spark | bolt | wallet | clock
    wide: bool = False


@dataclass
class AccountEntry:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str = ""
    email: str = ""
    api_key: str = ""
    upstream_base: str = ""  # optional per-account override
    plan: str = "API"
    org_name: str = ""
    workspace_name: str = ""
    priority: int = 50
    enabled: bool = True
    pinned: bool = False
    tags: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    quotas: list[QuotaBucket] = field(default_factory=list)
    last_checked_at: float = 0.0
    last_status: str = "unchecked"  # ok | 401 | 402 | 429 | error | unchecked
    last_error: str = ""
    cooldown_until: float = 0.0
    requests_served: int = 0
    raw_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, mask_secret: bool = False, include_raw: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if mask_secret:
            raw_key = self.api_key or ""
            d["key_preview"] = (raw_key[:4] + "…" + raw_key[-4:]) if len(raw_key) > 10 else (SECRET_MASK if raw_key else "")
            d["api_key"] = SECRET_MASK if raw_key else ""
        if not include_raw:
            d.pop("raw_snapshot", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccountEntry":
        raw_quotas = data.get("quotas") or []
        quotas = [
            QuotaBucket(**{k: v for k, v in q.items() if k in QuotaBucket.__dataclass_fields__})
            for q in raw_quotas
            if isinstance(q, dict)
        ]
        allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "quotas"}
        return cls(**allowed, quotas=quotas)


def _derive_model_tags(models: list[str]) -> list[str]:
    lower = [m.lower() for m in models]
    tags: list[str] = []
    checks = [
        ("GLM 5.3", lambda m: "glm-5-3" in m or "glm-5.3" in m),
        ("DeepSeek V4", lambda m: "deepseek-v4" in m),
        ("Mistral Large", lambda m: "mistral-large" in m),
        ("Codestral", lambda m: "codestral" in m),
        ("Devstral", lambda m: "devstral" in m),
        ("Magistral", lambda m: "magistral" in m),
        ("Pixtral", lambda m: "pixtral" in m),
    ]
    for label, pred in checks:
        if any(pred(m) for m in lower):
            tags.append(label)
    if not tags and models:
        tags.append(f"{len(models)} Models")
    return tags


def _clamp_pct(val: float) -> float:
    return round(max(0.0, min(100.0, val)), 1)


def _fmt_num(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def _build_quotas_from_probe(
    *,
    budget_data: dict[str, Any] | None,
    spend_data: dict[str, Any] | None,
    rate_data: dict[str, Any] | None,
    sub_data: dict[str, Any] | None,
    existing_live: list[QuotaBucket],
) -> list[QuotaBucket]:
    buckets: list[QuotaBucket] = []

    if isinstance(budget_data, dict):
        api_b = budget_data.get("api_budget")
        if isinstance(api_b, dict) and api_b.get("usage_percentage") is not None:
            used_pct = float(api_b.get("usage_percentage") or 0.0)
            init_b = api_b.get("initial_budget")
            cur = api_b.get("currency") or "USD"
            detail = f"总额 {init_b} {cur}" if init_b is not None else ""
            sub_tags = ["PAYG"] if api_b.get("payg_enabled") else ["API"]
            buckets.append(
                QuotaBucket(
                    id="api_budget",
                    label="API 预算配额",
                    remaining_pct=_clamp_pct(100.0 - used_pct),
                    reset_at=str(api_b.get("reset_at") or ""),
                    detail=detail,
                    sub_tags=sub_tags,
                    icon="spark",
                )
            )

        vibe_b = budget_data.get("vibe_budget")
        if isinstance(vibe_b, dict) and vibe_b.get("usage_percentage") is not None:
            used_pct = float(vibe_b.get("usage_percentage") or 0.0)
            init_b = vibe_b.get("initial_budget")
            cur = vibe_b.get("currency") or "USD"
            detail = f"总额 {init_b} {cur}" if init_b is not None else ""
            sub_tags = ["Vibe", "PAYG"] if vibe_b.get("payg_enabled") else ["Vibe"]
            buckets.append(
                QuotaBucket(
                    id="vibe_budget",
                    label="Vibe 编程配额",
                    remaining_pct=_clamp_pct(100.0 - used_pct),
                    reset_at=str(vibe_b.get("reset_at") or ""),
                    detail=detail,
                    sub_tags=sub_tags,
                    icon="bolt",
                )
            )

    if isinstance(spend_data, dict):
        limits = spend_data.get("limits") if isinstance(spend_data.get("limits"), dict) else {}
        comp = limits.get("completion") if isinstance(limits.get("completion"), dict) else {}
        cur = limits.get("currency") or "USD"
        total_u = comp.get("total_usage")
        if total_u is None:
            total_u = comp.get("usage")
        u_lim = comp.get("usage_limit")
        if comp.get("no_monthly_limit"):
            detail = f"本月已用 {total_u:.2f} {cur} / 无上限" if isinstance(total_u, (int, float)) else "无月度上限"
            buckets.append(
                QuotaBucket(
                    id="monthly_spend",
                    label="月度消费限额",
                    remaining_pct=100.0,
                    detail=detail,
                    sub_tags=["无上限"],
                    icon="wallet",
                )
            )
        elif isinstance(u_lim, (int, float)) and u_lim > 0:
            used_val = float(total_u or 0.0)
            rem_pct = _clamp_pct((1.0 - used_val / float(u_lim)) * 100.0)
            if comp.get("monthly_limit_reached"):
                rem_pct = 0.0
            buckets.append(
                QuotaBucket(
                    id="monthly_spend",
                    label="月度消费限额",
                    remaining_pct=rem_pct,
                    detail=f"已用 {used_val:.2f} / {u_lim:.2f} {cur}",
                    sub_tags=["月配额"],
                    icon="wallet",
                )
            )

    # Preserve live rate-limit buckets updated from response headers
    live_by_id = {b.id: b for b in existing_live if b.id.startswith("live_")}
    for lid in ("live_tpm", "live_monthly_tokens", "live_rpm"):
        if lid in live_by_id:
            buckets.append(live_by_id[lid])

    if isinstance(rate_data, dict):
        rps = rate_data.get("requests_per_second")
        tlm = rate_data.get("tokens_limits_by_model")
        sub_tags: list[str] = []
        if isinstance(rps, int) and rps > 0:
            sub_tags.append(f"{rps} RPS")
        if isinstance(tlm, dict) and tlm:
            sample_models = list(tlm.keys())[:3]
            sub_tags.extend(sample_models)
        if sub_tags:
            buckets.append(
                QuotaBucket(
                    id="rate_limits",
                    label="模型速率配额",
                    remaining_pct=100.0,
                    detail=f"覆盖 {len(tlm) if isinstance(tlm, dict) else 0} 个模型档位",
                    sub_tags=sub_tags[:4],
                    icon="spark",
                    wide=True,
                )
            )

    # Fallback wide bar if API key has model access but restricted billing scope
    if not buckets and isinstance(sub_data, dict):
        plan_name = str(sub_data.get("plan_name") or "Active")
        buckets.append(
            QuotaBucket(
                id="subscription_active",
                label=f"订阅状态 · {plan_name}",
                remaining_pct=100.0,
                detail="PAYG 已开启" if sub_data.get("payg_enabled") else "已激活（额度随实际请求头实时更新）",
                sub_tags=["PAYG"] if sub_data.get("payg_enabled") else ["在线"],
                icon="spark",
                wide=True,
            )
        )

    return buckets


class AccountPool:
    """
    Multi-account pool with priority + round-robin selection, cooldown on 402/429,
    live rate-limit header ingestion, and Mistral OpenAPI 3.1.0 quota probing.
    """

    def __init__(self, store_path: str = ""):
        self.store_path: str = store_path
        self._accounts: dict[str, AccountEntry] = {}
        self._rr_cursor: int = 0
        self.load()

    def load(self) -> None:
        if not self.store_path:
            return
        p = Path(self.store_path).expanduser()
        if not p.is_file():
            return
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            items = raw.get("accounts", []) if isinstance(raw, dict) else raw
            if isinstance(items, list):
                self._accounts = {}
                for item in items:
                    if isinstance(item, dict) and item.get("id"):
                        acc = AccountEntry.from_dict(item)
                        self._accounts[acc.id] = acc
        except Exception:
            pass

    def save(self) -> bool:
        if not self.store_path:
            return False
        p = Path(self.store_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        payload = {
            "accounts": [a.to_dict(mask_secret=False, include_raw=False) for a in self.list_entries()],
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(p)
        return True

    def list_entries(self) -> list[AccountEntry]:
        return sorted(
            self._accounts.values(),
            key=lambda a: (not a.pinned, not a.enabled, -a.priority, a.name or a.email or a.id),
        )

    def get(self, account_id: str) -> AccountEntry | None:
        return self._accounts.get(account_id)

    def has_enabled(self) -> bool:
        return any(a.enabled and a.api_key for a in self._accounts.values())

    def add(
        self,
        *,
        api_key: str,
        name: str = "",
        email: str = "",
        priority: int = 50,
        upstream_base: str = "",
        plan: str = "API",
        tags: list[str] | None = None,
        enabled: bool = True,
    ) -> AccountEntry:
        clean_key = api_key.strip()
        acc = AccountEntry(
            name=name.strip() or email.strip() or f"mistral-{uuid.uuid4().hex[:6]}",
            email=email.strip(),
            api_key=clean_key,
            upstream_base=upstream_base.strip().rstrip("/"),
            priority=max(0, min(999, int(priority))),
            plan=plan.strip() or "API",
            tags=[t.strip() for t in (tags or []) if t.strip()],
            enabled=bool(enabled),
        )
        self._accounts[acc.id] = acc
        self.save()
        return acc

    def update(self, account_id: str, patch: dict[str, Any]) -> AccountEntry | None:
        acc = self._accounts.get(account_id)
        if not acc:
            return None
        if "name" in patch and isinstance(patch["name"], str):
            acc.name = patch["name"].strip() or acc.name
        if "email" in patch and isinstance(patch["email"], str):
            acc.email = patch["email"].strip()
        if "api_key" in patch and isinstance(patch["api_key"], str):
            k = patch["api_key"].strip()
            if k and k != SECRET_MASK:
                acc.api_key = k
        if "upstream_base" in patch and isinstance(patch["upstream_base"], str):
            acc.upstream_base = patch["upstream_base"].strip().rstrip("/")
        if "plan" in patch and isinstance(patch["plan"], str):
            acc.plan = patch["plan"].strip() or acc.plan
        if "priority" in patch:
            try:
                acc.priority = max(0, min(999, int(patch["priority"])))
            except (TypeError, ValueError):
                pass
        if "enabled" in patch:
            acc.enabled = bool(patch["enabled"])
            if acc.enabled:
                acc.cooldown_until = 0.0
        if "pinned" in patch:
            if patch["pinned"]:
                for other in self._accounts.values():
                    other.pinned = False
            acc.pinned = bool(patch["pinned"])
        if "tags" in patch:
            val = patch["tags"]
            if isinstance(val, list):
                acc.tags = [str(x).strip() for x in val if str(x).strip()]
            elif isinstance(val, str):
                acc.tags = [x.strip() for x in val.split(",") if x.strip()]
        self.save()
        return acc

    def delete(self, account_id: str) -> bool:
        if account_id in self._accounts:
            self._accounts.pop(account_id)
            self.save()
            return True
        return False

    def activate(self, account_id: str) -> AccountEntry | None:
        acc = self._accounts.get(account_id)
        if not acc:
            return None
        for other in self._accounts.values():
            other.pinned = False
        acc.pinned = True
        acc.enabled = True
        acc.cooldown_until = 0.0
        self.save()
        return acc

    def select_account(
        self,
        model: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> AccountEntry | None:
        """
        Select the best available account:
        1. Pinned healthy account first
        2. Highest priority tier (round-robin among accounts sharing that priority)
        3. Prefers accounts whose discovered model list includes `model`
        """
        now = time.time()
        excl = exclude_ids or set()
        candidates = [
            a
            for a in self._accounts.values()
            if a.enabled and a.api_key and a.id not in excl and a.cooldown_until <= now
        ]
        if not candidates:
            return None

        if model:
            m_low = model.strip().lower()
            model_matched = [
                a for a in candidates if not a.models or any(x.lower() == m_low for x in a.models)
            ]
            if model_matched:
                candidates = model_matched

        pinned = [a for a in candidates if a.pinned]
        if pinned:
            return pinned[0]

        max_prio = max(a.priority for a in candidates)
        top_tier = sorted([a for a in candidates if a.priority == max_prio], key=lambda a: a.id)
        chosen = top_tier[self._rr_cursor % len(top_tier)]
        self._rr_cursor += 1
        return chosen

    def record_outcome(
        self,
        account_id: str,
        *,
        status: int,
        headers: httpx.Headers | dict[str, str] | None = None,
        error: str = "",
    ) -> None:
        acc = self._accounts.get(account_id)
        if not acc:
            return
        now = time.time()
        acc.requests_served += 1

        if status == 402:
            acc.last_status = "402"
            acc.last_error = error or "Payment Required (quota/balance exhausted)"
            acc.cooldown_until = now + 600.0
        elif status == 429:
            retry_after = 60.0
            if headers:
                ra = headers.get("retry-after")
                if ra:
                    try:
                        retry_after = max(5.0, min(600.0, float(ra)))
                    except ValueError:
                        pass
            acc.last_status = "429"
            acc.last_error = error or f"Rate limited (cooldown {int(retry_after)}s)"
            acc.cooldown_until = now + retry_after
        elif status == 401:
            acc.last_status = "401"
            acc.last_error = error or "Invalid API Key"
            acc.cooldown_until = now + 300.0
        elif status < 400:
            acc.last_status = "ok"
            acc.last_error = ""
            acc.cooldown_until = 0.0

        if headers:
            self._ingest_rate_headers(acc, headers)

    def _ingest_rate_headers(self, acc: AccountEntry, headers: httpx.Headers | dict[str, str]) -> None:
        h = {k.lower(): v for k, v in headers.items()}

        def _upsert(bucket: QuotaBucket) -> None:
            for idx, b in enumerate(acc.quotas):
                if b.id == bucket.id:
                    acc.quotas[idx] = bucket
                    return
            acc.quotas.append(bucket)

        rem_min = h.get("x-ratelimit-remaining-tokens-minute")
        lim_min = h.get("x-ratelimit-limit-tokens-minute")
        if rem_min is not None and lim_min is not None:
            try:
                rem_val, lim_val = float(rem_min), float(lim_min)
                if lim_val > 0:
                    reset_iso = datetime.fromtimestamp(time.time() + 60, tz=timezone.utc).isoformat()
                    _upsert(
                        QuotaBucket(
                            id="live_tpm",
                            label="分钟 Token 配额",
                            remaining_pct=_clamp_pct((rem_val / lim_val) * 100.0),
                            reset_at=reset_iso,
                            detail=f"{_fmt_num(rem_val)} / {_fmt_num(lim_val)} TPM",
                            sub_tags=["1M"],
                            icon="bolt",
                        )
                    )
            except ValueError:
                pass

        rem_mo = h.get("x-ratelimit-remaining-tokens-month")
        lim_mo = h.get("x-ratelimit-limit-tokens-month")
        if rem_mo is not None and lim_mo is not None:
            try:
                rem_val, lim_val = float(rem_mo), float(lim_mo)
                if lim_val > 0:
                    _upsert(
                        QuotaBucket(
                            id="live_monthly_tokens",
                            label="月度 Token 配额",
                            remaining_pct=_clamp_pct((rem_val / lim_val) * 100.0),
                            detail=f"{_fmt_num(rem_val)} / {_fmt_num(lim_val)}",
                            sub_tags=["月配额"],
                            icon="spark",
                            wide=True,
                        )
                    )
            except ValueError:
                pass

    async def refresh_account(
        self,
        account_id: str,
        client: httpx.AsyncClient,
        default_upstream_base: str,
    ) -> AccountEntry | None:
        acc = self._accounts.get(account_id)
        if not acc:
            return None
        base = (acc.upstream_base or default_upstream_base).rstrip("/")
        if base.endswith("/openai/v1"):
            base = base[: -len("/openai/v1")]
        elif base.endswith("/v1"):
            base = base[: -len("/v1")]

        headers = {"Authorization": f"Bearer {acc.api_key}", "Accept": "application/json"}

        async def _get_json(path: str) -> tuple[int, Any]:
            try:
                r = await client.get(f"{base}{path}", headers=headers, timeout=15.0)
                if r.status_code == 200:
                    return 200, r.json()
                return r.status_code, r.text[:240]
            except Exception as exc:
                return 0, str(exc)

        (
            (st_who, d_who),
            (st_me, d_me),
            (st_mod, d_mod),
            (st_sub, d_sub),
            (st_bud, d_bud),
            (st_spl, d_spl),
            (st_rl, d_rl),
        ) = await asyncio.gather(
            _get_json("/v1/whoami"),
            _get_json("/v1/users/me"),
            _get_json("/v1/models"),
            _get_json("/v1/billing/subscription"),
            _get_json("/v1/billing/budget"),
            _get_json("/v1/admin/spend-limit"),
            _get_json("/v1/admin/rate-limit"),
        )

        acc.last_checked_at = time.time()
        acc.raw_snapshot = {
            "whoami": {"status": st_who, "data": d_who},
            "users_me": {"status": st_me, "data": d_me},
            "models": {"status": st_mod, "count": len(d_mod.get("data", [])) if isinstance(d_mod, dict) else 0},
            "subscription": {"status": st_sub, "data": d_sub},
            "budget": {"status": st_bud, "data": d_bud},
            "spend_limit": {"status": st_spl, "data": d_spl},
            "rate_limit": {"status": st_rl, "data": d_rl},
        }

        if st_mod == 401:
            acc.last_status = "401"
            acc.last_error = "Invalid API Key (401)"
            self.save()
            return acc

        if st_mod == 0:
            acc.last_status = "error"
            acc.last_error = str(d_mod)[:200]
            self.save()
            return acc

        # Extract identity
        if st_who == 200 and isinstance(d_who, dict):
            principal = d_who.get("principal") or {}
            if isinstance(principal, dict):
                if principal.get("email"):
                    acc.email = str(principal["email"])
                elif principal.get("name") and not acc.name:
                    acc.name = str(principal["name"])
            org = d_who.get("organization")
            if isinstance(org, dict) and org.get("name"):
                acc.org_name = str(org["name"])
            ws = d_who.get("workspace")
            if isinstance(ws, dict) and ws.get("name"):
                acc.workspace_name = str(ws["name"])
        elif st_me == 200 and isinstance(d_me, dict):
            if d_me.get("email"):
                acc.email = str(d_me["email"])
            ws = d_me.get("workspace")
            if isinstance(ws, dict) and ws.get("name"):
                acc.workspace_name = str(ws["name"])
            org = d_me.get("organization")
            if isinstance(org, dict) and org.get("name"):
                acc.org_name = str(org["name"])

        if acc.email and (not acc.name or acc.name.startswith("mistral-")):
            acc.name = acc.email

        # Extract models & model family tags
        if st_mod == 200 and isinstance(d_mod, dict):
            acc.models = [str(m["id"]) for m in d_mod.get("data", []) if isinstance(m, dict) and m.get("id")]
            auto_tags = _derive_model_tags(acc.models)
            # Keep user custom tags that aren't auto-generated model count tags
            custom = [t for t in acc.tags if t not in auto_tags and not t.endswith(" Models")]
            acc.tags = (auto_tags + custom)[:8]

        # Extract plan badge
        if st_sub == 200 and isinstance(d_sub, dict):
            if d_sub.get("enterprise"):
                acc.plan = "ENTERPRISE"
            elif d_sub.get("plan_name"):
                acc.plan = str(d_sub["plan_name"]).upper()
            elif d_sub.get("payg_enabled"):
                acc.plan = "PAYG"

        acc.quotas = _build_quotas_from_probe(
            budget_data=d_bud if st_bud == 200 else None,
            spend_data=d_spl if st_spl == 200 else None,
            rate_data=d_rl if st_rl == 200 else None,
            sub_data=d_sub if st_sub == 200 else None,
            existing_live=acc.quotas,
        )

        # If spend-limit says monthly limit reached, flag 402
        if st_spl == 200 and isinstance(d_spl, dict):
            comp = (d_spl.get("limits") or {}).get("completion") or {}
            if comp.get("monthly_limit_reached"):
                acc.last_status = "402"
                acc.last_error = "Monthly spend limit reached"
            else:
                acc.last_status = "ok"
                acc.last_error = ""
                acc.cooldown_until = 0.0
        elif st_mod == 200:
            acc.last_status = "ok"
            acc.last_error = ""
            acc.cooldown_until = 0.0

        self.save()
        return acc

    def summary(self) -> dict[str, Any]:
        now = time.time()
        total = len(self._accounts)
        enabled = sum(1 for a in self._accounts.values() if a.enabled)
        healthy = sum(
            1 for a in self._accounts.values() if a.enabled and a.api_key and a.cooldown_until <= now
        )
        cooling = sum(
            1 for a in self._accounts.values() if a.enabled and a.cooldown_until > now
        )
        return {
            "total": total,
            "enabled": enabled,
            "healthy": healthy,
            "cooling": cooling,
        }
