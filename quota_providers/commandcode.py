"""CommandCode (Go / GOAT / Pro / Max / Ultra / Teams) quota fetcher.

CommandCode publishes no usage API, but its own CLI authenticates against
``https://api.commandcode.ai/alpha/*`` with the key it stores in
``~/.commandcode/auth.json``. This fetcher reads that key and maps the same
routes the CLI uses onto the plugin's window model:

  * ``credits.windowLimits.fiveHour`` / ``.weekly`` - the rolling caps every
    subscription carries (used / cap, reset as epoch-ms) -> "5h" / "Weekly"
    windows. Older payloads carried ``windowLimits`` next to ``credits``; both
    placements are read;
  * ``credits.monthlyCredits`` (REMAINING credit, not spent) against the plan's
    monthly pool -> the "Cycle" window, reset at the billing period end;
  * ``/alpha/usage/summary`` -> spend / request-count / token detail lines;
  * ``/alpha/whoami?limits=1`` -> ``org.id``: an organisation plan bills against
    the org, so every billing route is scoped with ``?orgId=`` when one exists.

Verified response shapes (live, 2026-09-18, Command Code CLI 1.53.0)::

    GET /alpha/billing/credits?orgId=<id>
    {"credits": {"planId": "individual-goat",
                 "monthlyCredits": 60.96, "purchasedCredits": 0, "freeCredits": 0,
                 "windowLimits": {"limited": true, "exceeded": null,
                                  "fiveHour": {"used": 3.6, "cap": 14, "exceeded": false,
                                               "resetAt": 1789700413797},
                                  "weekly":   {"used": 9.0, "cap": 35, "exceeded": false,
                                               "resetAt": 1789756647895}}}}

    GET /alpha/billing/subscriptions?orgId=<id>
    {"success": true, "data": {"status": "active", "planId": "individual-goat",
                               "currentPeriodStart": "2026-09-11T18:36:24.000Z",
                               "currentPeriodEnd": "2026-10-11T18:36:24.000Z"}}

    GET /alpha/usage/summary?orgId=<id>&since=<currentPeriodStart>
    {"totalCost": 8.95, "totalCount": 3086, "totalTokens": 329663340}

    GET /alpha/whoami?limits=1
    {"org": {"id": "...", "login": "..."}, "orgLimits": []}

Extra pay-as-you-go credits are never windowed by the provider, so they appear
as a detail line only. Plan pools mirror the table the CLI itself ships
(``getPlanTotalCredits``) and are documented at
https://commandcode.ai/docs/resources/usage-limits - re-check on plan changes.

The three waves (whoami -> credits + subscription -> summary) run under a single
wall-clock budget well inside the sweep's ``REFRESH_BUDGET_S``, and the two
billing calls are issued together: a hung endpoint costs a detail line, never
the windows already in hand.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

from .base import QuotaResult, QuotaWindow, build_unavailable
from .registry import register as _register

_PROVIDER = "commandcode"
_AUTH_PATH = os.path.join(os.path.expanduser("~"), ".commandcode", "auth.json")
_BASE = "https://api.commandcode.ai"
_UA = "command-code/1.53.0"
_TIMEOUT = 6.0  # per request
# Whole-provider wall clock. The sweep runs every provider concurrently under
# REFRESH_BUDGET_S (20s), so three waves of 12s would overrun it on a bad day.
_DEADLINE_S = 12.0

# planId -> (display name, monthly included credits in USD). Mirrors the plan
# table the Command Code CLI ships, legacy ids included: `individual-pro` is the
# first-gen $30 Pro and `individual-pro-v1` the current $80 one, so collapsing
# them onto one number would misreport the Cycle window for half the accounts.
_PLANS: dict[str, tuple[str, float]] = {
    "individual-go": ("Go", 10.0),
    "individual-goat": ("GOAT", 70.0),
    "individual-pro": ("Pro", 30.0),
    "individual-pro-v1": ("Pro", 80.0),
    "individual-provider": ("Provider", 15.0),
    "individual-max": ("Max", 150.0),
    "individual-ultra": ("Ultra", 300.0),
    "teams-pro": ("Teams Pro", 40.0),
}


class _Budget:
    """Wall clock for one provider run; every request is capped by what's left."""

    def __init__(self, seconds: Optional[float] = None) -> None:
        self._deadline = time.monotonic() + (_DEADLINE_S if seconds is None else seconds)

    def timeout(self) -> float:
        return min(_TIMEOUT, max(0.0, self._deadline - time.monotonic()))


def _load_api_key() -> Optional[str]:
    try:
        with open(_AUTH_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    if isinstance(data, dict):
        key = data.get("apiKey")
        if isinstance(key, str) and key.strip():
            return key.strip()
    return None


def _get(path: str, key: str, params: Optional[dict] = None, timeout: Optional[float] = None) -> Any:
    url = _BASE + path
    if params:
        query = urlencode([(k, v) for k, v in params.items() if v not in (None, "")])
        if query:
            url = f"{url}?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": _UA,
        },
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT if timeout is None else timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call(path: str, key: str, params: Optional[dict], budget: _Budget) -> tuple[Any, Optional[Exception]]:
    """One request inside the budget. Returns (payload, error) and never raises."""
    timeout = budget.timeout()
    if timeout <= 0:
        return None, TimeoutError(f"provider deadline reached before {path}")
    try:
        return _get(path, key, params, timeout), None
    except Exception as e:  # noqa: BLE001 - every failure becomes a typed card
        return None, e


def _await(future, budget: _Budget) -> tuple[Any, Optional[Exception]]:
    """Wait for one in-flight request inside the budget; a straggler is dropped."""
    try:
        return future.result(timeout=budget.timeout())
    except _FutureTimeout:
        return None, TimeoutError("provider deadline reached")
    except Exception as e:  # noqa: BLE001 - mirror _call's contract
        return None, e


def _failure_reason(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        if error.code in (401, 403):
            return "auth-failed"
        return f"http-{error.code}"
    return f"fetch-error:{type(error).__name__}"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _iso_from_ms(value: Any) -> Optional[str]:
    ms = _number(value)
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _iso_from_str(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _resolve_plan(plan_id: str) -> Optional[tuple[str, float]]:
    """Map a reported planId onto the plan table, or None when it is unknown."""
    key = plan_id.strip().lower().replace("_", "-")
    if not key:
        return None
    # No prefix fallback on purpose. The CLI's longest-prefix match would map a
    # future `individual-pro-v2` onto the legacy `individual-pro` $30 pool and
    # publish a wrong Cycle percent; an unknown id reports its balance instead.
    return _PLANS.get(key)


def _rolling_window(block: Any, label: str) -> Optional[QuotaWindow]:
    if not isinstance(block, dict):
        return None
    used = _number(block.get("used"))
    cap = _number(block.get("cap"))
    if used is None or cap is None or cap <= 0:
        return None
    pct = max(0.0, min(100.0, used / cap * 100.0))
    return QuotaWindow(
        label=label,
        used_percent=round(pct, 2),
        reset_at=_iso_from_ms(block.get("resetAt")),
    )


def _window_limits(credits_payload: dict, credit_block: dict) -> dict:
    """Rolling caps, from under ``credits`` (current) or beside it (older)."""
    for holder in (credit_block, credits_payload):
        limits = holder.get("windowLimits")
        if isinstance(limits, dict):
            return limits
    return {}


def _cycle_window(
    credit_block: dict, plan: Optional[tuple[str, float]], sub: Optional[dict]
) -> tuple[Optional[QuotaWindow], list[str]]:
    """Build the cycle window + its detail line from remaining credit + plan pool."""
    remaining = _number(credit_block.get("monthlyCredits"))
    if remaining is None:
        return None, []
    if plan is None:
        # Unknown plan: show the balance, never a fabricated denominator.
        return None, [f"Cycle credits left: ${remaining:.2f}"]
    pool = plan[1]
    if not (0.0 <= remaining <= pool):
        # Remaining above the pool means rollover/carry-over - the pool stops
        # being a meaningful denominator, so report the balance only.
        return None, [f"Cycle credits left: ${remaining:.2f} (pool ${pool:.2f})"]
    used_pct = max(0.0, min(100.0, (pool - remaining) / pool * 100.0))
    reset_at = _iso_from_str(sub.get("currentPeriodEnd")) if isinstance(sub, dict) else None
    window = QuotaWindow(label="Cycle", used_percent=round(used_pct, 2), reset_at=reset_at)
    return window, [f"${remaining:.2f} of ${pool:.2f} cycle credits left"]


def fetch_commandcode_quota() -> QuotaResult:
    """Registered fetcher: never raises, every failure is a typed card."""
    try:
        return _fetch()
    except Exception as e:
        return build_unavailable(_PROVIDER, f"fetch-error:{type(e).__name__}")


def _fetch() -> QuotaResult:
    key = _load_api_key()
    if not key:
        return build_unavailable(_PROVIDER, "no-credentials")

    budget = _Budget()

    # Wave 1: org scope. Team plans bill against the organisation, and the CLI
    # resolves the id the same way. Best-effort: individuals have no org and
    # lose nothing but the one request.
    whoami, _ = _call("/alpha/whoami", key, {"limits": "1"}, budget)
    org_id: Optional[str] = None
    if isinstance(whoami, dict) and isinstance(whoami.get("org"), dict):
        raw_org = whoami["org"].get("id")
        if isinstance(raw_org, str) and raw_org.strip():
            org_id = raw_org.strip()
    params = {"orgId": org_id}

    # Wave 2: credits are required, the subscription is not, and neither depends
    # on the other - issue them together so the budget covers one round trip.
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        credits_future = pool.submit(_call, "/alpha/billing/credits", key, params, budget)
        sub_future = pool.submit(_call, "/alpha/billing/subscriptions", key, params, budget)
        credits_payload, credits_error = _await(credits_future, budget)
        sub_payload, _ = _await(sub_future, budget)
    finally:
        # Never block the sweep on a straggler: its own socket timeout reaps it.
        pool.shutdown(wait=False)

    if credits_error is not None:
        return build_unavailable(_PROVIDER, _failure_reason(credits_error))
    if not isinstance(credits_payload, dict):
        return build_unavailable(_PROVIDER, "bad-json")

    sub: Optional[dict] = None
    if isinstance(sub_payload, dict):
        if isinstance(sub_payload.get("data"), dict):
            sub = sub_payload["data"]
        elif "planId" in sub_payload:
            sub = sub_payload

    credit_block = credits_payload.get("credits")
    credit_block = credit_block if isinstance(credit_block, dict) else {}

    # The CLI reads the plan id off credits first and the subscription second;
    # either can be the only one present.
    plan_id = ""
    if isinstance(credit_block.get("planId"), str):
        plan_id = credit_block["planId"].strip()
    if not plan_id and isinstance(sub, dict):
        plan_id = str(sub.get("planId") or "").strip()
    plan = _resolve_plan(plan_id) if plan_id else None

    # Wave 3: usage totals for the billing period. The period start comes from
    # the subscription, so this call is only as scoped as that response - the
    # label says which one it is.
    since = sub.get("currentPeriodStart") if isinstance(sub, dict) else None
    since = since if isinstance(since, str) and since.strip() else None
    summary, _ = _call("/alpha/usage/summary", key, {"orgId": org_id, "since": since}, budget)

    windows: list[QuotaWindow] = []
    details: list[str] = []

    limits = _window_limits(credits_payload, credit_block)
    if limits.get("limited") is False:
        details.append("No usage windows on this plan (pay-as-you-go)")
    else:
        for block, label in ((limits.get("fiveHour"), "5h"), (limits.get("weekly"), "Weekly")):
            window = _rolling_window(block, label)
            if window is not None:
                windows.append(window)
            if isinstance(block, dict) and block.get("exceeded") is True:
                details.append(f"{label} window exceeded - requests decline until it resets")

    cycle_window, cycle_details = _cycle_window(credit_block, plan, sub)
    if cycle_window is not None:
        windows.append(cycle_window)
    details.extend(cycle_details)
    if plan_id and plan is None:
        details.append(f"Unrecognized plan id: {plan_id}")

    purchased = _number(credit_block.get("purchasedCredits"))
    if purchased is not None and purchased > 0:
        details.append(f"Top-up credits: ${purchased:.2f} (never capped)")
    free = _number(credit_block.get("freeCredits"))
    if free is not None and free > 0:
        details.append(f"Free credits: ${free:.2f}")

    if isinstance(summary, dict):
        spent = _number(summary.get("totalCost"))
        count = _number(summary.get("totalCount"))
        tokens = _number(summary.get("totalTokens"))
        bits: list[str] = []
        if spent is not None:
            bits.append(f"${spent:.2f} spent")
        if count is not None:
            bits.append(f"{int(count):,} requests")
        if tokens is not None:
            bits.append(f"{tokens / 1_000_000:.1f}M tokens")
        if bits:
            # "Cycle so far" only when the totals really are period-scoped.
            prefix = "Cycle so far: " if since else "Usage so far (all time): "
            details.append(prefix + " · ".join(bits))

    if not windows and not details:
        return build_unavailable(_PROVIDER, "no-data")

    return QuotaResult(
        label=_PROVIDER,
        windows=windows,
        plan=plan[0] if plan else None,
        unavailable_reason=None,
        details=details,
    )


_register(_PROVIDER)(fetch_commandcode_quota)
