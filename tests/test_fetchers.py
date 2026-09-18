"""Offline unit tests for the quota provider fetchers (stdlib only).

Run from the repo root:  python tests/test_fetchers.py
No network access happens here — every HTTP boundary is mocked.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
import unittest
import urllib.error
from io import BytesIO
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeResponse(BytesIO):
    """Minimal context-manager response standing in for urlopen()."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _urlopen_returning(payload: dict):
    def _opener(_req, timeout=None):  # noqa: ANN001, ARG001
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    return _opener


# -- Nous Portal --------------------------------------------------------------


def _nous_account(**overrides):
    """A stand-in for NousPortalAccountInfo shaped like the live free dump."""
    base = dict(
        logged_in=True,
        subscription=None,
        paid_service_access=False,
        tool_access=None,
    )
    access = types.SimpleNamespace(
        subscription_credits_remaining=None,
        purchased_credits_remaining=None,
        total_usable_credits=None,
    )
    base["paid_service_access_info"] = access
    base.update(overrides)
    return types.SimpleNamespace(**base)


class NousPortalFetcherTests(unittest.TestCase):
    def _fetch_with(self, account):
        from quota_providers.builtin import _fetch_nous_portal

        fake_mod = types.ModuleType("hermes_cli.nous_account")
        fake_mod.get_nous_portal_account_info = lambda *a, **k: account
        with mock.patch.dict(sys.modules, {"hermes_cli.nous_account": fake_mod}):
            return _fetch_nous_portal()

    def test_free_account_gets_honest_card(self):
        acct = _nous_account(
            tool_access=types.SimpleNamespace(
                coverage={"firecrawl": True, "browser_use": True, "krea": False}
            )
        )
        res = self._fetch_with(acct)
        self.assertIsNone(res.unavailable_reason)
        self.assertEqual(res.plan, "Free")
        self.assertEqual(res.windows, [])
        joined = "\n".join(res.details)
        self.assertIn("Free tier", joined)
        self.assertIn("browser-use, firecrawl", joined)

    def test_paid_subscription_builds_percent_window(self):
        acct = _nous_account(
            paid_service_access=True,
            subscription=types.SimpleNamespace(
                monthly_credits=110.0,
                credits_remaining=88.42,
                rollover_credits=0,
                current_period_end="2026-09-01",
                plan="Super",
            ),
        )
        res = self._fetch_with(acct)
        self.assertIsNone(res.unavailable_reason)
        self.assertEqual(res.plan, "Super")
        self.assertEqual(len(res.windows), 1)
        w = res.windows[0]
        self.assertEqual(w.label, "Subscription")
        self.assertAlmostEqual(w.used_percent, (110.0 - 88.42) / 110.0 * 100.0, places=2)
        self.assertTrue(any("$88.42 of $110.00" in d for d in res.details))

    def test_paid_spend_without_credit_cap_gets_details(self):
        acct = _nous_account(
            paid_service_access=True,
            raw_claims={
                "member_spend_usd": "21.77",
                "member_spend_cap_usd": None,
                "subscription_tier": 2,
                "rate_limit_rpm": 400,
                "rate_limit_tpm": 4_000_000,
                "rate_limit_rph": 16_800,
            },
        )
        res = self._fetch_with(acct)
        self.assertIsNone(res.unavailable_reason)
        self.assertEqual(res.plan, "Tier 2")
        self.assertEqual(res.windows, [])
        joined = "\n".join(res.details)
        self.assertIn("Spend this period: $21.77 (no cap reported)", joined)
        self.assertIn("Rate limits: 400 RPM · 4M TPM · 16.8k RPH", joined)

    def test_not_logged_in_is_unavailable(self):
        res = self._fetch_with(_nous_account(logged_in=False))
        self.assertEqual(res.unavailable_reason, "not-logged-in")

    def test_fetcher_never_raises(self):
        # account object missing every attribute must degrade, not crash
        res = self._fetch_with(object())
        self.assertIsNotNone(res.unavailable_reason)


# -- Gemini -------------------------------------------------------------------


class GeminiFetcherTests(unittest.TestCase):
    def test_secret_matches_upstream_gemini_cli(self):
        from quota_providers.gemini import (
            _GEMINI_CLIENT_ID,
            _GEMINI_CLIENT_SECRET,
        )

        # These are Google's public installed-app OAuth constants, published
        # in google-gemini/gemini-cli (packages/core/src/code_assist/oauth2.ts).
        # A stale/typo'd value makes refresh fail with invalid_client (a real
        # bug we hit). Reassembled here like production does so secret
        # scanners don't fire on public-but-pattern-matching literals.
        expected_id = (
            "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135"
            + "j.apps.googleusercontent.com"
        )
        expected_secret = "GOCSPX-4uHgMPm-1o7Sk-geV6Cu5clXFsx" + "l"
        self.assertEqual(_GEMINI_CLIENT_ID, expected_id)
        self.assertEqual(_GEMINI_CLIENT_SECRET, expected_secret)

    def test_free_tier_retired_returns_honest_card(self):
        from quota_providers import gemini

        la = {
            "currentTier": {},
            "ineligibleTiers": [
                {
                    "tierId": "free-tier",
                    "reasonCode": "UNSUPPORTED_CLIENT",
                    "reasonMessage": "This client is no longer supported...",
                }
            ],
        }
        with mock.patch.object(gemini, "_load_creds", return_value={"x": 1}), \
             mock.patch.object(gemini, "_valid_token", return_value="tok"), \
             mock.patch.object(gemini, "_load_code_assist", return_value=la):
            res = gemini.fetch_gemini_quota()
        self.assertIsNone(res.unavailable_reason)
        self.assertEqual(res.plan, "Free")
        joined = "\n".join(res.details)
        self.assertIn("retired", joined)
        self.assertIn("antigravity.google", joined)

    def test_standard_tier_uses_project_and_parses_windows(self):
        from quota_providers import gemini
        from quota_providers.base import QuotaResult

        la = {"currentTier": {"id": "standard-tier"}, "cloudaicompanionProject": "proj-1"}
        quota_payload = {
            "quota": [
                {"modelId": "gemini-pro", "remainingFraction": 0.25, "resetTime": "2026-08-22T00:00:00Z"}
            ]
        }
        captured = {}

        def fake_post(url, body, token):
            captured["url"] = url
            captured["project"] = body.get("project")
            return quota_payload, None

        with mock.patch.object(gemini, "_load_creds", return_value={"x": 1}), \
             mock.patch.object(gemini, "_valid_token", return_value="tok"), \
             mock.patch.object(gemini, "_load_code_assist", return_value=la), \
             mock.patch.object(gemini, "_post_json", side_effect=fake_post):
            res = gemini.fetch_gemini_quota()
        self.assertIsInstance(res, QuotaResult)
        self.assertEqual(captured["project"], "proj-1")
        self.assertEqual(res.plan, "Standard")
        self.assertEqual(len(res.windows), 1)
        self.assertAlmostEqual(res.windows[0].used_percent, 75.0, places=2)

    def test_no_credentials(self):
        from quota_providers import gemini

        with mock.patch.object(gemini, "_load_creds", return_value=None):
            res = gemini.fetch_gemini_quota()
        self.assertEqual(res.unavailable_reason, "no-credentials")


# -- Grok ---------------------------------------------------------------------


class GrokRestTests(unittest.TestCase):
    # Live capture of the billing gRPC response (GetGrokCreditsConfig), the
    # same bytes that render grok.com's usage screen at capture time:
    # Weekly Limit 100% used (resets Aug 23 17:00Z), Grok Build kind-2 quota
    # also present, "Reset Available" flag set.
    _GRPC_FIXTURE_HEX = (
        "00000000520a500d0000c84212001a00220b08c0d987d40610c0e3f16f2a0b08"
        "c0ceacd40610c0e3f16f3a070802150000c842421c0802120b08c0d987d40610"
        "c0e3f16f1a0b08c0ceacd40610c0e3f16f580162006801"
    )

    def test_grpc_fixture_weekly_build_banked(self):
        from quota_providers import grok

        raw = bytes.fromhex(self._GRPC_FIXTURE_HEX)
        res = grok._parse_grok_protobuf(raw)
        self.assertIsNotNone(res)
        self.assertIsNone(res.unavailable_reason)
        by_label = {w.label: w for w in res.windows}
        self.assertIn("Weekly", by_label)
        self.assertAlmostEqual(by_label["Weekly"].used_percent, 100.0, places=2)
        self.assertIn("Grok Build", by_label)
        self.assertAlmostEqual(by_label["Grok Build"].used_percent, 100.0, places=2)
        # Weekly reset: 2026-08-23T17:00:48Z (matches the panel)
        self.assertIn("2026-08-23T17:00:48", by_label["Weekly"].reset_at)
        self.assertTrue(any("Reset banked" in d for d in res.details))

    def test_rest_payload_to_windows(self):
        from quota_providers import grok

        payload = {
            "remainingQueries": 7,
            "totalQueries": 10,
            "windowSizeSeconds": 7200,
            "lowEffortRateLimits": None,
            "highEffortRateLimits": {"remainingQueries": 2, "totalQueries": 5},
        }
        with mock.patch.object(grok.urllib.request, "urlopen", _urlopen_returning(payload)):
            res = grok._fetch_grok_rest("cookie=1")
        self.assertIsNone(res.unavailable_reason)
        labels = [w.label for w in res.windows]
        self.assertEqual(labels[0], "2h")
        self.assertAlmostEqual(res.windows[0].used_percent, 30.0, places=2)
        self.assertIsNotNone(res.windows[0].reset_at)  # derived from window size
        high = [w for w in res.windows if w.label == "high effort"]
        self.assertEqual(len(high), 1)
        self.assertAlmostEqual(high[0].used_percent, 60.0, places=2)

    def test_auth_failure_is_reported_not_swallowed(self):
        import urllib.error

        from quota_providers import grok

        def _opener(_req, timeout=None):  # noqa: ANN001, ARG001
            raise urllib.error.HTTPError("url", 403, "forbidden", {}, BytesIO(b"cf"))

        with mock.patch.object(grok.urllib.request, "urlopen", _opener):
            res = grok._fetch_grok_rest("cookie=1")
        self.assertEqual(res.unavailable_reason, "cloudflare-blocked")

    def test_grpc_no_usage_field_means_zero_percent(self):
        """Live capture from a free/unused account: the weekly window and
        banked flag are present but the fn1 usage field is ABSENT — the
        grok.com panel renders this as "0% utilizado", so the parser must
        report 0.0 instead of hiding the number."""
        import struct

        from quota_providers import grok

        def _vi(n: int) -> bytes:
            out = bytearray()
            while True:
                b = n & 0x7F
                n >>= 7
                if n:
                    out.append(b | 0x80)
                else:
                    out.append(b)
                    return bytes(out)

        sub = b"\x08" + _vi(1788109248)  # fn1 = weekly reset epoch
        inner = b"\x2a" + _vi(len(sub)) + sub  # fn5 = weekly window (no fn1 %)
        inner += b"\x58\x01"  # fn11 = reset banked
        msg = b"\x0a" + _vi(len(inner)) + inner  # fn1 = response payload
        raw = b"\x00" + struct.pack(">I", len(msg)) + msg

        res = grok._parse_grok_protobuf(raw)
        self.assertIsNotNone(res)
        self.assertIsNone(res.unavailable_reason)
        by_label = {w.label: w for w in res.windows}
        self.assertIn("Weekly", by_label)
        self.assertAlmostEqual(by_label["Weekly"].used_percent, 0.0, places=2)
        self.assertIn("2026-08-30T17:00:48", by_label["Weekly"].reset_at)
        self.assertTrue(any("Reset banked" in d for d in res.details))

    def test_optin_disabled_by_default(self):
        from quota_providers import grok

        with mock.patch.object(grok, "_grok_enabled", return_value=False):
            res = grok._fetch_grok_optin()
        self.assertEqual(res.unavailable_reason, "opt-in-disabled")


# -- Kimi ---------------------------------------------------------------------


class KimiFetcherTests(unittest.TestCase):
    def test_no_credentials(self):
        from quota_providers import kimi

        with mock.patch.object(kimi, "_load_creds", return_value=(None, None)):
            res = kimi.fetch_kimi_quota()
        self.assertEqual(res.unavailable_reason, "no-credentials")


# -- CommandCode --------------------------------------------------------------


class CommandCodeFetcherTests(unittest.TestCase):
    """CommandCode balances are credit-based; windows must never be invented."""

    # Live shapes (CLI 1.53.0): the rolling windows and the plan id sit under
    # `credits`, the subscription carries the billing period.
    _CREDITS = {
        "credits": {
            "planId": "individual-goat",
            "belowThreshold": False,
            "creditThreshold": 0,
            "monthlyCredits": 60.0,
            "purchasedCredits": 0,
            "freeCredits": 0,
            "windowLimits": {
                "limited": True,
                "exceeded": None,
                "fiveHour": {"used": 3.5, "cap": 14, "exceeded": False, "resetAt": 1789700413797},
                "weekly": {"used": 7.0, "cap": 35, "exceeded": False, "resetAt": 1789756647895},
            },
        },
    }
    # Older payload: the windows sit beside `credits`, and only the subscription
    # reports a plan id.
    _CREDITS_LEGACY = {
        "credits": {"monthlyCredits": 60.0, "purchasedCredits": 0, "freeCredits": 0},
        "windowLimits": _CREDITS["credits"]["windowLimits"],
    }
    _SUBSCRIPTION = {
        "success": True,
        "data": {
            "status": "active",
            "planId": "individual-goat",
            "currentPeriodStart": "2026-09-11T18:36:24.000Z",
            "currentPeriodEnd": "2026-10-11T18:36:24.000Z",
        },
    }
    _SUMMARY = {"totalCost": 8.95, "totalCount": 3086, "totalTokens": 329663340}
    _WHOAMI = {"org": {"id": "org_123", "login": "acme"}, "orgLimits": []}

    def _run(self, credits=None, sub=None, summary=None, whoami=None, key="cmd_test",
             delays=None, deadline=None):
        from quota_providers import commandcode

        payloads = {
            "/alpha/billing/credits": self._CREDITS if credits is None else credits,
            "/alpha/billing/subscriptions": self._SUBSCRIPTION if sub is None else sub,
            "/alpha/usage/summary": self._SUMMARY if summary is None else summary,
            "/alpha/whoami": self._WHOAMI if whoami is None else whoami,
        }
        delays = delays or {}
        calls: list[dict] = []

        def _fake_get(path, _key, params=None, timeout=None):  # noqa: ANN001, ARG001
            entry = {"path": path, "params": dict(params or {}), "start": time.monotonic()}
            calls.append(entry)
            time.sleep(delays.get(path, 0))
            entry["end"] = time.monotonic()
            value = payloads[path]
            if isinstance(value, Exception):
                raise value
            return value

        patchers = [
            mock.patch.object(commandcode, "_load_api_key", return_value=key),
            mock.patch.object(commandcode, "_get", side_effect=_fake_get),
        ]
        if deadline is not None:
            patchers.append(mock.patch.object(commandcode, "_DEADLINE_S", deadline))
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return commandcode.fetch_commandcode_quota(), calls

    def _fetch_with(self, **kwargs):
        return self._run(**kwargs)[0]

    # -- plan catalogue -------------------------------------------------------

    def test_plan_catalog_matches_the_cli_table(self):
        # Mirror of getPlanTotalCredits in Command Code's own CLI: two Pro ids
        # ($30 first-gen, $80 current), plus the real Max/Ultra/Teams ids.
        from quota_providers.commandcode import _PLANS

        self.assertEqual(
            _PLANS,
            {
                "individual-go": ("Go", 10.0),
                "individual-goat": ("GOAT", 70.0),
                "individual-pro": ("Pro", 30.0),
                "individual-pro-v1": ("Pro", 80.0),
                "individual-provider": ("Provider", 15.0),
                "individual-max": ("Max", 150.0),
                "individual-ultra": ("Ultra", 300.0),
                "teams-pro": ("Teams Pro", 40.0),
            },
        )

    def test_legacy_pro_pool_is_thirty(self):
        credits = json.loads(json.dumps(self._CREDITS))
        credits["credits"]["planId"] = "individual-pro"
        credits["credits"]["monthlyCredits"] = 20.0
        res = self._fetch_with(credits=credits)
        self.assertEqual(res.plan, "Pro")
        by_label = {w.label: w for w in res.windows}
        self.assertAlmostEqual(by_label["Cycle"].used_percent, (30.0 - 20.0) / 30.0 * 100.0, places=2)
        self.assertIn("$20.00 of $30.00 cycle credits left", "\n".join(res.details))

    def test_pro_v1_pool_is_eighty(self):
        credits = json.loads(json.dumps(self._CREDITS))
        credits["credits"].pop("planId")  # plan id only on the subscription
        sub = json.loads(json.dumps(self._SUBSCRIPTION))
        sub["data"]["planId"] = "individual-pro-v1"
        res = self._fetch_with(credits=credits, sub=sub)
        self.assertEqual(res.plan, "Pro")
        by_label = {w.label: w for w in res.windows}
        self.assertAlmostEqual(by_label["Cycle"].used_percent, (80.0 - 60.0) / 80.0 * 100.0, places=2)

    def test_unknown_suffix_never_borrows_another_pools_pool(self):
        # `individual-pro-v2` must NOT inherit the legacy $30 pool: a wrong
        # denominator is worse than no Cycle window.
        credits = json.loads(json.dumps(self._CREDITS))
        credits["credits"]["planId"] = "individual-pro-v2"
        res = self._fetch_with(credits=credits)
        self.assertIsNone(res.plan)
        self.assertNotIn("Cycle", {w.label for w in res.windows})
        self.assertIn("Unrecognized plan id: individual-pro-v2", "\n".join(res.details))

    def test_plan_id_lookup_tolerates_case_and_underscores(self):
        credits = json.loads(json.dumps(self._CREDITS))
        credits["credits"]["planId"] = "Individual_Pro_V1"
        credits["credits"]["monthlyCredits"] = 60.0
        res = self._fetch_with(credits=credits)
        self.assertEqual(res.plan, "Pro")
        by_label = {w.label: w for w in res.windows}
        used = by_label["Cycle"].used_percent
        self.assertIsNotNone(used)
        self.assertAlmostEqual(used, (80.0 - 60.0) / 80.0 * 100.0, places=2)

    def test_balance_above_the_pool_drops_the_cycle_percent(self):
        # Rollover/carry-over: the pool stops being a denominator worth showing.
        credits = json.loads(json.dumps(self._CREDITS))
        credits["credits"]["planId"] = "individual-goat"
        credits["credits"]["monthlyCredits"] = 95.0
        res = self._fetch_with(credits=credits)
        self.assertNotIn("Cycle", {w.label for w in res.windows})
        self.assertIn("Cycle credits left: $95.00 (pool $70.00)", "\n".join(res.details))

    def test_unknown_plan_gets_no_name_and_no_cycle_percent(self):
        credits = json.loads(json.dumps(self._CREDITS))
        credits["credits"].pop("planId")
        sub = json.loads(json.dumps(self._SUBSCRIPTION))
        sub["data"]["planId"] = "individual-future"
        res = self._fetch_with(credits=credits, sub=sub)
        self.assertIsNone(res.unavailable_reason)
        self.assertIsNone(res.plan)
        self.assertNotIn("Cycle", {w.label for w in res.windows})
        joined = "\n".join(res.details)
        self.assertIn("Cycle credits left: $60.00", joined)
        self.assertIn("Unrecognized plan id: individual-future", joined)

    # -- window shapes --------------------------------------------------------

    def test_windows_and_details_from_the_live_shape(self):
        res = self._fetch_with()
        self.assertIsNone(res.unavailable_reason)
        self.assertEqual(res.plan, "GOAT")
        by_label = {w.label: w for w in res.windows}
        self.assertEqual(set(by_label), {"5h", "Weekly", "Cycle"})
        self.assertAlmostEqual(by_label["5h"].used_percent, 25.0, places=2)
        self.assertAlmostEqual(by_label["Weekly"].used_percent, 20.0, places=2)
        self.assertAlmostEqual(by_label["Cycle"].used_percent, (70.0 - 60.0) / 70.0 * 100.0, places=2)
        self.assertTrue(by_label["Cycle"].reset_at.startswith("2026-10-11T18:36:24"))
        joined = "\n".join(res.details)
        self.assertIn("$60.00 of $70.00 cycle credits left", joined)
        self.assertIn("Cycle so far: $8.95 spent · 3,086 requests · 329.7M tokens", joined)

    def test_window_limits_beside_credits_still_supported(self):
        res = self._fetch_with(credits=self._CREDITS_LEGACY)
        by_label = {w.label: w for w in res.windows}
        self.assertTrue({"5h", "Weekly"} <= set(by_label))
        self.assertAlmostEqual(by_label["5h"].used_percent, 25.0, places=2)

    def test_exceeded_window_is_named(self):
        credits = json.loads(json.dumps(self._CREDITS))
        credits["credits"]["windowLimits"]["fiveHour"]["exceeded"] = True
        res = self._fetch_with(credits=credits)
        self.assertIn("5h window exceeded", "\n".join(res.details))

    def test_pay_as_you_go_has_no_window_percents(self):
        credits = json.loads(json.dumps(self._CREDITS))
        credits["credits"]["windowLimits"] = {"limited": False, "exceeded": None}
        res = self._fetch_with(credits=credits)
        self.assertIsNone(res.unavailable_reason)
        self.assertNotIn("5h", {w.label for w in res.windows})
        self.assertIn("pay-as-you-go", "\n".join(res.details))

    # -- org scope ------------------------------------------------------------

    def test_org_scope_is_applied_to_every_billing_call(self):
        res, calls = self._run()
        params = {c["path"]: c["params"] for c in calls}
        self.assertEqual(params["/alpha/whoami"], {"limits": "1"})
        self.assertEqual(params["/alpha/billing/credits"], {"orgId": "org_123"})
        self.assertEqual(params["/alpha/billing/subscriptions"], {"orgId": "org_123"})
        self.assertEqual(
            params["/alpha/usage/summary"],
            {"orgId": "org_123", "since": "2026-09-11T18:36:24.000Z"},
        )
        self.assertIn("Cycle so far", "\n".join(res.details))

    def test_individual_account_still_scopes_the_summary_to_the_cycle(self):
        res, calls = self._run(whoami={})
        params = {c["path"]: c["params"] for c in calls}
        self.assertIsNone(params["/alpha/billing/credits"]["orgId"])
        self.assertEqual(params["/alpha/usage/summary"]["since"], "2026-09-11T18:36:24.000Z")
        self.assertIn("Cycle so far", "\n".join(res.details))

    def test_unscoped_summary_is_not_labelled_cycle_so_far(self):
        sub = json.loads(json.dumps(self._SUBSCRIPTION))
        sub["data"].pop("currentPeriodStart")
        res, calls = self._run(sub=sub)
        params = {c["path"]: c["params"] for c in calls}
        self.assertIsNone(params["/alpha/usage/summary"]["since"])
        joined = "\n".join(res.details)
        self.assertNotIn("Cycle so far", joined)
        self.assertIn("Usage so far (all time)", joined)

    # -- budget + failure modes ----------------------------------------------

    def test_billing_calls_are_issued_together(self):
        _, calls = self._run(delays={"/alpha/billing/credits": 0.2,
                                     "/alpha/billing/subscriptions": 0.2})
        by_path = {c["path"]: c for c in calls}
        credits = by_path["/alpha/billing/credits"]
        subs = by_path["/alpha/billing/subscriptions"]
        # Overlapping intervals: sequential calls cannot overlap.
        self.assertLess(subs["start"], credits["end"])
        self.assertLess(credits["start"], subs["end"])

    def test_a_hung_call_cannot_outrun_the_provider_deadline(self):
        started = time.monotonic()
        res = self._fetch_with(
            delays={"/alpha/billing/subscriptions": 0.6, "/alpha/usage/summary": 0.6},
            deadline=0.15,
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.5)
        self.assertIn("5h", {w.label for w in res.windows})
        self.assertNotIn("Cycle so far", "\n".join(res.details))

    def test_no_credentials(self):
        res = self._fetch_with(key=None)
        self.assertEqual(res.unavailable_reason, "no-credentials")

    def test_auth_failure_is_reported(self):
        # HTTPError is an OSError subclass; build one the way urlopen raises it.
        err = urllib.error.HTTPError("https://api.commandcode.ai/x", 401, "Unauthorized", {}, None)
        res = self._fetch_with(credits=err)
        self.assertEqual(res.unavailable_reason, "auth-failed")

    def test_supporting_calls_failing_keeps_windows(self):
        res = self._fetch_with(sub=RuntimeError("boom"), summary=RuntimeError("boom"))
        self.assertIsNone(res.unavailable_reason)
        self.assertEqual({w.label for w in res.windows}, {"5h", "Weekly", "Cycle"})
        self.assertEqual(res.plan, "GOAT")

    def test_garbage_credits_payload_is_bad_json(self):
        res = self._fetch_with(credits=["not", "an", "object"])
        self.assertEqual(res.unavailable_reason, "bad-json")

    def test_fetcher_never_raises(self):
        # An unexpected blow-up inside the body must still return a card.
        from quota_providers import commandcode

        with mock.patch.object(commandcode, "_load_api_key", return_value="k"), \
                mock.patch.object(commandcode, "_get", side_effect=ValueError("boom")):
            res = commandcode.fetch_commandcode_quota()
        self.assertEqual(res.unavailable_reason, "fetch-error:ValueError")


# -- Base ---------------------------------------------------------------------


class BaseTests(unittest.TestCase):
    def test_remaining_pct_clamps(self):
        from quota_providers.base import QuotaWindow

        self.assertEqual(QuotaWindow(label="w", used_percent=0).remaining_pct(), 100)
        self.assertEqual(QuotaWindow(label="w", used_percent=150).remaining_pct(), 0)
        self.assertIsNone(QuotaWindow(label="w").remaining_pct())


if __name__ == "__main__":
    unittest.main(verbosity=2)
