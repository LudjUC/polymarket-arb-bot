"""
debug_markets.py — Run this to diagnose why 0 markets pass the tradeable filter.

Usage:
    python debug_markets.py

Prints:
  1. Raw API response for the first 3 markets (field-by-field)
  2. For ALL 100 markets: which exact check they fail
  3. A tally of rejection reasons
  4. The current BTC price for context
"""

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# 1. Get raw API data
# ---------------------------------------------------------------------------

BASE = "https://gamma-api.polymarket.com"
session = requests.Session()
session.headers.update({"User-Agent": "polymarket-arb-bot/debug/1.0"})

print("Fetching Gamma API…")
raw_all = []
for kw in ("bitcoin", "btc"):
    resp = session.get(
        f"{BASE}/markets",
        params={"active": "true", "closed": "false", "limit": "100", "question": kw},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    markets = data if isinstance(data, list) else data.get("markets", [])
    raw_all.extend(markets)

# deduplicate
seen = set()
markets = []
for m in raw_all:
    mid = m.get("id") or m.get("conditionId") or ""
    if mid and mid not in seen:
        seen.add(mid)
        markets.append(m)

print(f"Total unique markets returned: {len(markets)}\n")

# ---------------------------------------------------------------------------
# 2. Print the first 3 markets raw so we can see the real field structure
# ---------------------------------------------------------------------------

print("=" * 70)
print("RAW FIELD INSPECTION (first 3 markets)")
print("=" * 70)

FIELDS_WE_CARE_ABOUT = [
    "id", "conditionId", "question", "active", "closed",
    "endDate", "end_date_iso", "endDateIso",
    "liquidity", "volume",
    "tokens", "outcomePrices",
]

for i, m in enumerate(markets[:3]):
    print(f"\n--- Market {i+1} ---")
    for field in FIELDS_WE_CARE_ABOUT:
        val = m.get(field, "<MISSING>")
        if isinstance(val, str) and len(val) > 120:
            val = val[:120] + "…"
        print(f"  {field:20s}: {val!r}")
    # Also show any keys we didn't list
    extra = {k: v for k, v in m.items() if k not in FIELDS_WE_CARE_ABOUT}
    if extra:
        print(f"  {'(other keys)':20s}: {list(extra.keys())}")

# ---------------------------------------------------------------------------
# 3. Diagnose every market
# ---------------------------------------------------------------------------

import json as _json
import re
import math

def parse_tokens(m):
    tokens_raw = m.get("tokens", [])
    if isinstance(tokens_raw, str):
        try:
            tokens_raw = _json.loads(tokens_raw)
        except Exception:
            return [], []
    tokens = tokens_raw if isinstance(tokens_raw, list) else []
    yes = next((t for t in tokens if t.get("outcome", "").lower() == "yes"), None)
    no  = next((t for t in tokens if t.get("outcome", "").lower() == "no"),  None)
    return yes, no

def parse_prices(m, yes_token, no_token):
    outcome_prices_raw = m.get("outcomePrices", "[]")
    if isinstance(outcome_prices_raw, str):
        try:
            op = [float(p) for p in _json.loads(outcome_prices_raw)]
        except Exception:
            op = []
    elif isinstance(outcome_prices_raw, list):
        try:
            op = [float(p) for p in outcome_prices_raw]
        except Exception:
            op = []
    else:
        op = []

    yes_price = float(yes_token.get("price", 0) or 0) if yes_token else 0.0
    no_price  = float(no_token.get("price",  0) or 0) if no_token  else 0.0
    if yes_price == 0.0 and len(op) >= 2:
        yes_price, no_price = op[0], op[1]
    return yes_price, no_price

def parse_expiry(m):
    for field in ("endDate", "end_date_iso", "endDateIso", "gameStartTime"):
        val = m.get(field)
        if val:
            try:
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                return int(dt.timestamp())
            except Exception:
                pass
    return 0

NOW = time.time()
MIN_LIQUIDITY   = 500.0
MIN_EXPIRY_S    = 600.0

rejection_counter = Counter()
rejection_details = []

for m in markets:
    question = m.get("question", "")
    mid      = m.get("id") or m.get("conditionId") or ""

    reasons = []

    # --- tokens
    yes_token, no_token = parse_tokens(m)
    if not yes_token or not no_token:
        reasons.append("NO_TOKENS (yes or no token missing)")

    # --- prices
    if yes_token and no_token:
        yes_price, no_price = parse_prices(m, yes_token, no_token)
    else:
        yes_price = no_price = 0.0

    if yes_price <= 0:
        reasons.append(f"YES_PRICE_ZERO (yes={yes_price}, no={no_price})")
    if no_price <= 0:
        reasons.append(f"NO_PRICE_ZERO (yes={yes_price}, no={no_price})")

    # --- liquidity
    liquidity = float(m.get("liquidity", 0) or 0)
    if liquidity < MIN_LIQUIDITY:
        reasons.append(f"LOW_LIQUIDITY (${liquidity:.0f} < ${MIN_LIQUIDITY:.0f})")

    # --- expiry
    expiry_ts = parse_expiry(m)
    tte = expiry_ts - NOW
    if expiry_ts != 0 and tte < MIN_EXPIRY_S:
        if tte < 0:
            reasons.append(f"EXPIRED ({abs(tte/3600):.1f}h ago)")
        else:
            reasons.append(f"EXPIRY_TOO_SOON ({tte/60:.0f}m remaining, need {MIN_EXPIRY_S/60:.0f}m)")

    if not reasons:
        reasons.append("✅ PASSES ALL CHECKS")
    else:
        for r in reasons:
            rejection_counter[r.split(" (")[0]] += 1

    rejection_details.append({
        "question": question[:70],
        "mid": mid[:16],
        "yes_price": yes_price,
        "no_price": no_price,
        "liquidity": liquidity,
        "expiry_ts": expiry_ts,
        "tte_days": (expiry_ts - NOW) / 86400 if expiry_ts else None,
        "reasons": reasons,
    })

# ---------------------------------------------------------------------------
# 4. Print summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("REJECTION REASON TALLY")
print("=" * 70)
for reason, count in rejection_counter.most_common():
    print(f"  {count:3d}x  {reason}")

passing = [d for d in rejection_details if d["reasons"][0].startswith("✅")]
print(f"\n  {len(passing)} / {len(markets)} markets PASS all checks\n")

# ---------------------------------------------------------------------------
# 5. Show detail for first 20 markets
# ---------------------------------------------------------------------------

print("=" * 70)
print("PER-MARKET BREAKDOWN (first 20)")
print("=" * 70)
for d in rejection_details[:20]:
    status = "✅" if d["reasons"][0].startswith("✅") else "❌"
    tte_str = f"{d['tte_days']:.1f}d" if d["tte_days"] is not None else "no_expiry"
    print(
        f"  {status} yes={d['yes_price']:.3f} no={d['no_price']:.3f} "
        f"liq=${d['liquidity']:>8.0f} tte={tte_str:>8s}  "
        f"{d['question'][:55]}"
    )
    if not d["reasons"][0].startswith("✅"):
        for r in d["reasons"]:
            print(f"       ↳ {r}")

# ---------------------------------------------------------------------------
# 6. Show ANY markets that pass
# ---------------------------------------------------------------------------

if passing:
    print("\n" + "=" * 70)
    print("PASSING MARKETS")
    print("=" * 70)
    for d in passing:
        print(f"  yes={d['yes_price']:.3f} no={d['no_price']:.3f} "
              f"liq=${d['liquidity']:>8.0f}  {d['question']}")
else:
    print("\n⚠️  No markets pass the current filter thresholds.")
    print("    Possible fixes:")
    print("    • Lower min_market_liquidity_usd in config.py (currently $500)")
    print("    • Lower min_time_to_expiry_s in config.py (currently 600s)")
    print("    • Check that 'tokens' field contains YES/NO entries in API response")
    print("    • Polymarket may have changed their API response schema")

print()
