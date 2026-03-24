# Polymarket Arbitrage & Signal Detection Bot

A modular Python bot that detects mispricings between real-world BTC price data and Polymarket prediction market probabilities, generating ranked trade signals with full risk management.

---

## ⚠️ Risk Disclaimer

**This is experimental software. Prediction markets are inherently uncertain. Past edge does not guarantee future profit. You may lose your entire capital. Never trade with money you cannot afford to lose. Always run in simulation mode first.**

Key risks:
- **Latency risk**: By the time your order fills, the edge may have vanished
- **Model risk**: Implied certainty estimates are simplified — real BTC volatility is fat-tailed
- **Liquidity risk**: Polymarket markets can be thin; large orders move prices significantly  
- **Smart contract risk**: Polymarket runs on Polygon; bridge/contract failures are possible
- **Resolution risk**: Market resolvers may interpret conditions differently than you expect

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (Orchestrator)                  │
└────────────┬──────────────────────────────┬────────────────────┘
             │                              │
    ┌────────▼────────┐          ┌──────────▼──────────┐
    │  data_ingestion │          │   market_parser      │
    │  (Binance WS)   │          │   (Polymarket APIs)  │
    └────────┬────────┘          └──────────┬──────────┘
             │  PriceTick                   │  MarketCondition[]
             └──────────────┬───────────────┘
                            │
                 ┌──────────▼──────────┐
                 │  condition_engine   │  ← "Is BTC > $80k right now?"
                 └──────────┬──────────┘
                            │  ConditionResult (implied_certainty)
                 ┌──────────▼──────────┐
                 │  pricing_engine     │  ← Edge = certainty - market_price
                 └──────────┬──────────┘
                            │  PricingAnalysis (net_edge, Kelly size)
                 ┌──────────▼──────────┐
                 │  signal_generator   │  ← Filters, dedupes, ranks signals
                 └──────────┬──────────┘
                            │  TradeSignal[]
                 ┌──────────▼──────────┐
                 │   risk_manager      │  ← Max loss, position limits
                 └──────────┬──────────┘
                            │  RiskDecision
                 ┌──────────▼──────────┐
                 │  execution          │  ← Sim fill or live CLOB order
                 └─────────────────────┘

    ┌──────────────────────┐    ┌──────────────────────┐
    │  utils/logger.py     │    │  utils/metrics.py     │
    │  (structured JSON +  │    │  (counters, gauges,   │
    │   PnL ledger)        │    │   latency histograms) │
    └──────────────────────┘    └──────────────────────┘

    ┌──────────────────────┐
    │  core/simulator.py   │  ← Historical replay + synthetic GBM ticks
    └──────────────────────┘
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo>
cd polymarket-arb-bot
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Set credentials for live trading

```bash
export POLYMARKET_PRIVATE_KEY="0x..."   # Your Polygon wallet private key
```

**Never commit your private key. Use `.env` + `python-dotenv` in production.**

### 3. Run simulation mode (safe — no real orders)

```bash
# Quick simulation with synthetic BTC data
python main.py --mode simulate

# Simulation with 2000 ticks
python main.py --mode simulate --max-ticks 2000

# Generate a sample historical CSV, then simulate from it
python main.py --mode generate-data
python main.py --mode simulate --data data/historical/btc_prices.csv
```

### 4. Run live mode (reads real feeds, simulated orders by default)

```bash
# Live feeds, simulated execution (safe)
python main.py --mode live

# Live feeds, print signals only, no execution
python main.py --mode live --no-execute

# Live feeds + REAL orders (only after thorough testing!)
# Set execution.simulation_mode = False in config.py first
python main.py --mode live
```

---

## Configuration (`config.py`)

All parameters are in `config.py`. Key values to tune:

| Parameter | Default | Description |
|---|---|---|
| `signal.taker_fee_fraction` | `0.02` | Polymarket taker fee (~2%) |
| `signal.safety_margin` | `0.03` | Extra buffer above fees |
| `signal.min_implied_certainty` | `0.90` | Only act when 90%+ certain |
| `risk.max_capital_per_trade_usd` | `500` | Max size per trade |
| `risk.max_daily_loss_usd` | `1000` | Halts trading when hit |
| `risk.kelly_fraction` | `0.25` | Fractional Kelly (25% = conservative) |
| `risk.total_capital_usd` | `10000` | Used for Kelly sizing |
| `execution.simulation_mode` | `True` | **Set False for live trading** |
| `execution.max_slippage_fraction` | `0.01` | Cancel if slippage > 1% |
| `polymarket.min_market_liquidity_usd` | `5000` | Skip illiquid markets |

---

## How the Edge Calculation Works

```
BTC price:       $85,000
Market question: "Will BTC exceed $80,000 before March 2025?"
Polymarket YES:  $0.62  (market prices 62% probability)

Implied certainty (our model):
  - BTC is already $5,000 above threshold
  - Distance = 6.25% above threshold
  - Time discount for 30-day expiry + BTC vol
  → Certainty ≈ 0.92

Raw edge     = 0.92 - 0.62 = 0.30
Fee cost     = 0.02 (taker fee)
Safety margin= 0.03
Net edge     = 0.30 - 0.02 - 0.03 = 0.25

Kelly fraction f* = (0.92 × (1/0.62 - 1) - 0.08) / (1/0.62 - 1)
                  ≈ 0.68

Fractional Kelly (25%) = 0.17
Position size = 0.17 × $10,000 = $1,700 → capped at $500

Expected Value = 0.25 × $500 = $125
```

---

## Key Design Decisions

### Why not model upward BTC price movement?
The bot deliberately focuses on **already-met conditions**. Modeling whether BTC *will* hit a threshold requires a price prediction model — a separate, much harder problem. Keeping the scope narrow reduces model risk.

### Why fractional Kelly (25%)?
Full Kelly maximises long-run growth but leads to catastrophic drawdowns with any model error. 25% Kelly gives ~80% of the EV with far less variance. See: [Kelly Criterion — Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)

### Why the time discount?
Even if BTC is above a threshold *today*, a 30-day market still carries the risk of BTC falling back below before the resolution date. The discount is computed using a rough GBM model with ~70% annualised vol.

### Why dedup signals within 60 seconds?
Re-evaluating the same market on every tick would generate hundreds of redundant signals for the same opportunity. The dedup window prevents this without missing genuine new opportunities.

---

## Enabling Live Order Execution

1. Install the Polymarket CLOB client:
   ```bash
   pip install py-clob-client
   ```

2. Fund a Polygon wallet with USDC

3. Complete Polymarket KYC / API key setup at https://polymarket.com

4. Set your private key:
   ```bash
   export POLYMARKET_PRIVATE_KEY="0x..."
   ```

5. Implement the `_live_order()` stub in `core/execution.py` using `py_clob_client`

6. Set `execution.simulation_mode = False` in `config.py`

7. **Test extensively on small sizes before scaling up**

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Threshold parsing from market questions
- Condition engine evaluation (above/below/near-boundary)
- Time-to-expiry discount math
- Pricing engine edge + Kelly calculations
- Risk manager approvals/rejections
- Synthetic GBM tick distribution properties

---

## File Structure

```
polymarket-arb-bot/
├── main.py                   # Entry point + CLI
├── config.py                 # All configuration
├── requirements.txt
├── README.md
├── core/
│   ├── data_ingestion.py     # Binance WebSocket feed + price store
│   ├── market_parser.py      # Polymarket API client + market parsing
│   ├── condition_engine.py   # Condition evaluation + time discount
│   ├── pricing_engine.py     # Edge calculation + Kelly sizing
│   ├── signal_generator.py   # Pipeline orchestration + signal ranking
│   ├── risk_manager.py       # Pre-trade risk checks + position tracking
│   ├── execution.py          # Order placement (sim + live stub)
│   └── simulator.py          # Historical replay + GBM data generator
├── utils/
│   ├── logger.py             # Structured JSON logging + PnL ledger
│   └── metrics.py            # In-process counters/gauges/histograms
├── tests/
│   └── test_core.py          # Unit tests
├── logs/                     # Auto-created: bot.log, ledger.jsonl
└── data/
    └── historical/           # Place your BTC price CSV here
```

---

## Monitoring

All decisions are logged to `logs/bot.log` (JSON lines) and `logs/ledger.jsonl`.

Example signal log entry:
```json
{
  "ts": "2025-03-15T10:23:44Z",
  "level": "INFO",
  "logger": "signal_generator",
  "msg": "🟢 SIGNAL [sig_000001] market=test_mkt_0 side=YES price=0.620 certainty=0.920 net_edge=0.250 size=$500.00 EV=$125.0000",
  "market_id": "test_mkt_001",
  "edge": 0.25,
  "ev": 125.0
}
```

---

## Known Limitations & Future Work

- **No cross-market arbitrage**: Only single-market threshold signals
- **No options/spread pricing**: BTC options could sharpen the certainty model
- **No order book depth analysis**: Slippage is estimated, not modelled from real depth
- **Single asset**: Designed for BTC; extending to ETH/SOL requires minimal changes
- **No position management**: No stop-loss or take-profit on open positions
- **REST-only for Polymarket**: A WebSocket feed for Polymarket prices would reduce latency

---

## License

MIT — use at your own risk.
