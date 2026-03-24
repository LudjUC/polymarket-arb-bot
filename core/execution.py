"""
core/execution.py — Order placement and fill management.

Supports two modes:
  - Simulation: logs orders, assumes fill at signal price (with slippage noise).
  - Live: calls Polymarket CLOB API to place limit orders.

IMPORTANT: Live trading requires a funded Polygon wallet and Polymarket
API credentials. Never run live mode without thorough testing.

Failure scenarios handled:
  - Slippage exceeded: order cancelled
  - Price moved unfavorably before fill: order cancelled
  - Timeout: unfilled orders cancelled
  - API errors: logged, position NOT opened
"""

import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from config import CONFIG
from core.risk_manager import OpenPosition, RiskDecision, RiskManager
from core.signal_generator import TradeSignal
from utils.logger import get_logger
from utils.metrics import METRICS, LatencyTimer

log = get_logger("execution")


class OrderStatus(Enum):
    PENDING = auto()
    FILLED = auto()
    PARTIAL = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class OrderResult:
    signal_id: str
    status: OrderStatus
    filled_size_usd: float
    fill_price: float
    slippage_fraction: float
    latency_ms: float
    notes: str = ""


class ExecutionEngine:
    """
    Routes trade signals through risk approval and places orders.
    """

    def __init__(self, risk_manager: RiskManager) -> None:
        self._risk = risk_manager
        self._cfg = CONFIG.execution
        self._exec_cfg = CONFIG.execution

    # ------------------------------------------------------------------

    def execute(self, signal: TradeSignal) -> Optional[OrderResult]:
        """
        Main entry point.
        1. Run risk checks
        2. Place order (sim or live)
        3. Register position with risk manager
        Returns None if rejected or order fails.
        """
        decision: RiskDecision = self._risk.approve(signal)
        if not decision.approved:
            return None

        with LatencyTimer("execution_latency_ms", METRICS):
            if self._cfg.simulation_mode:
                result = self._simulate_fill(signal, decision.capped_size_usd)
            else:
                result = self._live_order(signal, decision.capped_size_usd)

        if result.status == OrderStatus.FILLED:
            pos = OpenPosition(
                signal_id=signal.signal_id,
                market_id=signal.market_id,
                token_id=signal.token_id,
                side=signal.target_outcome,
                size_usd=result.filled_size_usd,
                entry_price=result.fill_price,
            )
            self._risk.register_open_position(pos)
            METRICS.inc("orders_filled")

        elif result.status in (OrderStatus.CANCELLED, OrderStatus.FAILED):
            METRICS.inc("orders_cancelled_or_failed")
            log.warning(
                "Order not filled: signal=%s status=%s notes=%s",
                signal.signal_id,
                result.status.name,
                result.notes,
            )

        return result

    # ------------------------------------------------------------------
    # Simulation mode
    # ------------------------------------------------------------------

    def _simulate_fill(self, signal: TradeSignal, size_usd: float) -> OrderResult:
        """
        Simulates a fill with realistic slippage.
        Adds random latency to mimic real-world network conditions.
        """
        sim_latency_ms = CONFIG.simulation.simulated_latency_ms + random.gauss(0, 20)
        time.sleep(sim_latency_ms / 1000)  # Actually wait so timing is realistic

        # Simulate slippage: market impact + spread
        # For a $500 order on a Polymarket market, slippage might be 0.5–2%
        base_slippage = signal.recommended_size_usd / 100_000  # ~0.5% at $500
        random_slippage = abs(random.gauss(0, 0.003))
        total_slippage = base_slippage + random_slippage

        fill_price = signal.market_price * (1 + total_slippage)

        # Check slippage limit
        if total_slippage > self._cfg.max_slippage_fraction:
            log.warning(
                "SIM: Slippage %.4f exceeds limit %.4f for signal %s",
                total_slippage,
                self._cfg.max_slippage_fraction,
                signal.signal_id,
            )
            return OrderResult(
                signal_id=signal.signal_id,
                status=OrderStatus.CANCELLED,
                filled_size_usd=0.0,
                fill_price=fill_price,
                slippage_fraction=total_slippage,
                latency_ms=sim_latency_ms,
                notes=f"Slippage {total_slippage:.4f} > limit {self._cfg.max_slippage_fraction}",
            )

        # Simulate partial fill (10% chance for illiquid markets)
        fill_fraction = 1.0
        status = OrderStatus.FILLED
        if random.random() < 0.10:
            fill_fraction = random.uniform(0.4, 0.9)
            status = OrderStatus.PARTIAL
            log.info("SIM: Partial fill %.0f%% for signal %s", fill_fraction * 100, signal.signal_id)

        filled_usd = size_usd * fill_fraction

        log.info(
            "SIM FILL: signal=%s side=%s price=%.4f size=$%.2f slippage=%.4f latency=%.0fms",
            signal.signal_id,
            signal.target_outcome,
            fill_price,
            filled_usd,
            total_slippage,
            sim_latency_ms,
        )

        METRICS.observe("fill_slippage", total_slippage)

        return OrderResult(
            signal_id=signal.signal_id,
            status=status,
            filled_size_usd=filled_usd,
            fill_price=fill_price,
            slippage_fraction=total_slippage,
            latency_ms=sim_latency_ms,
        )

    # ------------------------------------------------------------------
    # Live mode (scaffold — requires Polymarket CLOB SDK)
    # ------------------------------------------------------------------

    def _live_order(self, signal: TradeSignal, size_usd: float) -> OrderResult:
        """
        Places a real limit order on Polymarket CLOB.
        
        STUB: Implement using py_clob_client or direct API calls.
        See: https://github.com/Polymarket/py-clob-client
        
        Steps:
        1. Fetch current best ask for token
        2. Verify price hasn't moved beyond slippage limit
        3. Submit limit order at (best_ask + 0.001) to ensure fill
        4. Poll for fill status with timeout
        5. Return result
        """
        log.warning(
            "LIVE ORDER STUB called for signal %s — implement with py_clob_client",
            signal.signal_id,
        )

        # Example structure (not functional):
        # from py_clob_client.client import ClobClient
        # from py_clob_client.clob_types import OrderArgs, OrderType
        #
        # client = ClobClient(
        #     host=CONFIG.polymarket.clob_api_url,
        #     key=CONFIG.polymarket.private_key,
        #     chain_id=CONFIG.polymarket.chain_id,
        # )
        # current_ask = client.get_order_book(signal.token_id).asks[0].price
        # if abs(current_ask - signal.market_price) / signal.market_price > max_slippage:
        #     return OrderResult(CANCELLED, ...)
        # order = client.create_limit_order(OrderArgs(
        #     token_id=signal.token_id,
        #     price=current_ask,
        #     size=size_usd / current_ask,
        #     side="BUY",
        # ))
        # result = client.post_order(order, OrderType.GTC)
        # ...poll for fill...

        return OrderResult(
            signal_id=signal.signal_id,
            status=OrderStatus.FAILED,
            filled_size_usd=0.0,
            fill_price=0.0,
            slippage_fraction=0.0,
            latency_ms=0.0,
            notes="Live order stub — not implemented",
        )
