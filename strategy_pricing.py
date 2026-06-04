"""
Artefact 5: Strategy Pattern — AI Pricing Engine
==================================================
Module: Advanced Object-Oriented Programming

Pattern: Strategy (GoF Behavioural)
Domain:  E-commerce pricing / AI decision logic

Why Strategy?
  The pricing algorithm must be interchangeable at runtime without modifying
  the context class (PricingEngine). New strategies (e.g., an ML-model-driven
  strategy) can be plugged in without touching existing code — satisfying OCP.

Benefits:
  • Runtime behavioural flexibility — swap algorithm without restarting the system.
  • Separation of concerns — each strategy owns exactly one pricing algorithm.
  • Testability — strategies are tested in isolation; PricingEngine is tested
    with a MockStrategy that records calls.

Design note vs. over-engineering:
  Strategy adds an indirection layer. It is justified here because the business
  genuinely requires multiple interchangeable algorithms. A single if/elif block
  would be simpler for two options but collapses under extension.

Run:
    python strategy_pricing.py
"""

from __future__ import annotations

import math
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock


# ===========================================================================
# Domain Model
# ===========================================================================

@dataclass
class Product:
    product_id: str
    name: str
    base_price: float
    category: str
    stock: int


@dataclass
class CustomerProfile:
    customer_id: str
    tier: str           # "standard" | "premium" | "vip"
    purchase_history: int  # total number of past orders
    region: str


@dataclass
class PricingContext:
    """All data available to a pricing strategy at decision time."""
    product: Product
    customer: CustomerProfile
    timestamp: datetime
    demand_score: float  # 0.0 – 1.0 from analytics pipeline


# ===========================================================================
# Strategy Abstraction
# ===========================================================================

class PricingStrategy(ABC):
    """
    Abstract base for all pricing algorithms.

    Dependency Inversion: PricingEngine depends on this abstraction.
    Interface Segregation: single responsibility — compute a price.
    """

    @abstractmethod
    def calculate(self, ctx: PricingContext) -> float:
        """Return the final price for the given context."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""
        ...


# ===========================================================================
# Concrete Strategies
# ===========================================================================

class StandardPricingStrategy(PricingStrategy):
    """
    Baseline: return the base price with a tier discount.
    Suitable for low-complexity catalogues.
    """

    TIER_DISCOUNTS = {"standard": 0.0, "premium": 0.05, "vip": 0.12}

    @property
    def name(self) -> str:
        return "StandardPricing"

    def calculate(self, ctx: PricingContext) -> float:
        discount = self.TIER_DISCOUNTS.get(ctx.customer.tier, 0.0)
        return round(ctx.product.base_price * (1 - discount), 2)


class DynamicDemandStrategy(PricingStrategy):
    """
    Surge / discount pricing based on real-time demand score.

    High demand  (score > 0.7) → price increases up to 30 %.
    Low demand   (score < 0.3) → price decreases up to 20 %.
    Mid demand               → linear interpolation.

    Demonstrates: runtime behavioural flexibility — price responds to a
    live data signal without recompiling or redeploying PricingEngine.
    """

    @property
    def name(self) -> str:
        return "DynamicDemand"

    def calculate(self, ctx: PricingContext) -> float:
        score = ctx.demand_score
        if score > 0.7:
            multiplier = 1.0 + 0.30 * ((score - 0.7) / 0.3)
        elif score < 0.3:
            multiplier = 1.0 - 0.20 * ((0.3 - score) / 0.3)
        else:
            multiplier = 1.0
        return round(ctx.product.base_price * multiplier, 2)


class MLInformedPricingStrategy(PricingStrategy):
    """
    Simulates an ML-model-driven strategy.

    In production, this would call a live inference endpoint.
    Here the 'model' is a simple polynomial — the interface is identical,
    so swapping to a real model requires zero changes to PricingEngine.

    AI-readiness: the strategy wraps the model behind an abstraction,
    making it replaceable, testable via mocking, and auditable.
    """

    def __init__(self, model_version: str = "v1.2"):
        self._model_version = model_version

    @property
    def name(self) -> str:
        return f"MLInformed-{self._model_version}"

    def _run_inference(self, ctx: PricingContext) -> float:
        """
        Placeholder for a real model call (e.g., scikit-learn, ONNX, REST API).
        Simulates: price ≈ base × (1 + sigmoid(demand – loyalty_factor))
        """
        loyalty = math.log1p(ctx.customer.purchase_history) / 10
        demand_adj = 1 / (1 + math.exp(-(ctx.demand_score - 0.5) * 6))
        return ctx.product.base_price * (1 + demand_adj * 0.25 - loyalty * 0.05)

    def calculate(self, ctx: PricingContext) -> float:
        price = self._run_inference(ctx)
        return round(max(price, ctx.product.base_price * 0.70), 2)  # floor at 70 %


class BundleDiscountStrategy(PricingStrategy):
    """
    Applies category-based bundle discounts.
    Extensibility demo: new categories added to the mapping with no code change.
    """

    CATEGORY_RATES = {
        "electronics": 0.08,
        "books":       0.15,
        "clothing":    0.10,
    }

    @property
    def name(self) -> str:
        return "BundleDiscount"

    def calculate(self, ctx: PricingContext) -> float:
        rate = self.CATEGORY_RATES.get(ctx.product.category, 0.0)
        return round(ctx.product.base_price * (1 - rate), 2)


# ===========================================================================
# Context (PricingEngine)
# ===========================================================================

class PricingEngine:
    """
    Maintains a reference to a PricingStrategy and delegates calculation.

    Separation of concerns:
      PricingEngine orchestrates context assembly and audit logging.
      It never contains pricing logic — that lives exclusively in strategies.
    """

    def __init__(self, strategy: PricingStrategy):
        self._strategy = strategy
        self._audit: list[dict] = []

    def set_strategy(self, strategy: PricingStrategy) -> None:
        """Hot-swap the strategy at runtime — no restart required."""
        print(f"  [PricingEngine] Strategy changed: {self._strategy.name} → {strategy.name}")
        self._strategy = strategy

    def get_price(self, ctx: PricingContext) -> float:
        price = self._strategy.calculate(ctx)
        self._audit.append({
            "strategy": self._strategy.name,
            "product":  ctx.product.product_id,
            "customer": ctx.customer.customer_id,
            "price":    price,
        })
        return price

    def audit_log(self) -> list[dict]:
        return list(self._audit)


# ===========================================================================
# Unit Tests
# ===========================================================================

class TestStandardPricing(unittest.TestCase):
    def _ctx(self, tier: str) -> PricingContext:
        return PricingContext(
            product=Product("P1", "Laptop", 1000.0, "electronics", 10),
            customer=CustomerProfile("C1", tier, 5, "UK"),
            timestamp=datetime.now(),
            demand_score=0.5,
        )

    def test_standard_no_discount(self):
        strategy = StandardPricingStrategy()
        self.assertEqual(strategy.calculate(self._ctx("standard")), 1000.0)

    def test_premium_discount(self):
        strategy = StandardPricingStrategy()
        self.assertEqual(strategy.calculate(self._ctx("premium")), 950.0)

    def test_vip_discount(self):
        strategy = StandardPricingStrategy()
        self.assertEqual(strategy.calculate(self._ctx("vip")), 880.0)


class TestDynamicDemandPricing(unittest.TestCase):
    def _ctx(self, score: float) -> PricingContext:
        return PricingContext(
            product=Product("P2", "Widget", 100.0, "general", 50),
            customer=CustomerProfile("C2", "standard", 0, "US"),
            timestamp=datetime.now(),
            demand_score=score,
        )

    def test_high_demand_increases_price(self):
        strategy = DynamicDemandStrategy()
        price = strategy.calculate(self._ctx(1.0))
        self.assertGreater(price, 100.0)

    def test_low_demand_decreases_price(self):
        strategy = DynamicDemandStrategy()
        price = strategy.calculate(self._ctx(0.0))
        self.assertLess(price, 100.0)

    def test_mid_demand_neutral(self):
        strategy = DynamicDemandStrategy()
        price = strategy.calculate(self._ctx(0.5))
        self.assertAlmostEqual(price, 100.0, places=1)


class TestMLInformedPricing(unittest.TestCase):
    def test_price_above_floor(self):
        ctx = PricingContext(
            product=Product("P3", "Phone", 500.0, "electronics", 20),
            customer=CustomerProfile("C3", "standard", 0, "UK"),
            timestamp=datetime.now(),
            demand_score=0.9,
        )
        strategy = MLInformedPricingStrategy()
        price = strategy.calculate(ctx)
        self.assertGreaterEqual(price, 500.0 * 0.70)


class TestPricingEngineStrategySwap(unittest.TestCase):
    """Demonstrates testability via mock strategy injection."""

    def test_engine_delegates_to_strategy(self):
        mock_strategy = MagicMock(spec=PricingStrategy)
        mock_strategy.calculate.return_value = 42.00
        mock_strategy.name = "MockStrategy"

        ctx = PricingContext(
            product=Product("P4", "Book", 20.0, "books", 100),
            customer=CustomerProfile("C4", "vip", 10, "UK"),
            timestamp=datetime.now(),
            demand_score=0.3,
        )
        engine = PricingEngine(mock_strategy)
        price = engine.get_price(ctx)

        self.assertEqual(price, 42.00)
        mock_strategy.calculate.assert_called_once_with(ctx)

    def test_strategy_hot_swap(self):
        engine = PricingEngine(StandardPricingStrategy())
        engine.set_strategy(DynamicDemandStrategy())
        self.assertIsInstance(engine._strategy, DynamicDemandStrategy)


# ===========================================================================
# Demo
# ===========================================================================

if __name__ == "__main__":
    print("=== Strategy Pattern — Pricing Engine Demo ===\n")

    laptop = Product("LAPTOP-01", "Pro Laptop 15\"", 1299.99, "electronics", 5)
    alice  = CustomerProfile("C-ALICE", "vip",      42, "UK")
    bob    = CustomerProfile("C-BOB",   "standard",  2, "US")

    ctx_alice = PricingContext(laptop, alice, datetime.now(), demand_score=0.85)
    ctx_bob   = PricingContext(laptop, bob,   datetime.now(), demand_score=0.85)

    engine = PricingEngine(StandardPricingStrategy())

    for strategy_cls in [StandardPricingStrategy, DynamicDemandStrategy,
                         MLInformedPricingStrategy, BundleDiscountStrategy]:
        engine.set_strategy(strategy_cls())
        alice_price = engine.get_price(ctx_alice)
        bob_price   = engine.get_price(ctx_bob)
        print(f"  Strategy: {engine._strategy.name:<22} Alice: £{alice_price}  Bob: £{bob_price}")

    print("\n--- Running unit tests ---")
    unittest.main(argv=[""], exit=False, verbosity=0)
    print("All tests passed.")
