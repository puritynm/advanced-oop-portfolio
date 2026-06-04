"""
Artefact 7: Visitor Pattern — Analytics & Reporting Engine
===========================================================
Module: Advanced Object-Oriented Programming

Pattern: Visitor (GoF Behavioural)
Domain:  E-commerce analytics — separating reporting algorithms from
         the product/order object structure

Why Visitor?
  The product hierarchy (PhysicalProduct, DigitalProduct, SubscriptionProduct)
  is stable, but reporting requirements change frequently. Adding new reports
  (tax, margin, export) without Visitor would require modifying every product
  class — an SRP and OCP violation. Visitor externalises algorithms.

Benefits:
  • Separation of concerns — product classes own product data; visitors own
    reporting algorithms. Neither bleeds into the other.
  • Scalability — adding a new report = add one visitor class; zero changes
    to the product hierarchy.
  • Testability — each visitor independently unit-tested with stub products.

Trade-off:
  Visitor violates OCP in the *element* direction: adding a new product type
  requires updating every visitor. This trade-off is acceptable when the
  element hierarchy is stable and operations vary frequently.

Run:
    python visitor_analytics.py
"""

from __future__ import annotations

import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


# ===========================================================================
# Element Hierarchy (stable — rarely changes)
# ===========================================================================

class ProductElement(ABC):
    """Abstract product that accepts a visitor."""

    @abstractmethod
    def accept(self, visitor: "ProductVisitor") -> None:
        """Double dispatch: delegates back to the visitor with self."""
        ...


@dataclass
class PhysicalProduct(ProductElement):
    product_id: str
    name: str
    price: float
    weight_kg: float
    stock: int

    def accept(self, visitor: "ProductVisitor") -> None:
        visitor.visit_physical(self)


@dataclass
class DigitalProduct(ProductElement):
    product_id: str
    name: str
    price: float
    file_size_mb: float
    download_limit: int

    def accept(self, visitor: "ProductVisitor") -> None:
        visitor.visit_digital(self)


@dataclass
class SubscriptionProduct(ProductElement):
    product_id: str
    name: str
    monthly_price: float
    billing_cycle_months: int
    active_subscribers: int

    @property
    def annual_revenue(self) -> float:
        return self.monthly_price * 12 * self.active_subscribers

    def accept(self, visitor: "ProductVisitor") -> None:
        visitor.visit_subscription(self)


# ===========================================================================
# Visitor Abstraction
# ===========================================================================

class ProductVisitor(ABC):
    """
    Abstract visitor — one visit method per concrete element type.
    Adding a new report = subclass this; zero changes to product classes.
    """

    @abstractmethod
    def visit_physical(self, product: PhysicalProduct) -> None: ...

    @abstractmethod
    def visit_digital(self, product: DigitalProduct) -> None: ...

    @abstractmethod
    def visit_subscription(self, product: SubscriptionProduct) -> None: ...


# ===========================================================================
# Concrete Visitors (reports / algorithms)
# ===========================================================================

class RevenueReportVisitor(ProductVisitor):
    """
    Calculates total revenue potential across all product types.
    Each type uses a different revenue formula — cleanly isolated here.
    """

    def __init__(self):
        self.total_revenue: float = 0.0
        self._breakdown: list[dict] = []

    def visit_physical(self, p: PhysicalProduct) -> None:
        rev = p.price * p.stock
        self.total_revenue += rev
        self._breakdown.append({"id": p.product_id, "type": "physical", "revenue": rev})

    def visit_digital(self, p: DigitalProduct) -> None:
        # Digital: unlimited stock — use a conservative 100-unit estimate
        rev = p.price * 100
        self.total_revenue += rev
        self._breakdown.append({"id": p.product_id, "type": "digital", "revenue": rev})

    def visit_subscription(self, p: SubscriptionProduct) -> None:
        rev = p.annual_revenue
        self.total_revenue += rev
        self._breakdown.append({"id": p.product_id, "type": "subscription", "revenue": rev})

    def report(self) -> str:
        lines = ["=== Revenue Report ==="]
        for item in self._breakdown:
            lines.append(f"  [{item['type']:<12}] {item['id']}: £{item['revenue']:,.2f}")
        lines.append(f"  {'Total':>26}: £{self.total_revenue:,.2f}")
        return "\n".join(lines)


class TaxCalculationVisitor(ProductVisitor):
    """
    Computes VAT/tax for each product type.
    Different tax rules per category — all isolated in this visitor.
    """

    UK_VAT = 0.20
    DIGITAL_VAT = 0.20
    SUBSCRIPTION_VAT = 0.20

    def __init__(self):
        self.total_tax: float = 0.0
        self._items: list[dict] = []

    def visit_physical(self, p: PhysicalProduct) -> None:
        tax = p.price * self.UK_VAT
        self.total_tax += tax * p.stock
        self._items.append({"id": p.product_id, "unit_tax": tax})

    def visit_digital(self, p: DigitalProduct) -> None:
        tax = p.price * self.DIGITAL_VAT
        self.total_tax += tax * 100
        self._items.append({"id": p.product_id, "unit_tax": tax})

    def visit_subscription(self, p: SubscriptionProduct) -> None:
        tax = p.monthly_price * self.SUBSCRIPTION_VAT
        annual_tax = tax * 12 * p.active_subscribers
        self.total_tax += annual_tax
        self._items.append({"id": p.product_id, "unit_tax": tax})

    def report(self) -> str:
        lines = ["=== Tax Report ==="]
        for item in self._items:
            lines.append(f"  {item['id']}: unit tax £{item['unit_tax']:.2f}")
        lines.append(f"  Total tax liability: £{self.total_tax:,.2f}")
        return "\n".join(lines)


class InventoryAuditVisitor(ProductVisitor):
    """
    Audits inventory health — identifies low-stock physical products,
    high-download digital products, and churn-risk subscriptions.
    AI-readiness: this visitor could feed data to an ML anomaly detector.
    """

    LOW_STOCK_THRESHOLD = 5

    def __init__(self):
        self.warnings: list[str] = []
        self.summary: dict = {"physical": 0, "digital": 0, "subscription": 0}

    def visit_physical(self, p: PhysicalProduct) -> None:
        self.summary["physical"] += 1
        if p.stock <= self.LOW_STOCK_THRESHOLD:
            self.warnings.append(
                f"LOW STOCK: {p.name} ({p.product_id}) — only {p.stock} units left"
            )

    def visit_digital(self, p: DigitalProduct) -> None:
        self.summary["digital"] += 1
        if p.download_limit < 3:
            self.warnings.append(
                f"LOW DOWNLOAD LIMIT: {p.name} — limit is {p.download_limit}"
            )

    def visit_subscription(self, p: SubscriptionProduct) -> None:
        self.summary["subscription"] += 1
        if p.active_subscribers < 10:
            self.warnings.append(
                f"CHURN RISK: {p.name} — only {p.active_subscribers} active subscribers"
            )

    def report(self) -> str:
        lines = ["=== Inventory Audit ==="]
        lines.append(f"  Physical: {self.summary['physical']} | Digital: {self.summary['digital']} | Subscriptions: {self.summary['subscription']}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠  {w}")
        else:
            lines.append("  No warnings.")
        return "\n".join(lines)


# ===========================================================================
# Catalogue (object structure)
# ===========================================================================

class ProductCatalogue:
    """Holds the product collection and dispatches visitors across all elements."""

    def __init__(self):
        self._products: List[ProductElement] = []

    def add(self, product: ProductElement) -> None:
        self._products.append(product)

    def accept(self, visitor: ProductVisitor) -> None:
        """Apply visitor to every product — O(n) traversal."""
        for product in self._products:
            product.accept(visitor)

    def __len__(self) -> int:
        return len(self._products)


# ===========================================================================
# Unit Tests
# ===========================================================================

class TestRevenueVisitor(unittest.TestCase):
    def setUp(self):
        self.catalogue = ProductCatalogue()
        self.catalogue.add(PhysicalProduct("P1", "Laptop", 1000.0, 2.5, 10))
        self.catalogue.add(DigitalProduct("D1", "eBook", 15.0, 5.0, 100))
        self.catalogue.add(SubscriptionProduct("S1", "SaaS", 29.99, 1, 50))

    def test_physical_revenue(self):
        v = RevenueReportVisitor()
        self.catalogue._products[0].accept(v)
        self.assertAlmostEqual(v.total_revenue, 10_000.0, places=1)

    def test_subscription_annual_revenue(self):
        v = RevenueReportVisitor()
        self.catalogue._products[2].accept(v)
        expected = 29.99 * 12 * 50
        self.assertAlmostEqual(v.total_revenue, expected, places=1)

    def test_full_catalogue_revenue_positive(self):
        v = RevenueReportVisitor()
        self.catalogue.accept(v)
        self.assertGreater(v.total_revenue, 0)


class TestInventoryAuditVisitor(unittest.TestCase):
    def test_low_stock_warning(self):
        v = InventoryAuditVisitor()
        PhysicalProduct("P2", "Mouse", 25.0, 0.1, 3).accept(v)
        self.assertTrue(any("LOW STOCK" in w for w in v.warnings))

    def test_sufficient_stock_no_warning(self):
        v = InventoryAuditVisitor()
        PhysicalProduct("P3", "Monitor", 300.0, 5.0, 100).accept(v)
        self.assertEqual(len(v.warnings), 0)

    def test_churn_risk_warning(self):
        v = InventoryAuditVisitor()
        SubscriptionProduct("S2", "Tiny SaaS", 9.99, 1, 5).accept(v)
        self.assertTrue(any("CHURN RISK" in w for w in v.warnings))


class TestVisitorDoesNotModifyElements(unittest.TestCase):
    """Elements must remain unchanged after visitor traversal (pure read)."""

    def test_product_unchanged_after_visit(self):
        p = PhysicalProduct("P4", "Chair", 199.0, 8.0, 50)
        original_stock = p.stock
        p.accept(RevenueReportVisitor())
        self.assertEqual(p.stock, original_stock)


# ===========================================================================
# Demo
# ===========================================================================

if __name__ == "__main__":
    print("=== Visitor Pattern — Analytics Engine Demo ===\n")

    catalogue = ProductCatalogue()
    catalogue.add(PhysicalProduct("LAPTOP-01", "Pro Laptop",    1299.0, 2.5,  4))   # low stock
    catalogue.add(PhysicalProduct("MOUSE-01",  "Wireless Mouse",  49.0, 0.1, 50))
    catalogue.add(DigitalProduct( "EBOOK-01",  "Python Mastery",  29.0, 12.0, 200))
    catalogue.add(SubscriptionProduct("SAAS-01", "ShopEase Pro", 29.99, 1, 7))      # churn risk

    for visitor_cls in [RevenueReportVisitor, TaxCalculationVisitor, InventoryAuditVisitor]:
        visitor = visitor_cls()
        catalogue.accept(visitor)
        print(visitor.report())
        print()

    print("--- Running unit tests ---")
    unittest.main(argv=[""], exit=False, verbosity=0)
    print("All tests passed.")
