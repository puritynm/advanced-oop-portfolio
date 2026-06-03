"""
Artifact 1: SOLID Principles — Online Shopping System
=======================================================
Module: Advanced Object-Oriented Programming

Demonstrates refactoring a tightly-coupled Order class into a design
that fully adheres to all five SOLID principles.

Run:
    python artifact1_solid_shopping.py
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


# ===========================================================================
# BEFORE: Poorly designed, monolithic Order class
# (shown for reference — do NOT use in production)
# ===========================================================================

class _OrderBefore:
    """
    Original implementation — included only to illustrate the problems.

    Violations:
      SRP — manages items AND handles payment logic.
      OCP — adding a new payment method means editing this class.
      DIP — directly references concrete payment strings.
    """
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def calculate_total(self):
        return sum(item.price for item in self.items)

    def pay(self, payment_type):
        if payment_type == "credit":
            print("Processing credit card payment...")
        elif payment_type == "paypal":
            print("Processing PayPal payment...")
        # Adding a new method requires modifying this class → OCP violation


# ===========================================================================
# AFTER: SOLID-compliant refactored design
# ===========================================================================

# ---------------------------------------------------------------------------
# Domain model — S in SRP: a product only knows about itself
# ---------------------------------------------------------------------------

@dataclass
class Product:
    """Represents a purchasable product. Knows only its own data."""
    name: str
    price: float
    sku: str = ""

    def __post_init__(self):
        if self.price < 0:
            raise ValueError(f"Price cannot be negative: {self.price}")


# ---------------------------------------------------------------------------
# Discount strategy — O in OCP: new discounts without modifying existing code
# ---------------------------------------------------------------------------

class DiscountStrategy(ABC):
    """
    Abstract base for discount algorithms.
    Open/Closed Principle: extend by adding subclasses, not by editing this.
    Interface Segregation Principle: focused single-method interface.
    """

    @abstractmethod
    def apply(self, total: float) -> float:
        """Return the discounted total."""
        ...


class NoDiscount(DiscountStrategy):
    """Null object — applies no discount."""
    def apply(self, total: float) -> float:
        return total


class PercentageDiscount(DiscountStrategy):
    """
    Liskov Substitution Principle: can replace DiscountStrategy anywhere.
    """
    def __init__(self, percent: float):
        if not (0 <= percent <= 100):
            raise ValueError(f"Percent must be 0–100, got {percent}")
        self._rate = percent / 100

    def apply(self, total: float) -> float:
        return round(total * (1 - self._rate), 2)


class FixedDiscount(DiscountStrategy):
    """Deducts a flat amount, floored at zero."""
    def __init__(self, amount: float):
        self._amount = amount

    def apply(self, total: float) -> float:
        return max(0.0, round(total - self._amount, 2))


# ---------------------------------------------------------------------------
# Payment abstraction — D in DIP / O in OCP
# ---------------------------------------------------------------------------

class PaymentMethod(ABC):
    """
    Dependency Inversion Principle:
      High-level Order depends on this abstraction, not on concrete classes.
    Interface Segregation Principle:
      Only the method relevant to payment is defined here.
    """

    @abstractmethod
    def pay(self, amount: float) -> bool:
        """
        Process a payment.

        Args:
            amount: Total to charge.

        Returns:
            True if payment succeeded.
        """
        ...


class CreditCardPayment(PaymentMethod):
    """Concrete payment via credit card. LSP: fully substitutable."""
    def __init__(self, card_number: str):
        self._card_number = card_number

    def pay(self, amount: float) -> bool:
        masked = f"****-****-****-{self._card_number[-4:]}"
        print(f"  [CreditCard] Charging £{amount:.2f} to {masked}")
        return True  # Simulate success


class PayPalPayment(PaymentMethod):
    """Concrete payment via PayPal. LSP: fully substitutable."""
    def __init__(self, email: str):
        self._email = email

    def pay(self, amount: float) -> bool:
        print(f"  [PayPal] Charging £{amount:.2f} via {self._email}")
        return True


class CryptoPayment(PaymentMethod):
    """
    New payment method added WITHOUT modifying any existing class.
    Demonstrates OCP and LSP.
    """
    def __init__(self, wallet_address: str):
        self._wallet = wallet_address

    def pay(self, amount: float) -> bool:
        print(f"  [Crypto] Sending £{amount:.2f} to wallet {self._wallet[:8]}…")
        return True


# ---------------------------------------------------------------------------
# Order — Single Responsibility: only manages cart items and totals
# ---------------------------------------------------------------------------

class Order:
    """
    Manages a shopping cart and delegates payment to an injected strategy.

    Single Responsibility Principle:
      Only computes totals and holds items — no payment logic.
    Dependency Inversion Principle:
      Accepts DiscountStrategy and PaymentMethod abstractions via constructor.
    """

    def __init__(
        self,
        discount: DiscountStrategy | None = None,
        payment_method: PaymentMethod | None = None,
    ):
        self._items: List[Product] = []
        self._discount: DiscountStrategy = discount or NoDiscount()
        self._payment_method: PaymentMethod | None = payment_method

    def add_item(self, product: Product, quantity: int = 1) -> None:
        """Add a product to the cart."""
        if quantity <= 0:
            raise ValueError("Quantity must be at least 1.")
        for _ in range(quantity):
            self._items.append(product)

    def calculate_total(self) -> float:
        """Compute the raw total before discounts."""
        return round(sum(p.price for p in self._items), 2)

    def calculate_discounted_total(self) -> float:
        """Compute the total after applying the discount strategy."""
        return self._discount.apply(self.calculate_total())

    def checkout(self) -> bool:
        """
        Finalise the order by processing payment.

        Returns:
            True if payment succeeded.

        Raises:
            ValueError: If no payment method has been set.
        """
        if self._payment_method is None:
            raise ValueError("No payment method configured.")
        if not self._items:
            raise ValueError("Cannot check out an empty cart.")

        total = self.calculate_discounted_total()
        print(f"\n  Cart total: £{self.calculate_total():.2f}")
        print(f"  After discount: £{total:.2f}")
        success = self._payment_method.pay(total)
        status = "✓ Payment successful" if success else "✗ Payment failed"
        print(f"  {status}")
        return success

    def receipt(self) -> str:
        """Return a formatted receipt string."""
        lines = ["=" * 40, "RECEIPT", "=" * 40]
        for item in self._items:
            lines.append(f"  {item.name:<25} £{item.price:.2f}")
        lines.append("-" * 40)
        lines.append(f"  {'Subtotal':<25} £{self.calculate_total():.2f}")
        lines.append(f"  {'Total (after discount)':<25} £{self.calculate_discounted_total():.2f}")
        lines.append("=" * 40)
        return "\n".join(lines)


# ===========================================================================
# Demo
# ===========================================================================

if __name__ == "__main__":
    # --- Scenario 1: Credit card with 10 % loyalty discount ---
    print("=== Scenario 1: Credit Card + 10% Discount ===")
    order1 = Order(
        discount=PercentageDiscount(10),
        payment_method=CreditCardPayment("4111111111111234"),
    )
    order1.add_item(Product("Wireless Headphones", 79.99))
    order1.add_item(Product("Phone Case", 12.50))
    order1.add_item(Product("USB-C Cable", 8.99), quantity=2)
    print(order1.receipt())
    order1.checkout()

    # --- Scenario 2: PayPal with a fixed £5 voucher ---
    print("\n=== Scenario 2: PayPal + £5 Voucher ===")
    order2 = Order(
        discount=FixedDiscount(5.0),
        payment_method=PayPalPayment("alice@example.com"),
    )
    order2.add_item(Product("Python Cookbook", 35.00))
    order2.checkout()

    # --- Scenario 3: New Crypto payment — no existing code changed ---
    print("\n=== Scenario 3: Crypto Payment (new method, OCP demo) ===")
    order3 = Order(
        payment_method=CryptoPayment("0xAbCdEf1234567890"),
    )
    order3.add_item(Product("Smart Watch", 199.99))
    order3.checkout()
