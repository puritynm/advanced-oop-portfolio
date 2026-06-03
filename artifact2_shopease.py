"""
Artifact 2: ShopEase — Scalable OOP E-Commerce Architecture
=============================================================
Module: Advanced Object-Oriented Programming

Implements a layered, modular e-commerce system with:
  • Presentation Layer   – CLI facade (simulating web/mobile UI)
  • Business Logic Layer – Authentication, catalogue, order processing
  • Data Access Layer    – In-memory repositories (swap for DB in production)

Design patterns used:
  Observer  — event notifications (order placed, shipped, etc.)
  Strategy  — pluggable payment methods
  Repository — decouples business logic from storage
  Dependency Injection — all dependencies passed via constructors

Run:
    python artifact2_shopease.py
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional


# ===========================================================================
# Data Access Layer — Repositories & Entities
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------

def generate_id() -> str:
    """Generate a short unique identifier."""
    return str(uuid.uuid4())[:8].upper()


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """
    Securely hash a password using PBKDF2-HMAC-SHA256.

    Returns:
        (hex_hash, hex_salt) — both stored; salt needed to verify.
    """
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return key.hex(), salt.hex()


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    salt = bytes.fromhex(stored_salt)
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------

@dataclass
class User:
    user_id: str = field(default_factory=generate_id)
    username: str = ""
    email: str = ""
    _password_hash: str = field(default="", repr=False)
    _salt: str = field(default="", repr=False)
    is_active: bool = True

    def set_password(self, plain: str) -> None:
        self._password_hash, self._salt = hash_password(plain)

    def check_password(self, plain: str) -> bool:
        return verify_password(plain, self._password_hash, self._salt)


@dataclass
class Product:
    product_id: str = field(default_factory=generate_id)
    name: str = ""
    description: str = ""
    price: float = 0.0
    stock: int = 0
    category: str = ""

    def is_available(self, quantity: int = 1) -> bool:
        return self.stock >= quantity


class OrderStatus(Enum):
    PENDING   = auto()
    PAID      = auto()
    SHIPPED   = auto()
    DELIVERED = auto()
    CANCELLED = auto()


@dataclass
class OrderItem:
    product: Product
    quantity: int
    unit_price: float  # snapshot at time of order

    @property
    def subtotal(self) -> float:
        return round(self.unit_price * self.quantity, 2)


@dataclass
class Order:
    order_id: str = field(default_factory=generate_id)
    user_id: str = ""
    items: List[OrderItem] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    created_at: float = field(default_factory=time.time)

    @property
    def total(self) -> float:
        return round(sum(i.subtotal for i in self.items), 2)


# ---------------------------------------------------------------------------
# Repositories (Data Access Layer)
# ---------------------------------------------------------------------------

class UserRepository:
    """In-memory store for User entities. Replace with DB adapter in prod."""

    def __init__(self):
        self._store: Dict[str, User] = {}

    def save(self, user: User) -> None:
        self._store[user.user_id] = user

    def find_by_id(self, user_id: str) -> Optional[User]:
        return self._store.get(user_id)

    def find_by_username(self, username: str) -> Optional[User]:
        return next(
            (u for u in self._store.values() if u.username == username), None
        )

    def all(self) -> List[User]:
        return list(self._store.values())


class ProductRepository:
    """In-memory store for Product entities."""

    def __init__(self):
        self._store: Dict[str, Product] = {}

    def save(self, product: Product) -> None:
        self._store[product.product_id] = product

    def find_by_id(self, product_id: str) -> Optional[Product]:
        return self._store.get(product_id)

    def search(self, query: str = "", category: str = "") -> List[Product]:
        results = list(self._store.values())
        if query:
            q = query.lower()
            results = [p for p in results if q in p.name.lower() or q in p.description.lower()]
        if category:
            results = [p for p in results if p.category == category]
        return results

    def all(self) -> List[Product]:
        return list(self._store.values())


class OrderRepository:
    """In-memory store for Order entities."""

    def __init__(self):
        self._store: Dict[str, Order] = {}

    def save(self, order: Order) -> None:
        self._store[order.order_id] = order

    def find_by_id(self, order_id: str) -> Optional[Order]:
        return self._store.get(order_id)

    def find_by_user(self, user_id: str) -> List[Order]:
        return [o for o in self._store.values() if o.user_id == user_id]


# ===========================================================================
# Business Logic Layer — Services & Patterns
# ===========================================================================

# ---------------------------------------------------------------------------
# Observer Pattern — event notifications
# ---------------------------------------------------------------------------

EventHandler = Callable[[str, dict], None]


class EventBus:
    """
    Simple publish/subscribe bus.
    Observer Pattern: services subscribe to events they care about.
    Extensibility: add listeners without modifying publishers.
    """

    def __init__(self):
        self._listeners: Dict[str, List[EventHandler]] = {}

    def subscribe(self, event: str, handler: EventHandler) -> None:
        self._listeners.setdefault(event, []).append(handler)

    def publish(self, event: str, payload: dict) -> None:
        for handler in self._listeners.get(event, []):
            handler(event, payload)


# ---------------------------------------------------------------------------
# Authentication Service
# ---------------------------------------------------------------------------

class AuthService:
    """
    Handles user registration and login.
    Security: passwords are hashed before storage; never stored in plain text.
    """

    def __init__(self, user_repo: UserRepository, event_bus: EventBus):
        self._repo = user_repo
        self._bus = event_bus
        self._sessions: Dict[str, str] = {}  # token → user_id

    def register(self, username: str, email: str, password: str) -> User:
        if self._repo.find_by_username(username):
            raise ValueError(f"Username '{username}' already taken.")
        user = User(username=username, email=email)
        user.set_password(password)
        self._repo.save(user)
        self._bus.publish("user.registered", {"user_id": user.user_id, "username": username})
        return user

    def login(self, username: str, password: str) -> str:
        """Returns a session token on success."""
        user = self._repo.find_by_username(username)
        if not user or not user.check_password(password):
            raise PermissionError("Invalid credentials.")
        token = generate_id()
        self._sessions[token] = user.user_id
        self._bus.publish("user.logged_in", {"user_id": user.user_id})
        return token

    def get_user_from_token(self, token: str) -> Optional[User]:
        uid = self._sessions.get(token)
        return self._repo.find_by_id(uid) if uid else None

    def logout(self, token: str) -> None:
        self._sessions.pop(token, None)


# ---------------------------------------------------------------------------
# Product Catalogue Service
# ---------------------------------------------------------------------------

class CatalogueService:
    """
    Manages product listings and stock.
    Scalability: delegates storage to a repository; easily sharded.
    """

    def __init__(self, product_repo: ProductRepository, event_bus: EventBus):
        self._repo = product_repo
        self._bus = event_bus

    def add_product(self, name: str, description: str, price: float,
                    stock: int, category: str) -> Product:
        if price < 0:
            raise ValueError("Price cannot be negative.")
        p = Product(name=name, description=description, price=price,
                    stock=stock, category=category)
        self._repo.save(p)
        self._bus.publish("product.added", {"product_id": p.product_id, "name": name})
        return p

    def search(self, query: str = "", category: str = "") -> List[Product]:
        return self._repo.search(query, category)

    def update_stock(self, product_id: str, delta: int) -> Product:
        p = self._repo.find_by_id(product_id)
        if not p:
            raise KeyError(f"Product {product_id} not found.")
        p.stock += delta
        self._repo.save(p)
        if p.stock == 0:
            self._bus.publish("product.out_of_stock", {"product_id": product_id})
        return p


# ---------------------------------------------------------------------------
# Payment (Strategy Pattern)
# ---------------------------------------------------------------------------

class PaymentGateway(ABC):
    """Abstraction for payment processing. DIP in action."""

    @abstractmethod
    def charge(self, amount: float, reference: str) -> bool: ...


class MockPaymentGateway(PaymentGateway):
    """Simulates a payment provider — replace with real SDK in production."""

    def charge(self, amount: float, reference: str) -> bool:
        print(f"  [MockGateway] Charged £{amount:.2f} for order {reference}")
        return True


# ---------------------------------------------------------------------------
# Order Processing Service
# ---------------------------------------------------------------------------

class OrderService:
    """
    Orchestrates order creation, payment, and fulfilment.
    Encapsulates all order-lifecycle business rules.
    """

    def __init__(
        self,
        order_repo: OrderRepository,
        product_repo: ProductRepository,
        payment_gateway: PaymentGateway,
        event_bus: EventBus,
    ):
        self._order_repo    = order_repo
        self._product_repo  = product_repo
        self._payment       = payment_gateway
        self._bus           = event_bus

    def create_order(self, user_id: str, cart: Dict[str, int]) -> Order:
        """
        Create and pay for an order.

        Args:
            user_id: ID of the purchasing user.
            cart: Mapping of product_id → quantity.

        Returns:
            The created Order, marked PAID on success.
        """
        items: List[OrderItem] = []
        for product_id, qty in cart.items():
            p = self._product_repo.find_by_id(product_id)
            if not p:
                raise KeyError(f"Unknown product: {product_id}")
            if not p.is_available(qty):
                raise ValueError(f"Insufficient stock for '{p.name}'")
            items.append(OrderItem(product=p, quantity=qty, unit_price=p.price))

        order = Order(user_id=user_id, items=items)
        self._order_repo.save(order)

        if self._payment.charge(order.total, order.order_id):
            order.status = OrderStatus.PAID
            # Deduct stock atomically
            for item in order.items:
                item.product.stock -= item.quantity
                self._product_repo.save(item.product)
            self._bus.publish("order.placed", {
                "order_id": order.order_id,
                "user_id": user_id,
                "total": order.total,
            })
        else:
            order.status = OrderStatus.CANCELLED
            self._bus.publish("order.failed", {"order_id": order.order_id})

        self._order_repo.save(order)
        return order

    def ship_order(self, order_id: str) -> Order:
        order = self._order_repo.find_by_id(order_id)
        if not order:
            raise KeyError(f"Order {order_id} not found.")
        if order.status != OrderStatus.PAID:
            raise ValueError(f"Cannot ship an order with status {order.status.name}.")
        order.status = OrderStatus.SHIPPED
        self._order_repo.save(order)
        self._bus.publish("order.shipped", {"order_id": order_id})
        return order


# ===========================================================================
# Presentation Layer — CLI facade
# ===========================================================================

class ShopEaseCLI:
    """
    Simulates the presentation layer (what a web controller would do).
    Delegates all logic to business-layer services.
    """

    def __init__(
        self,
        auth: AuthService,
        catalogue: CatalogueService,
        orders: OrderService,
    ):
        self._auth      = auth
        self._catalogue = catalogue
        self._orders    = orders
        self._token: Optional[str] = None

    # --- Auth ---
    def register(self, username: str, email: str, password: str) -> None:
        user = self._auth.register(username, email, password)
        print(f"[ShopEase] Registered user '{user.username}' (ID: {user.user_id})")

    def login(self, username: str, password: str) -> None:
        self._token = self._auth.login(username, password)
        print(f"[ShopEase] '{username}' logged in.")

    # --- Browse ---
    def list_products(self, query: str = "") -> None:
        products = self._catalogue.search(query=query)
        print(f"\n[ShopEase] Products{' matching ' + repr(query) if query else ''}:")
        for p in products:
            status = "✓ In stock" if p.stock > 0 else "✗ Out of stock"
            print(f"  [{p.product_id}] {p.name:<30} £{p.price:.2f}  {status}")

    # --- Purchase ---
    def place_order(self, cart: Dict[str, int]) -> None:
        user = self._auth.get_user_from_token(self._token or "")
        if not user:
            print("[ShopEase] Please log in first.")
            return
        order = self._orders.create_order(user.user_id, cart)
        print(f"\n[ShopEase] Order {order.order_id} — status: {order.status.name}")
        for item in order.items:
            print(f"  {item.product.name:<30} x{item.quantity}  £{item.subtotal:.2f}")
        print(f"  {'Total':<40} £{order.total:.2f}")


# ===========================================================================
# Bootstrap & Demo
# ===========================================================================

def build_shopease() -> ShopEaseCLI:
    """Wire up all dependencies using Dependency Injection."""
    user_repo    = UserRepository()
    product_repo = ProductRepository()
    order_repo   = OrderRepository()
    bus          = EventBus()
    payment      = MockPaymentGateway()

    # Register observers
    bus.subscribe("user.registered",  lambda e, p: print(f"  📧 Welcome email → {p['username']}"))
    bus.subscribe("order.placed",     lambda e, p: print(f"  📦 Confirmation sent for order {p['order_id']}"))
    bus.subscribe("order.shipped",    lambda e, p: print(f"  🚚 Shipping notification for order {p['order_id']}"))
    bus.subscribe("product.out_of_stock", lambda e, p: print(f"  ⚠️  Product {p['product_id']} is now out of stock"))

    auth      = AuthService(user_repo, bus)
    catalogue = CatalogueService(product_repo, bus)
    orders    = OrderService(order_repo, product_repo, payment, bus)

    return ShopEaseCLI(auth, catalogue, orders)


if __name__ == "__main__":
    shop = build_shopease()

    # Seed the catalogue
    catalogue = shop._catalogue
    laptop  = catalogue.add_product("Pro Laptop 15\"",  "Intel i9, 32 GB RAM", 1299.99, stock=5, category="electronics")
    mouse   = catalogue.add_product("Ergonomic Mouse",  "Wireless, silent click", 49.99,  stock=20, category="electronics")
    book    = catalogue.add_product("Clean Code",       "By Robert C. Martin",   29.99,  stock=10, category="books")

    # User journey
    print("=== ShopEase Demo ===\n")
    shop.register("alice", "alice@shopease.com", "S3cur3P@ss!")
    shop.login("alice", "S3cur3P@ss!")
    shop.list_products()

    print("\n--- Alice places an order ---")
    shop.place_order({laptop.product_id: 1, mouse.product_id: 2})

    # Ship the order (admin action)
    order_id = shop._orders._order_repo.find_by_user(
        shop._auth.get_user_from_token(shop._token)._id
        if hasattr(shop._auth.get_user_from_token(shop._token or ""), "_id")
        else shop._auth.get_user_from_token(shop._token or "").user_id
    )[0].order_id
    print(f"\n--- Shipping order {order_id} ---")
    shop._orders.ship_order(order_id)
    print("Done!")
