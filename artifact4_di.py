"""
Artifact 4: Dependency Injection & Inversion of Control
=========================================================
Module: Advanced Object-Oriented Programming

Demonstrates:
  1. The problem with tightly coupled code (original)
  2. Manual Dependency Injection via constructor
  3. IoC Container using the `dependency-injector` pattern (pure stdlib)
  4. Unit tests that exploit DI for fast, reliable testing (no real email sent)
  5. Additional services: SMS, Push Notification

Run tests:
    python -m pytest artifact4_di.py -v
Run demo:
    python artifact4_di.py
"""

from __future__ import annotations

import unittest
from abc import ABC, abstractmethod
from typing import List
from unittest.mock import MagicMock, call, patch


# ===========================================================================
# BEFORE: Tightly Coupled (original problem)
# ===========================================================================

class _EmailServiceBefore:
    """Original: sends a real email (side effect, hard to test)."""
    def send_email(self, user: str, message: str) -> None:
        print(f"[REAL EMAIL] To: {user} — {message}")


class _UserManagerBefore:
    """
    Original: instantiates EmailService internally.

    Problems:
      - Tight coupling: cannot swap EmailService for SMS without editing this class.
      - Low testability: tests send real emails (slow, unreliable, brittle).
      - OCP violation: adding SMS requires modifying UserManager.
    """
    def __init__(self):
        self.notifier = _EmailServiceBefore()   # ← Direct, hidden dependency

    def register_user(self, user: str) -> None:
        print(f"[UserManager] Registering '{user}'...")
        self.notifier.send_email(user, "Welcome!")


# ===========================================================================
# AFTER: Dependency Injection — SOLID-compliant
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Abstract notification interface (ISP / DIP)
# ---------------------------------------------------------------------------

class NotificationService(ABC):
    """
    Dependency Inversion Principle: high-level modules depend on this
    abstraction, not on concrete implementations.

    Interface Segregation Principle: one focused method — send_notification.
    """

    @abstractmethod
    def send_notification(self, user: str, message: str) -> None:
        """Send a notification to a user."""
        ...


# ---------------------------------------------------------------------------
# 2. Concrete implementations (LSP: all safely substitutable)
# ---------------------------------------------------------------------------

class EmailService(NotificationService):
    """Sends notifications via email."""

    def __init__(self, smtp_host: str = "smtp.example.com"):
        self._host = smtp_host

    def send_notification(self, user: str, message: str) -> None:
        # In production: call smtplib here
        print(f"[EmailService] SMTP({self._host}) → {user}: {message}")


class SMSService(NotificationService):
    """
    Sends notifications via SMS.
    Added WITHOUT modifying UserManager — Open/Closed Principle.
    """

    def __init__(self, api_key: str = "demo-key"):
        self._api_key = api_key

    def send_notification(self, user: str, message: str) -> None:
        print(f"[SMSService] SMS({self._api_key[:4]}…) → {user}: {message}")


class PushNotificationService(NotificationService):
    """Sends push notifications via a mobile gateway."""

    def __init__(self, app_id: str = "edu-app"):
        self._app_id = app_id

    def send_notification(self, user: str, message: str) -> None:
        print(f"[PushService] Push({self._app_id}) → {user}: {message}")


class MultiChannelService(NotificationService):
    """
    Composite pattern: fans out to multiple services simultaneously.
    Users receive notifications on every registered channel.
    """

    def __init__(self, services: List[NotificationService]):
        self._services = services

    def send_notification(self, user: str, message: str) -> None:
        for svc in self._services:
            svc.send_notification(user, message)


# ---------------------------------------------------------------------------
# 3. UserManager with Constructor Injection
# ---------------------------------------------------------------------------

class UserManager:
    """
    Manages user registration.

    Dependency Inversion Principle:
      Depends on NotificationService (abstraction), not EmailService (concrete).
    Constructor Injection:
      The notifier is supplied externally — easy to swap in tests or production.
    Single Responsibility Principle:
      Only manages user registration; delegates notifications.
    """

    def __init__(self, notifier: NotificationService):
        self._notifier = notifier   # ← Injected, not instantiated here

    def register_user(self, user: str) -> None:
        """Register a user and send a welcome notification."""
        if not user or not user.strip():
            raise ValueError("Username cannot be empty.")
        print(f"[UserManager] Registering '{user}'...")
        self._notifier.send_notification(user, "Welcome to the platform!")

    def reset_password(self, user: str) -> None:
        """Trigger a password-reset notification."""
        self._notifier.send_notification(user, "Your password has been reset.")


# ---------------------------------------------------------------------------
# 4. IoC Container (pure stdlib — no third-party library required)
# ---------------------------------------------------------------------------

class DIContainer:
    """
    Lightweight Inversion of Control container.

    Stores factory callables and resolves dependencies on demand.
    This replicates the core behaviour of libraries like `dependency-injector`
    without requiring pip install.

    Usage:
        container = DIContainer()
        container.register("notifier", lambda: EmailService())
        container.register("user_manager", lambda: UserManager(container.resolve("notifier")))
        mgr = container.resolve("user_manager")
    """

    def __init__(self):
        self._registry: dict = {}
        self._singletons: dict = {}

    def register(self, name: str, factory, singleton: bool = True) -> None:
        """
        Register a service factory.

        Args:
            name:      Logical name for the dependency.
            factory:   Zero-argument callable returning the service instance.
            singleton: If True, the same instance is reused on each resolve.
        """
        self._registry[name] = (factory, singleton)

    def resolve(self, name: str):
        """Resolve (create or return) a registered service."""
        if name not in self._registry:
            raise KeyError(f"No service registered under '{name}'.")
        factory, singleton = self._registry[name]
        if singleton:
            if name not in self._singletons:
                self._singletons[name] = factory()
            return self._singletons[name]
        return factory()


# ===========================================================================
# 5. Unit Tests — demonstrating DI testability benefits
# ===========================================================================

class MockNotificationService(NotificationService):
    """
    In-memory mock — records calls so tests can assert on them.
    Does NOT send real emails/SMS: fast, deterministic, isolated.
    """

    def __init__(self):
        self.calls: List[tuple] = []

    def send_notification(self, user: str, message: str) -> None:
        self.calls.append((user, message))


class TestUserManagerWithDI(unittest.TestCase):
    """
    Tests for UserManager using injected mock notifier.
    No real email/SMS is sent — tests are fast and deterministic.
    """

    def setUp(self):
        self.mock_notifier = MockNotificationService()
        self.manager = UserManager(notifier=self.mock_notifier)

    def test_register_sends_welcome_notification(self):
        self.manager.register_user("alice")
        self.assertEqual(len(self.mock_notifier.calls), 1)
        user, message = self.mock_notifier.calls[0]
        self.assertEqual(user, "alice")
        self.assertIn("Welcome", message)

    def test_reset_password_sends_notification(self):
        self.manager.reset_password("bob")
        _, message = self.mock_notifier.calls[0]
        self.assertIn("password", message.lower())

    def test_empty_username_raises(self):
        with self.assertRaises(ValueError):
            self.manager.register_user("")

    def test_whitespace_username_raises(self):
        with self.assertRaises(ValueError):
            self.manager.register_user("   ")

    def test_multiple_registrations_all_notified(self):
        for name in ("carol", "dave", "eve"):
            self.manager.register_user(name)
        self.assertEqual(len(self.mock_notifier.calls), 3)
        notified_users = [c[0] for c in self.mock_notifier.calls]
        self.assertIn("carol", notified_users)
        self.assertIn("dave", notified_users)
        self.assertIn("eve", notified_users)

    def test_swap_to_sms_without_changing_user_manager(self):
        """
        Demonstrates that swapping the notifier requires zero changes to UserManager.
        This is the core DI / OCP benefit.
        """
        sms_mock = MockNotificationService()
        manager_with_sms = UserManager(notifier=sms_mock)
        manager_with_sms.register_user("frank")
        self.assertEqual(len(sms_mock.calls), 1)


class TestDIContainer(unittest.TestCase):
    """Tests for the IoC container."""

    def test_resolve_returns_correct_type(self):
        container = DIContainer()
        container.register("notifier", lambda: EmailService())
        container.register("manager", lambda: UserManager(container.resolve("notifier")))
        manager = container.resolve("manager")
        self.assertIsInstance(manager, UserManager)

    def test_singleton_returns_same_instance(self):
        container = DIContainer()
        container.register("svc", lambda: EmailService(), singleton=True)
        a = container.resolve("svc")
        b = container.resolve("svc")
        self.assertIs(a, b)

    def test_non_singleton_returns_new_instance(self):
        container = DIContainer()
        container.register("svc", lambda: EmailService(), singleton=False)
        a = container.resolve("svc")
        b = container.resolve("svc")
        self.assertIsNot(a, b)

    def test_unregistered_service_raises(self):
        container = DIContainer()
        with self.assertRaises(KeyError):
            container.resolve("ghost")


# ===========================================================================
# Demo
# ===========================================================================

def build_container(notifier_type: str = "email") -> DIContainer:
    """Compose the application via an IoC container."""
    container = DIContainer()

    if notifier_type == "email":
        container.register("notifier", lambda: EmailService("smtp.shopease.com"))
    elif notifier_type == "sms":
        container.register("notifier", lambda: SMSService("live-api-key-xyz"))
    elif notifier_type == "multi":
        container.register(
            "notifier",
            lambda: MultiChannelService([
                EmailService(),
                SMSService(),
                PushNotificationService(),
            ])
        )

    container.register(
        "user_manager",
        lambda: UserManager(container.resolve("notifier"))
    )
    return container


if __name__ == "__main__":
    print("=== Running DI Unit Tests ===\n")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestUserManagerWithDI)
    suite.addTests(loader.loadTestsFromTestCase(TestDIContainer))
    unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n=== Dependency Injection Demo ===\n")

    print("--- Email only ---")
    c1 = build_container("email")
    c1.resolve("user_manager").register_user("alice")

    print("\n--- SMS only ---")
    c2 = build_container("sms")
    c2.resolve("user_manager").register_user("bob")

    print("\n--- Multi-channel (email + SMS + push) ---")
    c3 = build_container("multi")
    c3.resolve("user_manager").register_user("carol")
    c3.resolve("user_manager").reset_password("carol")
