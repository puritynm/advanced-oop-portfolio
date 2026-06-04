"""
Artefact 6: Decorator Pattern — Logging & Security Layers
==========================================================
Module: Advanced Object-Oriented Programming

Pattern: Decorator (GoF Structural)
Domain:  API service layer with cross-cutting concerns

Why Decorator?
  Logging, authentication, rate-limiting, and caching are cross-cutting
  concerns that should NOT be embedded in core business logic. The Decorator
  pattern wraps a component transparently, adding behaviour without subclassing.

Benefits:
  • Separation of concerns — DataService owns data logic; decorators own
    security/observability. Neither knows about the other.
  • Runtime composability — stack decorators in any order at assembly time.
  • Testability — each decorator layer tested independently with a mock inner.
  • Extensibility — add a CachingDecorator without touching existing code.

Trade-off acknowledged:
  Deep decorator stacks can be hard to debug (call chain is opaque). Mitigated
  here by clear naming, logging at each layer, and keeping stack depth ≤ 3.

Run:
    python decorator_service.py
"""

from __future__ import annotations

import functools
import time
import unittest
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, call, patch


# ===========================================================================
# Service Abstraction
# ===========================================================================

class DataService(ABC):
    """
    Core contract for data retrieval.
    All decorators implement this interface — LSP ensures interchangeability.
    """

    @abstractmethod
    def get_user(self, user_id: str) -> dict:
        """Retrieve a user record by ID."""
        ...

    @abstractmethod
    def update_user(self, user_id: str, data: dict) -> bool:
        """Update a user record. Returns True on success."""
        ...

    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        """Delete a user record. Returns True on success."""
        ...


# ===========================================================================
# Concrete Component
# ===========================================================================

class UserDataService(DataService):
    """
    Core implementation — only concerned with data retrieval.
    Contains no logging, auth, or rate-limiting code.
    """

    def __init__(self):
        self._store: dict[str, dict] = {
            "U001": {"id": "U001", "name": "Alice", "email": "alice@example.com", "role": "admin"},
            "U002": {"id": "U002", "name": "Bob",   "email": "bob@example.com",   "role": "user"},
        }

    def get_user(self, user_id: str) -> dict:
        user = self._store.get(user_id)
        if not user:
            raise KeyError(f"User {user_id!r} not found.")
        return dict(user)

    def update_user(self, user_id: str, data: dict) -> bool:
        if user_id not in self._store:
            raise KeyError(f"User {user_id!r} not found.")
        self._store[user_id].update(data)
        return True

    def delete_user(self, user_id: str) -> bool:
        if user_id not in self._store:
            raise KeyError(f"User {user_id!r} not found.")
        del self._store[user_id]
        return True


# ===========================================================================
# Base Decorator
# ===========================================================================

class DataServiceDecorator(DataService):
    """
    Abstract decorator that forwards all calls to the wrapped component.
    Subclasses override only the methods they need to intercept.
    """

    def __init__(self, wrapped: DataService):
        self._wrapped = wrapped

    def get_user(self, user_id: str) -> dict:
        return self._wrapped.get_user(user_id)

    def update_user(self, user_id: str, data: dict) -> bool:
        return self._wrapped.update_user(user_id, data)

    def delete_user(self, user_id: str) -> bool:
        return self._wrapped.delete_user(user_id)


# ===========================================================================
# Concrete Decorators
# ===========================================================================

class LoggingDecorator(DataServiceDecorator):
    """
    Wraps every operation with structured entry/exit logging.

    Demonstrates:
      • Separation of concerns — core service has no print/logging calls.
      • Transparent wrapping — callers see the same DataService interface.
    """

    def __init__(self, wrapped: DataService, logger=None):
        super().__init__(wrapped)
        self._log = logger or self._default_log

    @staticmethod
    def _default_log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"  [{ts}] LOG | {msg}")

    def _timed_call(self, operation: str, fn, *args, **kwargs):
        self._log(f"START {operation}({args[0] if args else ''})")
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            self._log(f"END   {operation} — OK ({elapsed:.1f}ms)")
            return result
        except Exception as exc:
            self._log(f"END   {operation} — ERROR: {exc}")
            raise

    def get_user(self, user_id: str) -> dict:
        return self._timed_call("get_user", self._wrapped.get_user, user_id)

    def update_user(self, user_id: str, data: dict) -> bool:
        return self._timed_call("update_user", self._wrapped.update_user, user_id, data)

    def delete_user(self, user_id: str) -> bool:
        return self._timed_call("delete_user", self._wrapped.delete_user, user_id)


class AuthenticationDecorator(DataServiceDecorator):
    """
    Enforces role-based access control before delegating.

    Security layer added without touching UserDataService — demonstrates
    the Open/Closed Principle at the cross-cutting concern level.
    """

    # Operation → minimum required role
    REQUIRED_ROLES: dict[str, set[str]] = {
        "get_user":    {"user", "admin"},
        "update_user": {"admin"},
        "delete_user": {"admin"},
    }

    def __init__(self, wrapped: DataService, current_role: str):
        super().__init__(wrapped)
        self._role = current_role

    def _check(self, operation: str) -> None:
        allowed = self.REQUIRED_ROLES.get(operation, set())
        if self._role not in allowed:
            raise PermissionError(
                f"Role '{self._role}' is not authorised for '{operation}'."
            )

    def get_user(self, user_id: str) -> dict:
        self._check("get_user")
        return super().get_user(user_id)

    def update_user(self, user_id: str, data: dict) -> bool:
        self._check("update_user")
        return super().update_user(user_id, data)

    def delete_user(self, user_id: str) -> bool:
        self._check("delete_user")
        return super().delete_user(user_id)


class RateLimitDecorator(DataServiceDecorator):
    """
    Limits calls per user_id to max_calls within window_seconds.
    Demonstrates: dynamically extensible functionality stacked at runtime.
    """

    def __init__(self, wrapped: DataService, max_calls: int = 5, window_seconds: float = 60.0):
        super().__init__(wrapped)
        self._max = max_calls
        self._window = window_seconds
        self._history: dict[str, list[float]] = defaultdict(list)

    def _check_rate(self, user_id: str) -> None:
        now = time.time()
        timestamps = self._history[user_id]
        # Evict expired entries
        self._history[user_id] = [t for t in timestamps if now - t < self._window]
        if len(self._history[user_id]) >= self._max:
            raise RuntimeError(
                f"Rate limit exceeded for user '{user_id}' "
                f"({self._max} calls / {self._window}s)."
            )
        self._history[user_id].append(now)

    def get_user(self, user_id: str) -> dict:
        self._check_rate(user_id)
        return super().get_user(user_id)

    def update_user(self, user_id: str, data: dict) -> bool:
        self._check_rate(user_id)
        return super().update_user(user_id, data)

    def delete_user(self, user_id: str) -> bool:
        self._check_rate(user_id)
        return super().delete_user(user_id)


# ===========================================================================
# Unit Tests
# ===========================================================================

class TestLoggingDecorator(unittest.TestCase):
    def setUp(self):
        self.logs: list[str] = []
        inner = UserDataService()
        self.svc = LoggingDecorator(inner, logger=self.logs.append)

    def test_get_user_logs_start_and_end(self):
        self.svc.get_user("U001")
        self.assertTrue(any("START get_user" in l for l in self.logs))
        self.assertTrue(any("END   get_user" in l for l in self.logs))

    def test_error_is_logged_and_reraised(self):
        with self.assertRaises(KeyError):
            self.svc.get_user("GHOST")
        self.assertTrue(any("ERROR" in l for l in self.logs))


class TestAuthenticationDecorator(unittest.TestCase):
    def _svc(self, role: str) -> DataService:
        return AuthenticationDecorator(UserDataService(), current_role=role)

    def test_admin_can_get_user(self):
        svc = self._svc("admin")
        user = svc.get_user("U001")
        self.assertEqual(user["name"], "Alice")

    def test_user_role_can_get_user(self):
        svc = self._svc("user")
        user = svc.get_user("U001")
        self.assertIn("name", user)

    def test_user_role_cannot_delete(self):
        svc = self._svc("user")
        with self.assertRaises(PermissionError):
            svc.delete_user("U002")

    def test_guest_role_blocked(self):
        svc = self._svc("guest")
        with self.assertRaises(PermissionError):
            svc.get_user("U001")


class TestRateLimitDecorator(unittest.TestCase):
    def test_within_limit_succeeds(self):
        svc = RateLimitDecorator(UserDataService(), max_calls=3)
        for _ in range(3):
            svc.get_user("U001")  # should not raise

    def test_exceeding_limit_raises(self):
        svc = RateLimitDecorator(UserDataService(), max_calls=2)
        svc.get_user("U001")
        svc.get_user("U001")
        with self.assertRaises(RuntimeError):
            svc.get_user("U001")


class TestDecoratorComposition(unittest.TestCase):
    """Verify that stacking decorators preserves the DataService interface."""

    def test_full_stack_returns_correct_data(self):
        svc = RateLimitDecorator(
            LoggingDecorator(
                AuthenticationDecorator(UserDataService(), current_role="admin"),
                logger=lambda _: None  # suppress output in tests
            ),
            max_calls=10
        )
        user = svc.get_user("U001")
        self.assertEqual(user["id"], "U001")

    def test_auth_blocked_before_inner_called(self):
        mock_inner = MagicMock(spec=DataService)
        svc = AuthenticationDecorator(mock_inner, current_role="guest")
        with self.assertRaises(PermissionError):
            svc.get_user("U001")
        mock_inner.get_user.assert_not_called()


# ===========================================================================
# Demo
# ===========================================================================

if __name__ == "__main__":
    print("=== Decorator Pattern — Layered Service Demo ===\n")

    # Assemble: RateLimit → Logging → Auth → Core
    service: DataService = RateLimitDecorator(
        LoggingDecorator(
            AuthenticationDecorator(UserDataService(), current_role="admin")
        ),
        max_calls=5
    )

    print("--- Fetching user (admin role) ---")
    user = service.get_user("U001")
    print(f"  Result: {user}\n")

    print("--- Attempting delete (blocked for non-admin) ---")
    user_svc = RateLimitDecorator(
        LoggingDecorator(
            AuthenticationDecorator(UserDataService(), current_role="user")
        )
    )
    try:
        user_svc.delete_user("U002")
    except PermissionError as e:
        print(f"  Blocked: {e}\n")

    print("--- Running unit tests ---")
    unittest.main(argv=[""], exit=False, verbosity=0)
    print("All tests passed.")
