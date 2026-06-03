"""
Unit Tests: Thread-Safe Banking System
========================================
Tests cover normal operations, edge cases, concurrent access,
and deadlock-prevention behaviour.

Run with:
    python -m pytest test_banking_system.py -v
"""

import threading
import time
import unittest

from banking_system import (
    BankAccount,
    InsufficientFundsError,
    InvalidAmountError,
    TransactionSimulator,
    transfer,
)


# ---------------------------------------------------------------------------
# BankAccount Unit Tests
# ---------------------------------------------------------------------------

class TestBankAccountCreation(unittest.TestCase):
    def test_default_balance_is_zero(self):
        acc = BankAccount("T-000")
        self.assertEqual(acc.get_balance(), 0.0)

    def test_initial_balance_set_correctly(self):
        acc = BankAccount("T-001", initial_balance=500.0)
        self.assertEqual(acc.get_balance(), 500.0)

    def test_negative_initial_balance_raises(self):
        with self.assertRaises(InvalidAmountError):
            BankAccount("T-002", initial_balance=-100.0)


class TestDeposit(unittest.TestCase):
    def setUp(self):
        self.acc = BankAccount("DEP-001", initial_balance=100.0)

    def test_deposit_increases_balance(self):
        self.acc.deposit(50.0)
        self.assertEqual(self.acc.get_balance(), 150.0)

    def test_deposit_returns_new_balance(self):
        result = self.acc.deposit(25.0)
        self.assertEqual(result, 125.0)

    def test_zero_deposit_raises(self):
        with self.assertRaises(InvalidAmountError):
            self.acc.deposit(0)

    def test_negative_deposit_raises(self):
        with self.assertRaises(InvalidAmountError):
            self.acc.deposit(-10)

    def test_multiple_deposits_accumulate(self):
        for _ in range(10):
            self.acc.deposit(10.0)
        self.assertEqual(self.acc.get_balance(), 200.0)


class TestWithdraw(unittest.TestCase):
    def setUp(self):
        self.acc = BankAccount("WD-001", initial_balance=500.0)

    def test_withdraw_decreases_balance(self):
        self.acc.withdraw(100.0)
        self.assertEqual(self.acc.get_balance(), 400.0)

    def test_withdraw_returns_new_balance(self):
        result = self.acc.withdraw(50.0)
        self.assertEqual(result, 450.0)

    def test_zero_withdrawal_raises(self):
        with self.assertRaises(InvalidAmountError):
            self.acc.withdraw(0)

    def test_negative_withdrawal_raises(self):
        with self.assertRaises(InvalidAmountError):
            self.acc.withdraw(-10)

    def test_overdraft_raises(self):
        with self.assertRaises(InsufficientFundsError):
            self.acc.withdraw(501.0)

    def test_exact_balance_withdrawal_succeeds(self):
        self.acc.withdraw(500.0)
        self.assertEqual(self.acc.get_balance(), 0.0)


class TestTransactionLog(unittest.TestCase):
    def test_log_records_deposit(self):
        acc = BankAccount("LOG-001")
        acc.deposit(100.0)
        log = acc.get_transaction_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["type"], "deposit")
        self.assertEqual(log[0]["amount"], 100.0)

    def test_log_records_withdrawal(self):
        acc = BankAccount("LOG-002", initial_balance=200.0)
        acc.withdraw(50.0)
        log = acc.get_transaction_log()
        self.assertEqual(log[0]["type"], "withdrawal")

    def test_log_is_copy(self):
        """Mutating the returned log should not affect the internal state."""
        acc = BankAccount("LOG-003")
        acc.deposit(100.0)
        log = acc.get_transaction_log()
        log.clear()
        self.assertEqual(len(acc.get_transaction_log()), 1)


# ---------------------------------------------------------------------------
# Transfer Tests
# ---------------------------------------------------------------------------

class TestTransfer(unittest.TestCase):
    def setUp(self):
        self.alice = BankAccount("A-001", initial_balance=1000.0)
        self.bob   = BankAccount("A-002", initial_balance=500.0)

    def test_transfer_moves_funds(self):
        transfer(self.alice, self.bob, 200.0)
        self.assertEqual(self.alice.get_balance(), 800.0)
        self.assertEqual(self.bob.get_balance(), 700.0)

    def test_transfer_insufficient_funds_raises(self):
        with self.assertRaises(InsufficientFundsError):
            transfer(self.alice, self.bob, 1500.0)

    def test_transfer_zero_raises(self):
        with self.assertRaises(InvalidAmountError):
            transfer(self.alice, self.bob, 0)

    def test_transfer_preserves_total(self):
        total_before = self.alice.get_balance() + self.bob.get_balance()
        transfer(self.alice, self.bob, 300.0)
        total_after = self.alice.get_balance() + self.bob.get_balance()
        self.assertAlmostEqual(total_before, total_after, places=5)


# ---------------------------------------------------------------------------
# Concurrency / Thread-Safety Tests
# ---------------------------------------------------------------------------

class TestConcurrentDeposits(unittest.TestCase):
    """
    Concurrent correctness test: many threads each deposit a fixed amount.
    The final balance must equal threads × amount_per_thread.
    Without proper locking this test would fail intermittently.
    """

    def test_concurrent_deposits_are_accurate(self):
        acc = BankAccount("CONC-001", initial_balance=0.0)
        num_threads = 50
        amount = 10.0

        threads = [
            threading.Thread(target=acc.deposit, args=(amount,))
            for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertAlmostEqual(acc.get_balance(), num_threads * amount, places=5)

    def test_concurrent_withdrawals_never_go_negative(self):
        """
        Multiple threads attempt withdrawals; the balance must never drop below zero.
        """
        acc = BankAccount("CONC-002", initial_balance=1000.0)
        errors = []
        lock = threading.Lock()

        def attempt_withdraw():
            try:
                acc.withdraw(50.0)
            except InsufficientFundsError:
                pass
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=attempt_withdraw) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertGreaterEqual(acc.get_balance(), 0.0)
        self.assertEqual(len(errors), 0)

    def test_mixed_concurrent_transactions_preserve_total(self):
        """
        Concurrent deposits and withdrawals on two accounts via transfer
        must preserve the combined total.
        """
        alice = BankAccount("CONC-A", initial_balance=5000.0)
        bob   = BankAccount("CONC-B", initial_balance=5000.0)
        total_before = alice.get_balance() + bob.get_balance()

        def transfer_task():
            for _ in range(20):
                try:
                    transfer(alice, bob, 10.0)
                    transfer(bob, alice, 10.0)
                except InsufficientFundsError:
                    pass
                time.sleep(0.001)

        threads = [threading.Thread(target=transfer_task) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_after = alice.get_balance() + bob.get_balance()
        self.assertAlmostEqual(total_before, total_after, places=5)


# ---------------------------------------------------------------------------
# TransactionSimulator Tests
# ---------------------------------------------------------------------------

class TestTransactionSimulator(unittest.TestCase):
    def test_simulator_runs_without_errors(self):
        alice = BankAccount("SIM-001", initial_balance=2000.0)
        bob   = BankAccount("SIM-002", initial_balance=2000.0)
        sim = TransactionSimulator([alice, bob], num_users=5, transactions_per_user=10)
        summary = sim.run()
        self.assertEqual(summary["errors"], 0)

    def test_simulator_balances_are_non_negative(self):
        alice = BankAccount("SIM-003", initial_balance=1000.0)
        sim = TransactionSimulator([alice], num_users=8, transactions_per_user=15)
        sim.run()
        self.assertGreaterEqual(alice.get_balance(), 0.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
