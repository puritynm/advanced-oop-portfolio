"""
Part 1: Thread-Safe Banking System
===================================
Module: Advanced Object-Oriented Programming
Description: A concurrent banking system using locks for thread safety,
             with deadlock prevention via ordered lock acquisition.
"""

import threading
import time
import random
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds the available balance."""
    pass


class InvalidAmountError(Exception):
    """Raised when a transaction amount is zero or negative."""
    pass


class BankAccount:
    """
    A thread-safe bank account supporting concurrent deposits,
    withdrawals, and balance queries.

    Thread Safety:
        All public methods acquire self._lock before modifying state,
        preventing race conditions in concurrent environments.

    Attributes:
        account_number (str): Unique identifier for the account.
        _balance (float): Internal balance protected by _lock.
        _lock (threading.RLock): Reentrant lock for thread safety.
        _transaction_log (list): Audit trail of all transactions.
    """

    # Class-level counter for unique ID generation (thread-safe via lock)
    _id_counter = 0
    _id_lock = threading.Lock()

    def __init__(self, account_number: str, initial_balance: float = 0.0):
        if initial_balance < 0:
            raise InvalidAmountError("Initial balance cannot be negative.")
        self.account_number = account_number
        self._balance = initial_balance
        # RLock allows the same thread to acquire the lock multiple times,
        # preventing self-deadlock in methods that call each other.
        self._lock = threading.RLock()
        self._transaction_log: list[dict] = []
        logger.info(f"Account {account_number} created with balance £{initial_balance:.2f}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def deposit(self, amount: float) -> float:
        """
        Deposit a positive amount into the account.

        Args:
            amount: The amount to deposit (must be > 0).

        Returns:
            The new balance after the deposit.

        Raises:
            InvalidAmountError: If amount is not positive.
        """
        self._validate_amount(amount)
        with self._lock:
            self._balance += amount
            self._log_transaction("deposit", amount, self._balance)
            logger.info(
                f"Account {self.account_number}: Deposited £{amount:.2f} → Balance £{self._balance:.2f}"
            )
            return self._balance

    def withdraw(self, amount: float) -> float:
        """
        Withdraw a positive amount from the account if funds allow.

        Args:
            amount: The amount to withdraw (must be > 0).

        Returns:
            The new balance after the withdrawal.

        Raises:
            InvalidAmountError: If amount is not positive.
            InsufficientFundsError: If balance is insufficient.
        """
        self._validate_amount(amount)
        with self._lock:
            if amount > self._balance:
                raise InsufficientFundsError(
                    f"Cannot withdraw £{amount:.2f}: balance is £{self._balance:.2f}"
                )
            self._balance -= amount
            self._log_transaction("withdrawal", amount, self._balance)
            logger.info(
                f"Account {self.account_number}: Withdrew £{amount:.2f} → Balance £{self._balance:.2f}"
            )
            return self._balance

    def get_balance(self) -> float:
        """
        Return the current balance (thread-safe snapshot).

        Returns:
            Current balance as a float.
        """
        with self._lock:
            return self._balance

    def get_transaction_log(self) -> list[dict]:
        """Return a copy of the transaction log for auditing."""
        with self._lock:
            return list(self._transaction_log)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_amount(amount: float) -> None:
        if amount <= 0:
            raise InvalidAmountError(f"Amount must be positive, got {amount}")

    def _log_transaction(self, t_type: str, amount: float, balance_after: float) -> None:
        """Record a transaction in the audit log (called inside lock)."""
        self._transaction_log.append({
            "type": t_type,
            "amount": amount,
            "balance_after": balance_after,
            "timestamp": time.time(),
        })

    def __repr__(self) -> str:
        return f"BankAccount(account_number={self.account_number!r}, balance={self._balance:.2f})"


# ---------------------------------------------------------------------------
# Deadlock-Safe Transfer Utility
# ---------------------------------------------------------------------------

def transfer(source: BankAccount, destination: BankAccount, amount: float) -> None:
    """
    Transfer funds between two accounts without risk of deadlock.

    Deadlock Prevention Strategy:
        Always acquire locks in a consistent global order (by account_number).
        This breaks the circular wait condition required for deadlock.

    Args:
        source: Account to debit.
        destination: Account to credit.
        amount: Amount to transfer (must be > 0).

    Raises:
        InsufficientFundsError: If source has insufficient funds.
        InvalidAmountError: If amount is not positive.
    """
    BankAccount._validate_amount(amount)

    # Determine a consistent lock-acquisition order to avoid deadlock
    first, second = (
        (source, destination)
        if source.account_number < destination.account_number
        else (destination, source)
    )

    with first._lock:
        with second._lock:
            # Now both locks are held — perform the transfer atomically
            if source._balance < amount:
                raise InsufficientFundsError(
                    f"Transfer failed: insufficient funds in {source.account_number}"
                )
            source._balance -= amount
            source._log_transaction("transfer_out", amount, source._balance)

            destination._balance += amount
            destination._log_transaction("transfer_in", amount, destination._balance)

            logger.info(
                f"Transferred £{amount:.2f} from {source.account_number} "
                f"to {destination.account_number}"
            )


# ---------------------------------------------------------------------------
# Transaction Simulator
# ---------------------------------------------------------------------------

class TransactionSimulator:
    """
    Simulates multiple users performing concurrent transactions on bank accounts.

    Each simulated user runs in its own thread and performs a random mix of
    deposits and withdrawals, exercising the thread-safety guarantees of
    BankAccount.

    Attributes:
        accounts (list[BankAccount]): Accounts available for simulation.
        num_users (int): Number of concurrent user threads.
        transactions_per_user (int): Operations each user will perform.
    """

    def __init__(
        self,
        accounts: list[BankAccount],
        num_users: int = 5,
        transactions_per_user: int = 10,
    ):
        self.accounts = accounts
        self.num_users = num_users
        self.transactions_per_user = transactions_per_user
        self._threads: list[threading.Thread] = []
        self._errors: list[Exception] = []
        self._errors_lock = threading.Lock()

    def _user_session(self, user_id: int) -> None:
        """Simulate a single user's transaction session."""
        account = random.choice(self.accounts)
        for i in range(self.transactions_per_user):
            try:
                action = random.choice(["deposit", "withdraw", "transfer"])

                if action == "deposit":
                    amount = round(random.uniform(10, 500), 2)
                    account.deposit(amount)

                elif action == "withdraw":
                    amount = round(random.uniform(10, 200), 2)
                    try:
                        account.withdraw(amount)
                    except InsufficientFundsError:
                        pass  # Expected; not an error condition

                elif action == "transfer" and len(self.accounts) > 1:
                    dest = random.choice(
                        [a for a in self.accounts if a is not account]
                    )
                    amount = round(random.uniform(10, 100), 2)
                    try:
                        transfer(account, dest, amount)
                    except InsufficientFundsError:
                        pass

                # Small delay to increase thread interleaving
                time.sleep(random.uniform(0.001, 0.01))

            except Exception as exc:
                with self._errors_lock:
                    self._errors.append(exc)
                logger.error(f"User {user_id} encountered error: {exc}")

    def run(self) -> dict:
        """
        Launch all user threads and wait for them to complete.

        Returns:
            A summary dict with final balances and any errors encountered.
        """
        logger.info(
            f"Starting simulation: {self.num_users} users × "
            f"{self.transactions_per_user} transactions"
        )
        self._threads = [
            threading.Thread(
                target=self._user_session,
                args=(uid,),
                name=f"User-{uid}",
                daemon=True,
            )
            for uid in range(self.num_users)
        ]

        start = time.perf_counter()
        for t in self._threads:
            t.start()
        for t in self._threads:
            t.join()
        elapsed = time.perf_counter() - start

        summary = {
            "elapsed_seconds": round(elapsed, 3),
            "errors": len(self._errors),
            "final_balances": {
                acc.account_number: acc.get_balance() for acc in self.accounts
            },
        }
        logger.info(f"Simulation complete in {elapsed:.3f}s — summary: {summary}")
        return summary


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create two accounts with seed balances
    alice = BankAccount("ACC-001", initial_balance=1000.0)
    bob   = BankAccount("ACC-002", initial_balance=1000.0)

    # Record combined balance before simulation
    total_before = alice.get_balance() + bob.get_balance()

    # Run the concurrent simulation
    simulator = TransactionSimulator(
        accounts=[alice, bob],
        num_users=10,
        transactions_per_user=20,
    )
    summary = simulator.run()

    # Verify conservation of money (transfers should not create/destroy funds)
    total_after = alice.get_balance() + bob.get_balance()
    print("\n=== Simulation Summary ===")
    print(f"Total balance before: £{total_before:.2f}")
    print(f"Total balance after:  £{total_after:.2f}")
    print(f"Balance conserved:    {abs(total_before - total_after) < 0.01}")
    print(f"Errors encountered:   {summary['errors']}")
    print(f"Elapsed time:         {summary['elapsed_seconds']}s")
