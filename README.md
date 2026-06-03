# Advanced Object-Oriented Programming: E-Portfolio

> Module: Advanced Object-Oriented Programming

## Contents

| File | Description |
|------|-------------|
| `part1/banking_system.py` | Thread-safe banking system (main assignment) |
| `part1/test_banking_system.py` | 26 unit tests covering concurrency |
| `artifact1/artifact1_solid_shopping.py` | SOLID principles refactoring |
| `artifact2/artifact2_shopease.py` | Layered OOP e-commerce architecture |
| `artifact3/artifact3_elearning.py` | TDD e-learning platform (tests + implementation) |
| `artifact4/artifact4_di.py` | Dependency Injection & IoC container |
| `index.html` | Portfolio page |

## Running the code

```bash
# Run all tests
python -m pytest part1/test_banking_system.py artifact3/artifact3_elearning.py artifact4/artifact4_di.py -v

# Run individual demos
python part1/banking_system.py
python artifact1/artifact1_solid_shopping.py
python artifact2/artifact2_shopease.py
python artifact3/artifact3_elearning.py
python artifact4/artifact4_di.py
```

## Part 1: Thread-Safe Banking System

Implements `BankAccount` with `deposit()`, `withdraw()`, `get_balance()` and a `transfer()` utility.

**Thread safety:** All methods acquire an `RLock` before modifying state.  
**Deadlock prevention:** `transfer()` always acquires locks in account-number order, eliminating circular wait.

Key classes:
- `BankAccount` : thread-safe account with audit log
- `TransactionSimulator` : concurrent load tester, verifies money is conserved
- `transfer()` : deadlock-safe cross-account transfer

## Artifact 1 : SOLID Shopping System

Refactors a monolithic `Order` class:
- **S** : `Order` manages items only; `PaymentMethod` handles payments
- **O** : new payment methods added without modifying existing classes
- **L** : `CryptoPayment` is fully substitutable for `PaymentMethod`
- **I** : focused single-method interfaces
- **D** : `Order` depends on `PaymentMethod` abstraction, not concrete classes

## Artifact 2 : ShopEase Architecture

Layered architecture (Presentation → Business Logic → Data Access) with:
- **Observer Pattern** : `EventBus` for decoupled notifications
- **Repository Pattern** : storage decoupled from business rules
- **Security** : PBKDF2-HMAC-SHA256 passwords, timing-safe comparisons

## Artifact 3 : E-Learning Platform (TDD)

Tests written _before_ implementation (Red → Green → Refactor):
- `UserManagementService` : registration, auth, role-based access
- `CourseManagementService` : instructor-only course creation
- `EnrolmentService` : capacity limits, progress tracking, completion

## Artifact 4 : Dependency Injection

Demonstrates constructor injection eliminating tight coupling:
- `NotificationService` abstract base (DIP)
- `EmailService`, `SMSService`, `PushNotificationService` (LSP / OCP)
- `UserManager` accepts any `NotificationService` : zero code changes to swap
- `DIContainer` : lightweight IoC container (pure stdlib)
- Unit tests using `MockNotificationService` : fast, no real email sent

## Academic integrity

All code written originally for this portfolio. External references:
- Python `threading` module documentation: https://docs.python.org/3/library/threading.html
- NIST SP 800-63B (password hashing guidance): https://pages.nist.gov/800-63-3/
- SOLID principles: Martin, R.C. (2003) *Agile Software Development*
