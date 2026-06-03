"""
Artifact 3: Secure E-Learning Platform — TDD Implementation
=============================================================
Module: Advanced Object-Oriented Programming

Architecture: Layered monolith (justified below)
  • Scalability   — modules are loosely coupled; can be extracted to microservices later
  • Maintainability — each module has a single clear responsibility
  • Security       — hashed passwords, input validation, role-based access

Modules implemented:
  1. User Management  (with TDD — tests written first)
  2. Course Management
  3. Enrolment Management

TDD Cycle followed for User Management:
  Red   → write failing test
  Green → write minimal code to pass
  Refactor → improve without breaking tests

Run tests:
    python -m pytest artifact3_elearning.py -v
Run demo:
    python artifact3_elearning.py
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
import unittest
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


# ===========================================================================
# Shared Utilities
# ===========================================================================

def new_id() -> str:
    return str(uuid.uuid4())[:8].upper()


def secure_hash(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """PBKDF2-HMAC-SHA256 with 260 000 iterations (NIST recommendation)."""
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return dk.hex(), salt.hex()


def verify_hash(password: str, stored_hash: str, salt_hex: str) -> bool:
    candidate, _ = secure_hash(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, stored_hash)


def validate_email(email: str) -> bool:
    """Basic RFC-5322-ish email validation."""
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


# ===========================================================================
# Domain Entities
# ===========================================================================

class Role(Enum):
    STUDENT    = auto()
    INSTRUCTOR = auto()
    ADMIN      = auto()


@dataclass
class User:
    user_id: str = field(default_factory=new_id)
    username: str = ""
    email: str = ""
    role: Role = Role.STUDENT
    _hash: str = field(default="", repr=False)
    _salt: str = field(default="", repr=False)
    created_at: float = field(default_factory=time.time)
    is_active: bool = True

    def set_password(self, plain: str) -> None:
        if len(plain) < 8:
            raise ValueError("Password must be at least 8 characters.")
        self._hash, self._salt = secure_hash(plain)

    def check_password(self, plain: str) -> bool:
        return verify_hash(plain, self._hash, self._salt)


@dataclass
class Course:
    course_id: str = field(default_factory=new_id)
    title: str = ""
    description: str = ""
    instructor_id: str = ""
    max_capacity: int = 30
    created_at: float = field(default_factory=time.time)
    is_published: bool = False


class EnrolmentStatus(Enum):
    ACTIVE    = auto()
    COMPLETED = auto()
    DROPPED   = auto()


@dataclass
class Enrolment:
    enrolment_id: str = field(default_factory=new_id)
    user_id: str = ""
    course_id: str = ""
    status: EnrolmentStatus = EnrolmentStatus.ACTIVE
    enrolled_at: float = field(default_factory=time.time)
    progress: float = 0.0  # 0.0 – 100.0 %


# ===========================================================================
# Repositories
# ===========================================================================

class UserRepository:
    def __init__(self): self._db: Dict[str, User] = {}
    def save(self, u: User):  self._db[u.user_id] = u
    def get(self, uid: str) -> Optional[User]: return self._db.get(uid)
    def by_username(self, name: str) -> Optional[User]:
        return next((u for u in self._db.values() if u.username == name), None)
    def all(self) -> List[User]: return list(self._db.values())


class CourseRepository:
    def __init__(self): self._db: Dict[str, Course] = {}
    def save(self, c: Course):  self._db[c.course_id] = c
    def get(self, cid: str) -> Optional[Course]: return self._db.get(cid)
    def all(self) -> List[Course]: return list(self._db.values())
    def published(self) -> List[Course]: return [c for c in self._db.values() if c.is_published]


class EnrolmentRepository:
    def __init__(self): self._db: Dict[str, Enrolment] = {}
    def save(self, e: Enrolment): self._db[e.enrolment_id] = e
    def get(self, eid: str) -> Optional[Enrolment]: return self._db.get(eid)
    def by_user(self, uid: str) -> List[Enrolment]:
        return [e for e in self._db.values() if e.user_id == uid]
    def by_course(self, cid: str) -> List[Enrolment]:
        return [e for e in self._db.values() if e.course_id == cid]
    def find(self, uid: str, cid: str) -> Optional[Enrolment]:
        return next(
            (e for e in self._db.values() if e.user_id == uid and e.course_id == cid),
            None,
        )


# ===========================================================================
# Business Logic — Services (TDD-driven)
# ===========================================================================

class UserManagementService:
    """
    Manages user registration, authentication, and profile operations.

    Security:
      - Passwords hashed with PBKDF2-HMAC-SHA256
      - Email validated before persistence
      - Username uniqueness enforced
    """

    def __init__(self, repo: UserRepository):
        self._repo = repo

    def register(self, username: str, email: str, password: str,
                 role: Role = Role.STUDENT) -> User:
        self._assert_valid_username(username)
        self._assert_valid_email(email)
        user = User(username=username, email=email, role=role)
        user.set_password(password)
        self._repo.save(user)
        return user

    def authenticate(self, username: str, password: str) -> User:
        user = self._repo.by_username(username)
        if not user or not user.check_password(password):
            raise PermissionError("Invalid username or password.")
        if not user.is_active:
            raise PermissionError("Account is deactivated.")
        return user

    def deactivate(self, user_id: str) -> None:
        user = self._repo.get(user_id)
        if not user:
            raise KeyError(f"User {user_id} not found.")
        user.is_active = False
        self._repo.save(user)

    # --- Validation helpers ---
    def _assert_valid_username(self, username: str) -> None:
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if self._repo.by_username(username):
            raise ValueError(f"Username '{username}' is already taken.")

    @staticmethod
    def _assert_valid_email(email: str) -> None:
        if not validate_email(email):
            raise ValueError(f"Invalid email address: {email}")


class CourseManagementService:
    """Manages course creation, publishing, and discovery."""

    def __init__(self, course_repo: CourseRepository, user_repo: UserRepository):
        self._courses = course_repo
        self._users   = user_repo

    def create_course(self, title: str, description: str,
                      instructor_id: str, max_capacity: int = 30) -> Course:
        instructor = self._users.get(instructor_id)
        if not instructor or instructor.role not in (Role.INSTRUCTOR, Role.ADMIN):
            raise PermissionError("Only instructors can create courses.")
        if not title.strip():
            raise ValueError("Course title cannot be empty.")
        course = Course(title=title, description=description,
                        instructor_id=instructor_id, max_capacity=max_capacity)
        self._courses.save(course)
        return course

    def publish(self, course_id: str) -> Course:
        course = self._courses.get(course_id)
        if not course:
            raise KeyError(f"Course {course_id} not found.")
        course.is_published = True
        self._courses.save(course)
        return course

    def list_published(self) -> List[Course]:
        return self._courses.published()


class EnrolmentService:
    """Manages student enrolment and progress tracking."""

    def __init__(
        self,
        enrolment_repo: EnrolmentRepository,
        course_repo:    CourseRepository,
        user_repo:      UserRepository,
    ):
        self._enrolments = enrolment_repo
        self._courses    = course_repo
        self._users      = user_repo

    def enrol(self, user_id: str, course_id: str) -> Enrolment:
        user   = self._users.get(user_id)
        course = self._courses.get(course_id)
        if not user:   raise KeyError("User not found.")
        if not course: raise KeyError("Course not found.")
        if not course.is_published:
            raise ValueError("Cannot enrol in an unpublished course.")
        if self._enrolments.find(user_id, course_id):
            raise ValueError("User is already enrolled in this course.")
        active = [e for e in self._enrolments.by_course(course_id)
                  if e.status == EnrolmentStatus.ACTIVE]
        if len(active) >= course.max_capacity:
            raise ValueError("Course is at full capacity.")
        e = Enrolment(user_id=user_id, course_id=course_id)
        self._enrolments.save(e)
        return e

    def update_progress(self, enrolment_id: str, progress: float) -> Enrolment:
        e = self._enrolments.get(enrolment_id)
        if not e: raise KeyError("Enrolment not found.")
        if not (0 <= progress <= 100):
            raise ValueError("Progress must be between 0 and 100.")
        e.progress = progress
        if progress >= 100:
            e.status = EnrolmentStatus.COMPLETED
        self._enrolments.save(e)
        return e

    def drop(self, enrolment_id: str) -> Enrolment:
        e = self._enrolments.get(enrolment_id)
        if not e: raise KeyError("Enrolment not found.")
        e.status = EnrolmentStatus.DROPPED
        self._enrolments.save(e)
        return e


# ===========================================================================
# TDD Unit Tests
# Written before implementation (Red → Green → Refactor)
# ===========================================================================

class TestUserManagement(unittest.TestCase):
    """Tests for UserManagementService — written before the implementation."""

    def setUp(self):
        self.repo = UserRepository()
        self.service = UserManagementService(self.repo)

    # --- Registration ---
    def test_register_creates_user(self):
        u = self.service.register("alice", "alice@edu.com", "Password1!")
        self.assertEqual(u.username, "alice")
        self.assertEqual(u.email, "alice@edu.com")
        self.assertEqual(u.role, Role.STUDENT)

    def test_register_stores_hashed_password(self):
        u = self.service.register("bob", "bob@edu.com", "Password1!")
        self.assertNotEqual(u._hash, "Password1!")
        self.assertTrue(len(u._hash) > 32)

    def test_register_duplicate_username_raises(self):
        self.service.register("carol", "carol@edu.com", "Password1!")
        with self.assertRaises(ValueError):
            self.service.register("carol", "carol2@edu.com", "Password1!")

    def test_register_invalid_email_raises(self):
        with self.assertRaises(ValueError):
            self.service.register("dave", "not-an-email", "Password1!")

    def test_register_short_username_raises(self):
        with self.assertRaises(ValueError):
            self.service.register("ab", "ab@edu.com", "Password1!")

    def test_register_short_password_raises(self):
        with self.assertRaises(ValueError):
            self.service.register("eve", "eve@edu.com", "short")

    # --- Authentication ---
    def test_authenticate_correct_password_succeeds(self):
        self.service.register("frank", "frank@edu.com", "Correct!1")
        user = self.service.authenticate("frank", "Correct!1")
        self.assertEqual(user.username, "frank")

    def test_authenticate_wrong_password_raises(self):
        self.service.register("grace", "grace@edu.com", "Right!Pass1")
        with self.assertRaises(PermissionError):
            self.service.authenticate("grace", "WrongPass1!")

    def test_authenticate_unknown_user_raises(self):
        with self.assertRaises(PermissionError):
            self.service.authenticate("nobody", "anything")

    def test_authenticate_deactivated_user_raises(self):
        u = self.service.register("harry", "h@edu.com", "SecureP@ss1")
        self.service.deactivate(u.user_id)
        with self.assertRaises(PermissionError):
            self.service.authenticate("harry", "SecureP@ss1")

    # --- Role assignment ---
    def test_register_as_instructor(self):
        u = self.service.register("ivan", "ivan@edu.com", "Teach!ng1", Role.INSTRUCTOR)
        self.assertEqual(u.role, Role.INSTRUCTOR)


class TestCourseManagement(unittest.TestCase):
    def setUp(self):
        self.user_repo   = UserRepository()
        self.course_repo = CourseRepository()
        self.users   = UserManagementService(self.user_repo)
        self.courses = CourseManagementService(self.course_repo, self.user_repo)
        self.instructor = self.users.register("prof", "prof@edu.com", "Teach!1ng", Role.INSTRUCTOR)
        self.student    = self.users.register("stu",  "stu@edu.com",  "Stud3nt!", Role.STUDENT)

    def test_instructor_can_create_course(self):
        c = self.courses.create_course("Python 101", "Intro course", self.instructor.user_id)
        self.assertEqual(c.title, "Python 101")

    def test_student_cannot_create_course(self):
        with self.assertRaises(PermissionError):
            self.courses.create_course("Hack", "desc", self.student.user_id)

    def test_empty_title_raises(self):
        with self.assertRaises(ValueError):
            self.courses.create_course("", "desc", self.instructor.user_id)

    def test_publish_marks_course_published(self):
        c = self.courses.create_course("OOP", "Advanced", self.instructor.user_id)
        self.assertFalse(c.is_published)
        self.courses.publish(c.course_id)
        self.assertTrue(self.course_repo.get(c.course_id).is_published)

    def test_list_published_returns_only_published(self):
        c1 = self.courses.create_course("A", "desc", self.instructor.user_id)
        c2 = self.courses.create_course("B", "desc", self.instructor.user_id)
        self.courses.publish(c1.course_id)
        published = self.courses.list_published()
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0].course_id, c1.course_id)


class TestEnrolment(unittest.TestCase):
    def setUp(self):
        self.user_repo      = UserRepository()
        self.course_repo    = CourseRepository()
        self.enrolment_repo = EnrolmentRepository()
        users   = UserManagementService(self.user_repo)
        courses = CourseManagementService(self.course_repo, self.user_repo)
        self.enrolments = EnrolmentService(self.enrolment_repo, self.course_repo, self.user_repo)

        self.student    = users.register("stu2", "stu2@edu.com", "Stud3nt2!")
        instructor      = users.register("prof2", "prof2@edu.com", "T3ach!ng", Role.INSTRUCTOR)
        course          = courses.create_course("AI Basics", "Intro to AI", instructor.user_id)
        self.course     = courses.publish(course.course_id)

    def test_enrol_student_succeeds(self):
        e = self.enrolments.enrol(self.student.user_id, self.course.course_id)
        self.assertEqual(e.status, EnrolmentStatus.ACTIVE)

    def test_double_enrolment_raises(self):
        self.enrolments.enrol(self.student.user_id, self.course.course_id)
        with self.assertRaises(ValueError):
            self.enrolments.enrol(self.student.user_id, self.course.course_id)

    def test_progress_update(self):
        e = self.enrolments.enrol(self.student.user_id, self.course.course_id)
        e2 = self.enrolments.update_progress(e.enrolment_id, 75.0)
        self.assertEqual(e2.progress, 75.0)
        self.assertEqual(e2.status, EnrolmentStatus.ACTIVE)

    def test_complete_on_100_percent(self):
        e = self.enrolments.enrol(self.student.user_id, self.course.course_id)
        e2 = self.enrolments.update_progress(e.enrolment_id, 100.0)
        self.assertEqual(e2.status, EnrolmentStatus.COMPLETED)

    def test_invalid_progress_raises(self):
        e = self.enrolments.enrol(self.student.user_id, self.course.course_id)
        with self.assertRaises(ValueError):
            self.enrolments.update_progress(e.enrolment_id, 110.0)


# ===========================================================================
# Demo
# ===========================================================================

if __name__ == "__main__":
    # Run unit tests first
    print("=== Running TDD Tests ===\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestUserManagement, TestCourseManagement, TestEnrolment]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n=== E-Learning Platform Demo ===\n")
    # Wire up services
    user_repo = UserRepository(); course_repo = CourseRepository()
    enrolment_repo = EnrolmentRepository()
    users     = UserManagementService(user_repo)
    courses   = CourseManagementService(course_repo, user_repo)
    enrolment = EnrolmentService(enrolment_repo, course_repo, user_repo)

    # Create accounts
    prof = users.register("dr_smith", "smith@uni.edu", "Pr0fessor!", Role.INSTRUCTOR)
    alice = users.register("alice99", "alice@student.edu", "Al1ce!Pass")
    print(f"Created instructor: {prof.username}")
    print(f"Created student:    {alice.username}")

    # Create & publish a course
    c = courses.create_course("Advanced OOP with Python", "Threads, SOLID, DI", prof.user_id)
    courses.publish(c.course_id)
    print(f"Published course:   {c.title}")

    # Enrol & track progress
    e = enrolment.enrol(alice.user_id, c.course_id)
    enrolment.update_progress(e.enrolment_id, 60.0)
    print(f"Alice's progress:   {enrolment_repo.get(e.enrolment_id).progress}%")
    enrolment.update_progress(e.enrolment_id, 100.0)
    final = enrolment_repo.get(e.enrolment_id)
    print(f"Course status:      {final.status.name}")
