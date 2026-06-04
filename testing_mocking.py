"""
Artefact 9: Mocking & AI-Driven Testing
=========================================
Module: Advanced Object-Oriented Programming

Demonstrates:
  1. Isolating classes with unittest.mock — testing without live dependencies
  2. Testing AI-dependent components without real model calls
  3. Reflection on AI-assisted testing (test-case generation, edge cases,
     regression support) — framed as augmenting engineering judgement

Why mocking for AI systems?
  AI model APIs are:
    • Slow      — network latency breaks fast TDD feedback loops
    • Costly    — API calls are metered; test suites run thousands of times
    • Non-deterministic — LLMs produce different outputs; assertions would be fragile
    • Unavailable — offline development, CI/CD pipelines without secrets
  Mocking solves all four problems while still validating integration logic.

Run:
    python testing_mocking.py
"""

from __future__ import annotations

import time
import unittest
from abc import ABC, abstractmethod
from typing import List
from unittest.mock import MagicMock, Mock, call, patch, PropertyMock


# ===========================================================================
# Production classes under test
# ===========================================================================

class AIModelClient(ABC):
    """Abstract interface to any AI inference endpoint."""

    @abstractmethod
    def predict(self, prompt: str) -> str:
        """Send a prompt; return the model's text response."""
        ...

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a list of texts."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True if the model endpoint is reachable."""
        ...


class ContentModerationService:
    """
    Classifies user-submitted content using an AI model.
    Depends on AIModelClient — injected for testability.
    """

    BLOCKED_RESPONSE = "[Content removed by moderation]"

    def __init__(self, model: AIModelClient, confidence_threshold: float = 0.7):
        self._model = model
        self._threshold = confidence_threshold
        self._moderated_count = 0

    def moderate(self, content: str) -> str:
        """
        Returns the original content if safe, or BLOCKED_RESPONSE if flagged.
        Raises RuntimeError if the model is unavailable.
        """
        if not self._model.is_available:
            raise RuntimeError("AI moderation model is unavailable.")
        response = self._model.predict(
            f"Classify this content as SAFE or UNSAFE with a confidence score: {content}"
        )
        if "UNSAFE" in response:
            self._moderated_count += 1
            return self.BLOCKED_RESPONSE
        return content

    @property
    def moderated_count(self) -> int:
        return self._moderated_count


class RecommendationEngine:
    """
    Generates product recommendations using semantic embeddings from an AI model.
    The actual similarity logic is decoupled from the model provider.
    """

    def __init__(self, model: AIModelClient):
        self._model = model

    def recommend(self, query: str, candidates: List[str], top_k: int = 3) -> List[str]:
        """Return the top_k most semantically relevant candidates."""
        if not candidates:
            return []
        all_texts  = [query] + candidates
        embeddings = self._model.get_embeddings(all_texts)
        query_vec  = embeddings[0]
        scores = [
            self._cosine_similarity(query_vec, emb)
            for emb in embeddings[1:]
        ]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:top_k]]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x ** 2 for x in a) ** 0.5
        mag_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (mag_a * mag_b + 1e-9)


# ===========================================================================
# Test Suite 1: Isolating ContentModerationService
# ===========================================================================

class TestContentModerationService(unittest.TestCase):
    """
    Tests ContentModerationService in complete isolation.
    The real AIModelClient is NEVER called — all tests are deterministic
    and run in milliseconds.
    """

    def _make_model(self, predict_return: str = "SAFE", available: bool = True) -> MagicMock:
        """Factory for a configured mock AI model."""
        mock = MagicMock(spec=AIModelClient)
        # PropertyMock required for @property attributes
        type(mock).is_available = PropertyMock(return_value=available)
        mock.predict.return_value = predict_return
        return mock

    def test_safe_content_returned_unchanged(self):
        model = self._make_model("SAFE: This content is appropriate.")
        svc = ContentModerationService(model)
        result = svc.moderate("Hello, welcome to our platform!")
        self.assertEqual(result, "Hello, welcome to our platform!")

    def test_unsafe_content_is_blocked(self):
        model = self._make_model("UNSAFE: This content violates guidelines. Confidence: 0.95")
        svc = ContentModerationService(model)
        result = svc.moderate("Harmful content here")
        self.assertEqual(result, ContentModerationService.BLOCKED_RESPONSE)

    def test_moderated_count_increments(self):
        model = self._make_model("UNSAFE: flagged")
        svc = ContentModerationService(model)
        svc.moderate("bad content 1")
        svc.moderate("bad content 2")
        self.assertEqual(svc.moderated_count, 2)

    def test_unavailable_model_raises(self):
        model = self._make_model(available=False)
        svc = ContentModerationService(model)
        with self.assertRaises(RuntimeError):
            svc.moderate("any content")

    def test_model_called_with_correct_prompt(self):
        """
        Verifies that the prompt sent to the model contains the original content.
        This tests the integration logic without caring about the model's response format.
        """
        model = self._make_model("SAFE")
        svc = ContentModerationService(model)
        svc.moderate("Test message")
        # Assert the model was called, and the call included the content
        model.predict.assert_called_once()
        args, _ = model.predict.call_args
        self.assertIn("Test message", args[0])

    def test_model_called_exactly_once_per_moderate(self):
        model = self._make_model("SAFE")
        svc = ContentModerationService(model)
        svc.moderate("message 1")
        svc.moderate("message 2")
        self.assertEqual(model.predict.call_count, 2)


# ===========================================================================
# Test Suite 2: Isolating RecommendationEngine
# ===========================================================================

class TestRecommendationEngine(unittest.TestCase):
    """
    Tests the recommendation ranking logic independently of any real embeddings.
    Mock embeddings are crafted to produce predictable similarity scores.
    """

    def _make_model(self, embeddings: List[List[float]]) -> MagicMock:
        mock = MagicMock(spec=AIModelClient)
        type(mock).is_available = PropertyMock(return_value=True)
        mock.get_embeddings.return_value = embeddings
        return mock

    def test_most_similar_returned_first(self):
        """
        [1,0] is orthogonal to [0,1] (score≈0) and parallel to [1,0] (score=1).
        """
        embeddings = [
            [1.0, 0.0],   # query
            [0.0, 1.0],   # candidate A — dissimilar
            [1.0, 0.0],   # candidate B — identical
        ]
        model = self._make_model(embeddings)
        engine = RecommendationEngine(model)
        results = engine.recommend("laptop", ["candidate_A", "candidate_B"], top_k=1)
        self.assertEqual(results, ["candidate_B"])

    def test_empty_candidates_returns_empty(self):
        model = self._make_model([])
        engine = RecommendationEngine(model)
        results = engine.recommend("query", [])
        self.assertEqual(results, [])

    def test_top_k_limits_results(self):
        embeddings = [[1.0]] * 6  # query + 5 identical candidates
        model = self._make_model(embeddings)
        engine = RecommendationEngine(model)
        results = engine.recommend("q", ["a", "b", "c", "d", "e"], top_k=2)
        self.assertEqual(len(results), 2)

    def test_embeddings_called_with_all_texts(self):
        """Verify that both query and all candidates are sent in one batch call."""
        embeddings = [[1.0, 0.0]] * 4  # query + 3 candidates
        model = self._make_model(embeddings)
        engine = RecommendationEngine(model)
        engine.recommend("search query", ["A", "B", "C"])
        args, _ = model.get_embeddings.call_args
        self.assertIn("search query", args[0])
        self.assertIn("A", args[0])
        self.assertIn("C", args[0])


# ===========================================================================
# Test Suite 3: Edge Cases (AI-assisted test discovery illustration)
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    """
    Edge cases that an AI-assisted test generator would typically surface.

    AI testing tools (e.g., Pynguin, GitHub Copilot test generation, Diffblue)
    are effective at discovering:
      • Boundary conditions (empty inputs, single items, max values)
      • Type mismatches and None handling
      • Concurrent access patterns
      • Equivalence partitioning

    Note: AI-generated tests require engineering review. They verify *behaviour*
    but cannot replace human judgement about what behaviour is *correct*.
    """

    def _svc_safe(self) -> ContentModerationService:
        mock = MagicMock(spec=AIModelClient)
        type(mock).is_available = PropertyMock(return_value=True)
        mock.predict.return_value = "SAFE"
        return ContentModerationService(mock)

    def test_empty_string_content(self):
        """Edge: empty content should not crash the service."""
        svc = self._svc_safe()
        result = svc.moderate("")
        self.assertEqual(result, "")

    def test_very_long_content(self):
        """Edge: content approaching token limits."""
        svc = self._svc_safe()
        long_text = "word " * 5000
        result = svc.moderate(long_text)
        self.assertIsNotNone(result)

    def test_unicode_content(self):
        """Edge: non-ASCII characters."""
        svc = self._svc_safe()
        result = svc.moderate("こんにちは — مرحبا — Привет")
        self.assertIsNotNone(result)

    def test_content_with_sql_injection(self):
        """Security edge: injection strings should not break the pipeline."""
        svc = self._svc_safe()
        result = svc.moderate("'; DROP TABLE users; --")
        self.assertIsNotNone(result)


# ===========================================================================
# Demo
# ===========================================================================

if __name__ == "__main__":
    print("=== Mocking & AI-Driven Testing Demo ===\n")

    print("The tests below run WITHOUT any real AI model or API call.")
    print("Mock objects replace the live AIModelClient — deterministic and instant.\n")

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestContentModerationService, TestRecommendationEngine, TestEdgeCases]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    t0 = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed = time.perf_counter() - t0

    print(f"\nAll {result.testsRun} tests completed in {elapsed*1000:.0f}ms")
    print("(A real API call would add 500ms–5s per test — mocking gives 100× speedup)")
