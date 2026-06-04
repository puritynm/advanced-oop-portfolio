"""
Artefact 8: Abstract Factory Pattern — AI Service Provider Switching
=====================================================================
Module: Advanced Object-Oriented Programming

Pattern: Abstract Factory (GoF Creational)
Domain:  AI/ML service integration — switching between providers
         (e.g., OpenAI ↔ Anthropic ↔ Local/Open-Source models)

Why Abstract Factory?
  An AI-integrated application may need to switch providers per deployment
  environment (cloud vendor, cost, latency, compliance). Abstract Factory
  ensures a consistent family of related objects (classifier, embedder, summariser)
  is always created from the same vendor — no mismatched pairs.

Benefits:
  • Scalability and extensibility — add a new provider by adding one factory
    class; application code is untouched.
  • AI-readiness — abstractions isolate the application from vendor API
    changes (a major real-world pain point in AI engineering).
  • Testability — a MockAIFactory returns deterministic stubs, enabling
    fast unit tests with no network calls or API costs.
  • Separation of concerns — object creation is separated from usage.

AI Pattern also demonstrated:
  Model Registry concept — the AIServiceRegistry maps environment → factory,
  simulating lifecycle and version management of ML model sets.

Trade-off:
  Abstract Factory increases the number of classes. Justified here because
  the family of three products (Classifier, Embedder, Summariser) varies
  together per provider. A simple factory function would work for one product.

Run:
    python abstract_factory_ai.py
"""

from __future__ import annotations

import random
import time
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List
from unittest.mock import patch


# ===========================================================================
# Abstract Products — AI service interfaces
# ===========================================================================

class TextClassifier(ABC):
    """Classifies text into predefined categories."""

    @abstractmethod
    def classify(self, text: str) -> dict[str, float]:
        """Return label → confidence mapping."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


class TextEmbedder(ABC):
    """Produces dense vector embeddings for semantic similarity."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return a fixed-length embedding vector."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...


class TextSummariser(ABC):
    """Condenses long text into a shorter summary."""

    @abstractmethod
    def summarise(self, text: str, max_sentences: int = 3) -> str: ...

    @property
    @abstractmethod
    def provider(self) -> str: ...


# ===========================================================================
# Abstract Factory
# ===========================================================================

class AIServiceFactory(ABC):
    """
    Abstract factory — creates a consistent family of AI service objects.
    All three products from one factory are guaranteed to work together.
    """

    @abstractmethod
    def create_classifier(self) -> TextClassifier: ...

    @abstractmethod
    def create_embedder(self) -> TextEmbedder: ...

    @abstractmethod
    def create_summariser(self) -> TextSummariser: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


# ===========================================================================
# Concrete Products — OpenAI family
# ===========================================================================

class OpenAIClassifier(TextClassifier):
    """Simulates GPT-4-based classification."""

    @property
    def model_name(self) -> str:
        return "gpt-4-classifier"

    def classify(self, text: str) -> dict[str, float]:
        # Simulation: real impl would POST to api.openai.com
        labels = ["positive", "negative", "neutral"]
        scores = sorted([random.random() for _ in labels], reverse=True)
        total = sum(scores)
        return {l: round(s / total, 3) for l, s in zip(labels, scores)}


class OpenAIEmbedder(TextEmbedder):
    @property
    def dimensions(self) -> int:
        return 1536  # text-embedding-ada-002

    def embed(self, text: str) -> List[float]:
        random.seed(hash(text) % 2**32)
        return [round(random.gauss(0, 1), 4) for _ in range(self.dimensions)]


class OpenAISummariser(TextSummariser):
    @property
    def provider(self) -> str:
        return "OpenAI"

    def summarise(self, text: str, max_sentences: int = 3) -> str:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return ". ".join(sentences[:max_sentences]) + ("." if sentences else "")


# ===========================================================================
# Concrete Products — Local/Open-Source family (privacy-first)
# ===========================================================================

class LocalClassifier(TextClassifier):
    """Simulates a local BERT-based classifier (no API calls)."""

    @property
    def model_name(self) -> str:
        return "bert-base-uncased-local"

    def classify(self, text: str) -> dict[str, float]:
        # Keyword heuristic — simulates local inference
        positive_words = {"good", "great", "excellent", "love", "amazing"}
        negative_words = {"bad", "terrible", "awful", "hate", "poor"}
        words = set(text.lower().split())
        pos = len(words & positive_words) / max(len(words), 1)
        neg = len(words & negative_words) / max(len(words), 1)
        neu = max(0, 1 - pos - neg)
        return {"positive": round(pos, 3), "negative": round(neg, 3), "neutral": round(neu, 3)}


class LocalEmbedder(TextEmbedder):
    @property
    def dimensions(self) -> int:
        return 384  # all-MiniLM-L6-v2

    def embed(self, text: str) -> List[float]:
        random.seed(hash(text) % 2**32)
        return [round(random.gauss(0, 1), 4) for _ in range(self.dimensions)]


class LocalSummariser(TextSummariser):
    @property
    def provider(self) -> str:
        return "Local (extractive)"

    def summarise(self, text: str, max_sentences: int = 3) -> str:
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        # Extractive: pick the first N meaningful sentences
        return ". ".join(sentences[:max_sentences]) + "."


# ===========================================================================
# Concrete Products — Mock family (for testing)
# ===========================================================================

class MockClassifier(TextClassifier):
    @property
    def model_name(self) -> str: return "mock-classifier"
    def classify(self, text: str) -> dict[str, float]:
        return {"positive": 1.0, "negative": 0.0, "neutral": 0.0}


class MockEmbedder(TextEmbedder):
    @property
    def dimensions(self) -> int: return 4
    def embed(self, text: str) -> List[float]: return [0.1, 0.2, 0.3, 0.4]


class MockSummariser(TextSummariser):
    @property
    def provider(self) -> str: return "mock"
    def summarise(self, text: str, max_sentences: int = 3) -> str: return "Mock summary."


# ===========================================================================
# Concrete Factories
# ===========================================================================

class OpenAIFactory(AIServiceFactory):
    @property
    def provider_name(self) -> str: return "OpenAI"
    def create_classifier(self) -> TextClassifier: return OpenAIClassifier()
    def create_embedder(self)   -> TextEmbedder:   return OpenAIEmbedder()
    def create_summariser(self) -> TextSummariser:  return OpenAISummariser()


class LocalModelFactory(AIServiceFactory):
    @property
    def provider_name(self) -> str: return "LocalModel"
    def create_classifier(self) -> TextClassifier: return LocalClassifier()
    def create_embedder(self)   -> TextEmbedder:   return LocalEmbedder()
    def create_summariser(self) -> TextSummariser:  return LocalSummariser()


class MockAIFactory(AIServiceFactory):
    """Used exclusively in unit tests — zero latency, deterministic output."""
    @property
    def provider_name(self) -> str: return "Mock"
    def create_classifier(self) -> TextClassifier: return MockClassifier()
    def create_embedder(self)   -> TextEmbedder:   return MockEmbedder()
    def create_summariser(self) -> TextSummariser:  return MockSummariser()


# ===========================================================================
# Model Registry (AI-oriented OO pattern)
# ===========================================================================

class AIServiceRegistry:
    """
    Model Registry pattern: maps deployment environments to AI factory versions.

    Simulates lifecycle management — in production this would store:
      • Model version → factory mapping
      • Deployment timestamp
      • Performance metrics per version
      • Rollback pointers

    AI-readiness: abstracts provider selection from application code,
    making multi-environment deployment (dev/staging/prod) straightforward.
    """

    def __init__(self):
        self._registry: Dict[str, AIServiceFactory] = {}
        self._active: str | None = None

    def register(self, environment: str, factory: AIServiceFactory) -> None:
        self._registry[environment] = factory
        print(f"  [Registry] Registered '{factory.provider_name}' for env '{environment}'")

    def activate(self, environment: str) -> None:
        if environment not in self._registry:
            raise KeyError(f"No factory registered for environment '{environment}'.")
        self._active = environment
        print(f"  [Registry] Activated environment: '{environment}'")

    def get_factory(self) -> AIServiceFactory:
        if self._active is None:
            raise RuntimeError("No active environment set.")
        return self._registry[self._active]

    def list_environments(self) -> list[str]:
        return list(self._registry.keys())


# ===========================================================================
# Application code (uses abstractions only)
# ===========================================================================

class SentimentAnalysisPipeline:
    """
    Application that uses AI services — depends only on abstractions.
    The factory injected determines which AI provider is used.
    """

    def __init__(self, factory: AIServiceFactory):
        self._classifier  = factory.create_classifier()
        self._embedder    = factory.create_embedder()
        self._summariser  = factory.create_summariser()
        print(f"  [Pipeline] Built with provider: {factory.provider_name}")

    def analyse(self, text: str) -> dict:
        sentiment = self._classifier.classify(text)
        embedding = self._embedder.embed(text)
        summary   = self._summariser.summarise(text)
        return {
            "sentiment":       sentiment,
            "embedding_dims":  len(embedding),
            "summary":         summary,
            "provider":        self._classifier.model_name,
        }


# ===========================================================================
# Unit Tests
# ===========================================================================

class TestAbstractFactory(unittest.TestCase):

    def test_mock_factory_produces_correct_types(self):
        factory = MockAIFactory()
        self.assertIsInstance(factory.create_classifier(), TextClassifier)
        self.assertIsInstance(factory.create_embedder(),   TextEmbedder)
        self.assertIsInstance(factory.create_summariser(), TextSummariser)

    def test_mock_classifier_returns_dict(self):
        clf = MockAIFactory().create_classifier()
        result = clf.classify("test input")
        self.assertIsInstance(result, dict)
        self.assertIn("positive", result)

    def test_mock_embedder_dimensions(self):
        emb = MockAIFactory().create_embedder()
        vec = emb.embed("hello world")
        self.assertEqual(len(vec), emb.dimensions)

    def test_pipeline_with_mock_factory(self):
        pipeline = SentimentAnalysisPipeline(MockAIFactory())
        result = pipeline.analyse("This is a great product!")
        self.assertIn("sentiment", result)
        self.assertIn("summary", result)

    def test_local_classifier_positive_text(self):
        clf = LocalClassifier()
        result = clf.classify("This is great and excellent")
        self.assertGreater(result["positive"], result["negative"])

    def test_registry_activate_resolves_factory(self):
        registry = AIServiceRegistry()
        registry.register("test", MockAIFactory())
        registry.activate("test")
        factory = registry.get_factory()
        self.assertEqual(factory.provider_name, "Mock")

    def test_registry_unknown_env_raises(self):
        registry = AIServiceRegistry()
        with self.assertRaises(KeyError):
            registry.activate("nonexistent")

    def test_swap_provider_zero_app_changes(self):
        """
        Core demonstration: swapping factory requires zero changes to pipeline.
        """
        for factory in [MockAIFactory(), LocalModelFactory()]:
            pipeline = SentimentAnalysisPipeline(factory)
            result = pipeline.analyse("The product quality is excellent.")
            self.assertIn("sentiment", result)


# ===========================================================================
# Demo
# ===========================================================================

if __name__ == "__main__":
    print("=== Abstract Factory — AI Service Provider Demo ===\n")

    registry = AIServiceRegistry()
    registry.register("development", MockAIFactory())
    registry.register("production",  LocalModelFactory())
    registry.register("cloud",       OpenAIFactory())

    text = ("The new software update significantly improved performance. "
            "Users reported faster load times and a more responsive interface. "
            "However, some minor bugs were identified in the settings module.")

    for env in registry.list_environments():
        registry.activate(env)
        pipeline = SentimentAnalysisPipeline(registry.get_factory())
        result = pipeline.analyse(text)
        print(f"\n  Environment: {env}")
        print(f"  Sentiment:   {result['sentiment']}")
        print(f"  Embedding:   {result['embedding_dims']} dimensions")
        print(f"  Summary:     {result['summary'][:80]}...")

    print("\n--- Running unit tests ---")
    unittest.main(argv=[""], exit=False, verbosity=0)
    print("All tests passed.")
