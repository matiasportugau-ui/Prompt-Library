#!/usr/bin/env python3
"""
Prompt Classifier Module
Automatically categorizes prompts by domain, type, and extracts keywords.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PromptClassification:
    """Classification result for a prompt."""

    domain: str
    prompt_type: str
    keywords: List[str]
    version: Optional[str]
    language: str
    quality_score: float  # 0.0 - 1.0


class PromptClassifier:
    """Classifies prompts based on content and metadata."""

    # Domain detection patterns
    DOMAIN_PATTERNS = {
        "chatbot": [
            r"chatbot",
            r"chat\s*bot",
            r"conversaci[oó]n",
            r"mensaje",
            r"respuesta",
            r"bot",
            r"asistente",
            r"customer\s*service",
        ],
        "shopify": [
            r"shopify",
            r"tienda",
            r"producto",
            r"checkout",
            r"theme",
            r"liquid",
            r"storefront",
            r"ecommerce",
            r"orden",
        ],
        "audio": [
            r"audio",
            r"mastering",
            r"podcast",
            r"ableton",
            r"eq",
            r"compressor",
            r"limiter",
            r"mezcla",
            r"sonido",
        ],
        "automation": [
            r"automat",
            r"workflow",
            r"n8n",
            r"zapier",
            r"script",
            r"cron",
            r"schedule",
            r"trigger",
            r"webhook",
        ],
        "coding": [
            r"code",
            r"program",
            r"function",
            r"class",
            r"api",
            r"endpoint",
            r"database",
            r"python",
            r"javascript",
            r"typescript",
        ],
        "ai_agent": [
            r"agent",
            r"ai\s*agent",
            r"llm",
            r"gpt",
            r"claude",
            r"gemini",
            r"openai",
            r"anthropic",
            r"model",
            r"prompt\s*eng",
        ],
        "mercadolibre": [
            r"mercado\s*libre",
            r"meli",
            r"ml\b",
            r"publicaci[oó]n",
            r"anuncio",
            r"vendedor",
            r"mercadopago",
        ],
        "documentation": [
            r"readme",
            r"documentation",
            r"manual",
            r"gu[ií]a",
            r"protocolo",
            r"instrucciones",
        ],
    }

    # Prompt type patterns
    TYPE_PATTERNS = {
        "system_prompt": [
            r"^#?\s*system",
            r"eres\s+un",
            r"you\s+are\s+a",
            r"actúa\s+como",
            r"act\s+as",
            r"role:",
            r"instrucciones\s*:",
        ],
        "template": [
            r"\{\{",
            r"\}\}",
            r"\[placeholder\]",
            r"<variable>",
            r"\$\{",
            r"{{.*}}",
            r"plantilla",
        ],
        "protocol": [
            r"protocolo",
            r"protocol",
            r"paso\s*\d",
            r"step\s*\d",
            r"fase\s*\d",
            r"phase\s*\d",
            r"secuencia",
        ],
        "master_prompt": [
            r"master",
            r"maestro",
            r"principal",
            r"main\s*prompt",
            r"root\s*prompt",
            r"base\s*prompt",
        ],
        "user_prompt": [
            r"quiero",
            r"necesito",
            r"help\s*me",
            r"ayúdame",
            r"genera",
            r"crea",
            r"escribe",
        ],
    }

    # Version extraction pattern
    VERSION_PATTERN = re.compile(
        r"v?(\d+\.\d+(?:\.\d+)?)|" r"(vFinal|final|FINAL)|" r"(v\d+)|" r"(_v[\d.]+)",
        re.IGNORECASE,
    )

    # Language detection
    SPANISH_MARKERS = [
        r"\bel\b",
        r"\bla\b",
        r"\blos\b",
        r"\blas\b",
        r"\bde\b",
        r"\bque\b",
        r"\ben\b",
        r"\bpara\b",
        r"\bcon\b",
        r"\bpor\b",
        r"\buna?\b",
        r"\besto\b",
        r"\beste\b",
        r"\besta\b",
    ]

    ENGLISH_MARKERS = [
        r"\bthe\b",
        r"\bis\b",
        r"\bare\b",
        r"\bwith\b",
        r"\bfor\b",
        r"\band\b",
        r"\bthat\b",
        r"\bthis\b",
        r"\bwill\b",
        r"\byou\b",
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self._domain_compiled = {
            domain: [re.compile(p, re.IGNORECASE) for p in patterns]
            for domain, patterns in self.DOMAIN_PATTERNS.items()
        }
        self._type_compiled = {
            ptype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for ptype, patterns in self.TYPE_PATTERNS.items()
        }

    def classify(self, content: str, filename: str = "") -> PromptClassification:
        """
        Classify a prompt based on its content and filename.

        Args:
            content: The prompt text content
            filename: Original filename (optional, improves classification)

        Returns:
            PromptClassification with detected attributes
        """
        combined = f"{filename} {content}"

        domain = self._detect_domain(combined)
        prompt_type = self._detect_type(combined)
        keywords = self._extract_keywords(content)
        version = self._extract_version(filename, content)
        language = self._detect_language(content)
        quality = self._calculate_quality(content, filename)

        return PromptClassification(
            domain=domain,
            prompt_type=prompt_type,
            keywords=keywords,
            version=version,
            language=language,
            quality_score=quality,
        )

    def _detect_domain(self, text: str) -> str:
        """Detect the primary domain of the prompt."""
        scores = {}
        for domain, patterns in self._domain_compiled.items():
            score = sum(1 for p in patterns if p.search(text))
            if score > 0:
                scores[domain] = score

        if not scores:
            return "general"

        return max(scores, key=scores.get)

    def _detect_type(self, text: str) -> str:
        """Detect the prompt type."""
        scores = {}
        for ptype, patterns in self._type_compiled.items():
            score = sum(1 for p in patterns if p.search(text))
            if score > 0:
                scores[ptype] = score

        if not scores:
            return "user_prompt"

        return max(scores, key=scores.get)

    def _extract_keywords(self, content: str, max_keywords: int = 8) -> List[str]:
        """Extract top keywords from content."""
        # Remove common stopwords and punctuation
        stopwords = {
            "el",
            "la",
            "los",
            "las",
            "de",
            "del",
            "en",
            "que",
            "y",
            "a",
            "para",
            "con",
            "un",
            "una",
            "es",
            "por",
            "se",
            "como",
            "su",
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "for",
            "on",
            "is",
            "it",
            "be",
            "this",
            "that",
            "with",
            "as",
            "at",
            "by",
        }

        # Extract words (3+ chars, alphanumeric)
        words = re.findall(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]{3,}\b", content.lower())

        # Count frequencies
        freq = {}
        for word in words:
            if word not in stopwords:
                freq[word] = freq.get(word, 0) + 1

        # Sort by frequency and return top N
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]

    def _extract_version(self, filename: str, content: str) -> Optional[str]:
        """Extract version string from filename or content."""
        # Check filename first
        match = self.VERSION_PATTERN.search(filename)
        if match:
            return match.group(0).strip("_")

        # Check first few lines of content
        first_lines = content[:500]
        match = self.VERSION_PATTERN.search(first_lines)
        if match:
            return match.group(0).strip("_")

        return None

    def _detect_language(self, content: str) -> str:
        """Detect primary language (es/en/mixed)."""
        sample = content[:2000].lower()

        spanish_score = sum(1 for p in self.SPANISH_MARKERS if re.search(p, sample))
        english_score = sum(1 for p in self.ENGLISH_MARKERS if re.search(p, sample))

        if spanish_score > english_score * 1.5:
            return "es"
        elif english_score > spanish_score * 1.5:
            return "en"
        else:
            return "mixed"

    def _calculate_quality(self, content: str, filename: str) -> float:
        """Calculate a quality score for the prompt (0.0 - 1.0)."""
        score = 0.5  # Base score

        # Length bonus (longer prompts tend to be more complete)
        if len(content) > 500:
            score += 0.1
        if len(content) > 2000:
            score += 0.1

        # Structure bonus (has headers, sections)
        if re.search(r"^#+\s", content, re.MULTILINE):
            score += 0.1

        # Version indicator (versioned = likely maintained)
        if self._extract_version(filename, content):
            score += 0.1

        # Not a copy/duplicate
        if not re.search(r"copia\s+de|copy\s+of", filename, re.IGNORECASE):
            score += 0.1

        # Final/production marker
        if re.search(r"final|vfinal|production|prod", filename, re.IGNORECASE):
            score += 0.1

        return min(1.0, score)


# Convenience function
def classify_prompt(content: str, filename: str = "") -> PromptClassification:
    """Classify a prompt using the default classifier."""
    classifier = PromptClassifier()
    return classifier.classify(content, filename)


if __name__ == "__main__":
    # Test classification
    test_content = """
    # Sistema de Cotizaciones BMC

    Eres un asistente de ventas para BMC Uruguay. Tu objetivo es ayudar a los
    clientes a obtener cotizaciones de productos de construcción.

    ## Instrucciones
    1. Saluda al cliente
    2. Pregunta qué productos necesita
    3. Genera una cotización automática
    """

    result = classify_prompt(test_content, "prompt_bmc_cotizaciones_v2.1.txt")
    print(f"Domain: {result.domain}")
    print(f"Type: {result.prompt_type}")
    print(f"Keywords: {result.keywords}")
    print(f"Version: {result.version}")
    print(f"Language: {result.language}")
    print(f"Quality: {result.quality_score:.2f}")
