"""
Text utility functions for common text processing tasks
"""

import re
from typing import List


def extract_question_numbers(message: str) -> List[int]:
    """
    Extract question numbers from text message

    Args:
        message: Text containing potential question number references

    Returns:
        List of unique question numbers found in the message
    """
    question_numbers = []
    patterns = [
        r"question\s+(\d+)",
        r"problem\s+(\d+)",
        r"mcq\s+(\d+)",
        r"equation\s+(\d+)",
        r"exercise\s+(\d+)",
        r"task\s+(\d+)",
        r"(\d+)\.\s*$",  # Number at end of line
        r"explain\s+(?:the\s+)?(?:equation|question|problem|mcq)\s+(\d+)",
        r"what\s+is\s+(?:the\s+)?(?:equation|question|problem|mcq)\s+(\d+)",
        r"solve\s+(?:the\s+)?(?:equation|question|problem|mcq)\s+(\d+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, message, re.IGNORECASE)
        for match in matches:
            try:
                num = int(match)
                if num not in question_numbers:
                    question_numbers.append(num)
            except ValueError:
                continue

    return sorted(question_numbers)


def clean_text(text: str) -> str:
    """
    Clean and normalize text

    Args:
        text: Text to clean

    Returns:
        Cleaned text with normalized whitespace and printable characters
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Remove any non-printable characters
    text = "".join(char for char in text if char.isprintable() or char.isspace())

    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix

    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def extract_key_concepts(text: str, max_concepts: int = 10) -> List[str]:
    """
    Extract key concepts from text using pattern matching

    Args:
        text: Text to analyze
        max_concepts: Maximum number of concepts to return

    Returns:
        List of key concepts
    """
    concepts = set()

    # Pattern matching for key concepts
    concept_patterns = [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Capitalized terms
        r"\b\w+ing\b",  # -ing words
        r"\b\w+tion\b",  # -tion words
    ]

    for pattern in concept_patterns:
        matches = re.findall(pattern, text)
        concepts.update(matches[:5])  # Limit to 5 per pattern

    return list(concepts)[:max_concepts]


def detect_content_type(text: str) -> str:
    """
    Detect the type of educational content

    Args:
        text: Text to analyze

    Returns:
        Content type: 'mcq', 'equation', 'explanation', 'problem', or 'general'
    """
    text_lower = text.lower()

    if any(
        keyword in text_lower
        for keyword in ["mcq", "multiple choice", "a)", "b)", "c)", "d)"]
    ):
        return "mcq"
    elif any(
        keyword in text_lower
        for keyword in ["equation", "formula", "solve", "calculate"]
    ):
        return "equation"
    elif any(
        keyword in text_lower
        for keyword in ["explain", "describe", "what is", "how does"]
    ):
        return "explanation"
    elif any(keyword in text_lower for keyword in ["problem", "question", "exercise"]):
        return "problem"
    else:
        return "general"


def extract_topic_tags(text: str, title: str = "") -> List[str]:
    """
    Extract topic tags from text content

    Args:
        text: Text to analyze
        title: Optional title for additional context

    Returns:
        List of topic tags
    """
    topics = set()

    topic_keywords = {
        "mathematics": [
            "math",
            "algebra",
            "geometry",
            "calculus",
            "trigonometry",
            "equation",
            "formula",
        ],
        "science": [
            "physics",
            "chemistry",
            "biology",
            "experiment",
            "hypothesis",
            "theory",
        ],
        "programming": [
            "code",
            "programming",
            "algorithm",
            "function",
            "variable",
            "loop",
        ],
        "language": [
            "grammar",
            "vocabulary",
            "sentence",
            "paragraph",
            "essay",
            "writing",
        ],
        "history": [
            "historical",
            "century",
            "war",
            "revolution",
            "ancient",
            "medieval",
        ],
    }

    text_lower = (text + " " + title).lower()
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.add(topic)

    return list(topics)


def determine_difficulty_level(text: str) -> str:
    """
    Determine the difficulty level of content

    Args:
        text: Text to analyze

    Returns:
        Difficulty level: 'beginner', 'intermediate', or 'advanced'
    """
    text_lower = text.lower()

    if any(
        keyword in text_lower for keyword in ["basic", "simple", "easy", "beginner"]
    ):
        return "beginner"
    elif any(
        keyword in text_lower
        for keyword in ["advanced", "complex", "difficult", "expert"]
    ):
        return "advanced"
    else:
        return "intermediate"
