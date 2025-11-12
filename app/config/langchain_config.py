"""
LangChain configuration for StudyGuru Pro - Optimized Version
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# Import native Google Search tool
try:
    from google.genai.types import Tool, GoogleSearch
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:
    Tool = None
    GoogleSearch = None
    genai = None
    types = None
    GENAI_AVAILABLE = False

from app.config.cache_manager import cache_manager
from app.core.config import settings

# Model mapping for dynamic selection - matches frontend modelService.ts
MODEL_MAPPING = {
    # Gemini models
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    # GPT models
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-5": "gpt-5",
    # Moonshot AI (Kimi) models
    "kimi-k2-thinking": "kimi-k2-thinking",
    "kimi-k2": "kimi-k2-thinking",  # Alias
}


# Subscription tiers
class SubscriptionTier:
    ESSENTIAL = "ESSENTIAL"  # Free tier
    PLUS = "PLUS"
    ELITE = "ELITE"


def get_model_row(model_name: str) -> Optional[int]:
    """
    Determine which row a model belongs to based on model name

    Returns:
        Row number (1, 2, or 3) or None if model not found
    """
    # Extract base model name
    base_model = model_name.lower()

    # Row 1: Default models - gemini-2.5-pro (visualize only), gemini-2.5-flash, gpt-4.1 (visualize only), gpt-4.1-mini, kimi models
    row1_models = {
        "gemini-2.5-pro-visualize": 1,
        "gemini-2.5-flash": 1,
        "gemini-2.5-flash-assistant": 1,
        "gpt-4.1-visualize": 1,
        "gpt-4.1-mini": 1,
        "gpt-4.1-mini-assistant": 1,
        "kimi-k2-thinking": 1,
        "kimi-k2": 1,
        "kimi-k2-visualize": 1,
        "kimi-k2-assistant": 1,
    }

    # Row 2: PLUS models - gemini-2.5-pro (both), gpt-4.1 (both), kimi models
    row2_models = {
        "gemini-2.5-pro-plus-visualize": 2,
        "gemini-2.5-pro-plus-assistant": 2,
        "gemini-2.5-pro": 2,  # When used for both visualize and assistant
        "gpt-4.1-plus-visualize": 2,
        "gpt-4.1-plus-assistant": 2,
        "gpt-4.1": 2,  # When used for both visualize and assistant
        "kimi-k2-thinking": 2,
        "kimi-k2": 2,
        "kimi-k2-visualize": 2,
        "kimi-k2-assistant": 2,
    }

    # Row 3: ELITE models - gpt-5 (both), kimi models
    row3_models = {
        "gpt-5-elite-visualize": 3,
        "gpt-5-elite-assistant": 3,
        "gpt-5": 3,
        "kimi-k2-thinking": 3,
        "kimi-k2": 3,
        "kimi-k2-visualize": 3,
        "kimi-k2-assistant": 3,
    }

    # Check each row
    if base_model in row1_models:
        return row1_models[base_model]
    elif base_model in row2_models:
        return row2_models[base_model]
    elif base_model in row3_models:
        return row3_models[base_model]

    # Fallback: try to detect from base model name
    if (
        "gemini-2.5-flash" in base_model
        or "gpt-4.1-mini" in base_model
        or "kimi" in base_model
    ):
        return 1
    elif "gemini-2.5-pro" in base_model or (
        "gpt-4.1" in base_model and "gpt-5" not in base_model
    ):
        # Could be row 1 or row 2, check context
        if "plus" in base_model or (
            "visualize" not in base_model and "assistant" not in base_model
        ):
            return 2
        return 1
    elif "gpt-5" in base_model:
        return 3

    return None


def validate_model_access(model_name: str, subscription_plan: str) -> bool:
    """
    Validate if user has access to the selected model based on subscription

    Row 1: Available to all (ESSENTIAL, PLUS, ELITE)
    Row 2: Available to ESSENTIAL + PLUS only (not ELITE)
    Row 3: Available to all (ESSENTIAL, PLUS, ELITE)

    Args:
        model_name: The model name (e.g., 'gemini-2.5-pro', 'gpt-4.1', 'gpt-5')
        subscription_plan: User's subscription plan (ESSENTIAL, PLUS, ELITE)

    Returns:
        bool: True if user has access, False otherwise
    """
    # Get the row for this model
    row = get_model_row(model_name)

    if row is None:
        # Unknown model, deny access
        return False

    # Row 1: Available to all subscribers
    if row == 1:
        return True

    # Row 2: Available to ESSENTIAL + PLUS only (not ELITE)
    if row == 2:
        return subscription_plan in [SubscriptionTier.ESSENTIAL, SubscriptionTier.PLUS]

    # Row 3: Available to all subscribers
    if row == 3:
        return True

    # Default: deny access
    return False


class CompatibleEmbeddings(Embeddings):
    """Embeddings wrapper ensuring compatibility between different models"""

    __slots__ = ("base_embeddings", "target_dimension", "source_dimension")

    DIMENSION_MAP = {"GoogleGenerativeAIEmbeddings": 768, "OpenAIEmbeddings": 1536}

    def __init__(self, base_embeddings: Embeddings, target_dimension: int = 1536):
        self.base_embeddings = base_embeddings
        self.target_dimension = target_dimension
        self.source_dimension = self.DIMENSION_MAP.get(
            type(base_embeddings).__name__, 1536
        )

    def _adjust_dimensions(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Pad or truncate embeddings to target dimension"""
        if self.source_dimension == self.target_dimension:
            return embeddings

        if self.source_dimension < self.target_dimension:
            # Pad with zeros
            padding_size = self.target_dimension - self.source_dimension
            return [emb + [0.0] * padding_size for emb in embeddings]
        else:
            # Truncate
            return [emb[: self.source_dimension] for emb in embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.base_embeddings.embed_documents(texts)
        return self._adjust_dimensions(embeddings)

    def embed_query(self, text: str) -> List[float]:
        embedding = self.base_embeddings.embed_query(text)
        return self._adjust_dimensions([embedding])[0]


class MarkdownJsonOutputParser(JsonOutputParser):
    """Custom JSON parser handling markdown code blocks and various formats"""

    # Compile regex patterns once for better performance
    PATTERNS: List[re.Pattern] = [
        re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL),
        re.compile(r"```\s*(\{.*?\})\s*```", re.DOTALL),
        re.compile(r"\{.*\}", re.DOTALL),
        re.compile(r"\[.*\]", re.DOTALL),
    ]

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks"""
        if not text or not text.strip():
            raise ValueError("Empty or whitespace-only text provided")

        text = text.strip()

        # Try direct JSON parsing first (fastest path)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try regex extraction strategies
        for pattern in self.PATTERNS:
            match = pattern.search(text)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try:
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue

        # Last resort: find matching brackets
        for i, char in enumerate(text):
            if char in "{[":
                end_char = "}" if char == "{" else "]"
                bracket_count = 0

                for j in range(i, len(text)):
                    if text[j] == char:
                        bracket_count += 1
                    elif text[j] == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            try:
                                return json.loads(text[i : j + 1])
                            except json.JSONDecodeError:
                                break

        raise ValueError(f"Could not parse JSON. Preview: {text[:200]}...")


class StudyGuruPrompts:
    """Centralized prompt templates for StudyGuru"""

    DOCUMENT_ANALYSIS = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Analyze educational content and respond with valid JSON only.

TYPE DETECTION:
- "mcq": Any numbered/lettered questions (1,2,3 or a,b,c), multiple choice, worksheets, quizzes
- "written": Explanatory text, essays, concept explanations
- "other": Mixed/unclear

RULES:
1. Detect language
2. MCQ: Extract ALL questions separately
   - Add "options" ONLY if A/B/C/D choices exist
   - "answer": option letter OR direct solution
3. Written: Organized explanatory content

JSON FORMAT:

MCQ:
{{"type": "mcq", "language": "lang", "_result": {{"questions": [{{"question": "text", "options": {{"a": "1", "b": "2", "c": "3", "d": "4"}}, "answer": "letter", "explanation": "solution"}}]}}}}

Written:
{{"type": "written", "language": "lang", "_result": {{"content": "text"}}}}

Return valid JSON only.""",
            ),
            (
                "human",
                [
                    {"type": "text", "text": "Please analyze this document/image:"},
                    {"type": "image_url", "image_url": {"url": "{file_url}"}},
                ],
            ),
        ]
    )

    GUARDRAIL_CHECK = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Check if content is educational.

ACCEPT: Textbooks, worksheets, math, notes, quizzes, exams, homework, diagrams, equations, scientific content. Educational content with faces = ACCEPT. Empty/minimal = ACCEPT.
Simple greetings,  Conversation starters (eg: hi, how are you) = ACCEPT.
REJECT: Selfies, social media, inappropriate material, personal photos.

JSON only:
Accept: {{"is_violation": false, "violation_type": null, "reasoning": "[brief]"}}
Reject: {{"is_violation": true, "violation_type": "non_educational_content", "reasoning": "[why]"}}""",
            ),
            ("human", "{content}"),
        ]
    )

    CONVERSATION_WITH_CONTEXT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """StudyGuru AI - Educational assistant with learning history access.

Topic: {interaction_title}
Context: {interaction_summary}

Use context to personalize. Reference past discussions. Answer specific question numbers from context.

FORMAT: Plain text, ### headers, number questions (1., 2.), options (A., B., C., D.) if present, Answer: [solution], Explanation: [text], • for lists.""",
            ),
            ("human", "{content}"),
        ]
    )

    CONVERSATION_WITHOUT_CONTEXT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """StudyGuru AI - Advanced educational assistant.

FORMAT: ### headers, number questions (1., 2.), options (A., B., C., D.) if exist else direct solution, Answer: [solution] (no bold), Explanation: [text] (no bold), • for lists, plain text math (e.g., "x squared"), clear paragraphs.

Be encouraging, professional, focused on learning.""",
            ),
            ("human", "{content}"),
        ]
    )

    TITLE_GENERATION = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Generate appropriate titles for conversations.

Rules: 
- Short title (max 50 chars), summary (max 100 chars)
- Simple language, no special chars
- For greetings/casual messages: use "Chat with StudyGuru" or similar
- For educational questions: focus on the subject/topic
- For general questions: use the main topic discussed

JSON: {{"title": "conversation title", "summary_title": "brief description"}}""",
            ),
            ("human", "User: {message}\nResponse: {response_preview}"),
        ]
    )

    CONVERSATION_SUMMARIZATION = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Extract key facts from educational conversation.

Extract: key facts, main concepts, problems solved, formulas/equations, question numbers, learning progress, areas needing help.

CRITICAL: Valid JSON only. NO markdown blocks. Plain text, no LaTeX. Escaped strings. Default values if empty.

JSON:
{{
    "key_facts": ["fact 1: info with context", "fact 2: concept with details"],
    "main_topics": ["topic1", "topic2"],
    "semantic_summary": "Detailed 3-4 sentence summary (min 50 chars)",
    "important_terms": ["term1", "term2"],
    "context_for_future": "Context for follow-ups",
    "question_numbers": [1, 2, 3],
    "learning_progress": "What user learned",
    "potential_follow_ups": ["question 1", "question 2"],
    "difficulty_level": "beginner|intermediate|advanced",
    "subject_area": "math|science|language|other"
}}""",
            ),
            ("human", "User: {user_message}\n\nAI: {ai_response}"),
        ]
    )

    INTERACTION_SUMMARY_UPDATE = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Maintain running summary of educational conversation.

Update summary: incorporate new facts/topics, keep important context, remove redundant info, prioritize recent/relevant, under 500 words, track progress, identify patterns.

Valid JSON only. NO markdown blocks.

{{
    "updated_summary": "Comprehensive running summary",
    "key_topics": ["topics covered"],
    "recent_focus": "Recent focus (last 2-3 exchanges)",
    "accumulated_facts": ["critical facts for future"],
    "question_numbers": [1, 2, 3],
    "learning_progression": "How understanding evolved",
    "difficulty_trend": "beginner|intermediate|advanced",
    "learning_patterns": ["pattern1", "pattern2"],
    "struggling_areas": ["area1", "area2"],
    "mastered_concepts": ["concept1", "concept2"]
}}""",
            ),
            (
                "human",
                "Current: {current_summary}\n\nNew user: {new_user_message}\n\nNew AI: {new_ai_response}",
            ),
        ]
    )

    MCQ_GENERATION = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """StudyGuru AI - MCQ expert.

Create 5-10 high-quality MCQs: test understanding not memorization, clear questions, exactly 4 options (a,b,c,d), one correct answer, plausible distractors, educational explanations, proper notation, varied difficulty.

Valid JSON only:
{{
    "type": "mcq",
    "language": "English",
    "title": "Topic – Multiple Choice Questions",
    "summary_title": "Solved MCQs with answers",
    "_result": {{
        "questions": [
            {{
                "question": "Question?",
                "options": {{"a": "opt1", "b": "opt2", "c": "opt3", "d": "opt4"}},
                "answer": "correct letter (a/b/c/d)",
                "explanation": "Why correct, why others wrong"
            }}
        ]
    }}
}}""",
            ),
            ("human", "Generate MCQs for: {topic_or_content}"),
        ]
    )

    WEB_SEARCH_CONVERSATION = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """StudyGuru AI - Educational assistant with web search capabilities.

You have access to Google Search to find current information, verify facts, and provide up-to-date educational content.

FORMAT: ### headers, number questions (1., 2.), options (A., B., C., D.) if exist else direct solution, Answer: [solution] (no bold), Explanation: [text] (no bold), • for lists, plain text math (e.g., "x squared"), clear paragraphs.

When using web search:
- Search for current information when needed
- Verify facts and provide accurate, up-to-date information
- Cite sources when referencing web search results
- Be encouraging, professional, focused on learning

Use web search to enhance your educational responses with current, accurate information.""",
            ),
            ("human", "{question}"),
        ]
    )


class StudyGuruModels:
    """Model configurations - supports GPT, Gemini, and Moonshot AI (Kimi)"""

    USE_FALLBACK_MODELS = False

    @staticmethod
    @lru_cache(maxsize=1)
    def _is_gemini_model() -> bool:
        return settings.LLM_MODEL.lower() == "gemini"

    @staticmethod
    def _get_cache():
        return cache_manager.get_response_cache()

    @staticmethod
    def get_chat_model(
        temperature=0.2,
        max_tokens=5000,
        reasoning_effort="low",
        verbosity="low",
        streaming=True,
        web_search=False,
        thinking_config=None,
        model_name=None,
        subscription_plan=None,
    ):
        """Get chat model with optional web search capability and thinking config

        Args:
            model_name: Specific model to use (e.g., 'gemini-2.5-pro', 'gpt-4.1', 'gpt-5', 'kimi-k2-thinking')
                       Can also include type suffix like 'gemini-2.5-pro-assistant' or 'kimi-k2-assistant'
            subscription_plan: User's subscription plan for access validation
        """
        cache = StudyGuruModels._get_cache()

        # Clean model name (remove type suffixes for validation and mapping)
        clean_model_name = model_name
        if model_name:
            # Remove type suffixes for validation
            clean_model_name = (
                model_name.replace("-visualize", "")
                .replace("-assistant", "")
                .replace("-plus", "")
                .replace("-elite", "")
            )

        # Validate model access if subscription plan provided
        if model_name and subscription_plan:
            if not validate_model_access(model_name, subscription_plan):
                raise ValueError(
                    f"Access denied: User with {subscription_plan} subscription does not have access to model: {model_name}"
                )

        # Determine the model to use
        if model_name:
            # Use specified model - map to actual model name
            actual_model = MODEL_MAPPING.get(clean_model_name, clean_model_name)
            is_gemini = actual_model.startswith("gemini")
            is_kimi = actual_model.startswith("kimi")
        else:
            # Fall back to default behavior
            is_gemini = StudyGuruModels._is_gemini_model()
            is_kimi = False
            actual_model = (
                "gemini-2.5-pro"
                if is_gemini
                else ("gpt-4.1" if StudyGuruModels.USE_FALLBACK_MODELS else "gpt-5")
            )

        # Moonshot AI (Kimi) models - use OpenAI-compatible API
        if is_kimi or (model_name and actual_model.startswith("kimi")):
            if not settings.MOONSHOT_API_KEY:
                raise ValueError(
                    "MOONSHOT_API_KEY is required for Kimi models. Please set it in your environment variables."
                )

            # Apply thinking config for Kimi models
            model_kwargs = {}
            if thinking_config:
                # Moonshot AI supports similar parameters to OpenAI
                if "reasoning_effort" in thinking_config:
                    model_kwargs["reasoning_effort"] = thinking_config[
                        "reasoning_effort"
                    ]

            return ChatOpenAI(
                model=actual_model,
                temperature=temperature,
                openai_api_key=settings.MOONSHOT_API_KEY,
                base_url="https://api.moonshot.ai/v1",
                max_tokens=max_tokens,
                request_timeout=120,
                streaming=streaming,
                cache=cache,
                **model_kwargs,
            )

        if is_gemini or (model_name and actual_model.startswith("gemini")):
            # Prepare tools list for web search
            tools = []
            if web_search and Tool is not None and GoogleSearch is not None:
                grounding_tool = Tool(google_search=GoogleSearch())
                tools = [grounding_tool]

            # Apply thinking config if provided
            model_kwargs = {}
            if thinking_config and GENAI_AVAILABLE:
                model_kwargs.update(thinking_config)

            return ChatGoogleGenerativeAI(
                model=actual_model,
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=120,
                cache=cache,
                tools=tools,  # Include web search tool if enabled
                **model_kwargs,
            )

        # GPT models
        # Apply thinking config for GPT models
        model_kwargs = {}
        if thinking_config:
            # For GPT models, use reasoning_effort parameter
            if "reasoning_effort" in thinking_config:
                model_kwargs["reasoning_effort"] = thinking_config["reasoning_effort"]

        return ChatOpenAI(
            model=actual_model,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
            request_timeout=120,
            streaming=streaming,
            cache=cache,
            **model_kwargs,
        )

    @staticmethod
    def get_vision_model(
        temperature: float = 0.3,
        max_tokens: int = 5000,
        verbosity: str = "low",
        streaming: bool = True,
        web_search: bool = False,
        thinking_config=None,
        model_name=None,
        subscription_plan=None,
    ):
        """Get configured vision model with optional web search capability and thinking config - supports GPT, Gemini, and Moonshot AI (Kimi)

        Args:
            model_name: Specific model to use (e.g., 'gemini-2.5-pro', 'gpt-4.1', 'gpt-5', 'kimi-k2-thinking')
                       Can also include type suffix like 'gpt-4.1-visualize' or 'kimi-k2-visualize'
            subscription_plan: User's subscription plan for access validation
        """
        # Clean model name (remove type suffixes for validation and mapping)
        clean_model_name = model_name
        if model_name:
            # Remove type suffixes for validation
            clean_model_name = (
                model_name.replace("-visualize", "")
                .replace("-assistant", "")
                .replace("-plus", "")
                .replace("-elite", "")
            )

        # Validate model access if subscription plan provided
        if model_name and subscription_plan:
            if not validate_model_access(model_name, subscription_plan):
                raise ValueError(
                    f"Access denied: User with {subscription_plan} subscription does not have access to model: {model_name}"
                )

        # Determine the model to use
        if model_name:
            actual_model = MODEL_MAPPING.get(clean_model_name, clean_model_name)
            is_gemini = actual_model.startswith("gemini")
            is_kimi = actual_model.startswith("kimi")
        else:
            is_gemini = StudyGuruModels._is_gemini_model()
            is_kimi = False
            actual_model = (
                "gemini-2.5-pro"
                if is_gemini
                else ("gpt-4.1" if StudyGuruModels.USE_FALLBACK_MODELS else "gpt-5")
            )

        # Moonshot AI (Kimi) models - use OpenAI-compatible API
        if is_kimi or (model_name and actual_model.startswith("kimi")):
            if not settings.MOONSHOT_API_KEY:
                raise ValueError(
                    "MOONSHOT_API_KEY is required for Kimi models. Please set it in your environment variables."
                )

            # Apply thinking config for Kimi models
            model_kwargs = {}
            if thinking_config:
                # Moonshot AI supports similar parameters to OpenAI
                if "reasoning_effort" in thinking_config:
                    model_kwargs["reasoning_effort"] = thinking_config[
                        "reasoning_effort"
                    ]

            return ChatOpenAI(
                model=actual_model,
                temperature=temperature,
                openai_api_key=settings.MOONSHOT_API_KEY,
                base_url="https://api.moonshot.ai/v1",
                max_tokens=max_tokens,
                request_timeout=120,
                streaming=streaming,
                cache=cache_manager.get_response_cache(),  # Enable response caching
                **model_kwargs,
            )

        if is_gemini or (model_name and actual_model.startswith("gemini")):
            # Prepare tools list for web search
            tools = []
            if web_search and Tool is not None and GoogleSearch is not None:
                grounding_tool = Tool(google_search=GoogleSearch())
                tools = [grounding_tool]

            # Apply thinking config if provided
            model_kwargs = {}
            if thinking_config and GENAI_AVAILABLE:
                model_kwargs.update(thinking_config)

            # Gemini with vision capabilities
            return ChatGoogleGenerativeAI(
                model=actual_model,
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=120,
                cache=cache_manager.get_response_cache(),  # Enable response caching
                tools=tools,  # Include web search tool if enabled
                **model_kwargs,
            )
        else:
            # Apply thinking config for GPT models
            model_kwargs = {}
            if thinking_config:
                # For GPT models, use reasoning_effort parameter
                if "reasoning_effort" in thinking_config:
                    model_kwargs["reasoning_effort"] = thinking_config[
                        "reasoning_effort"
                    ]

            # GPT models
            return ChatOpenAI(
                model=actual_model,
                temperature=temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=max_tokens,
                request_timeout=120,
                streaming=streaming,
                cache=cache_manager.get_response_cache(),  # Enable response caching
                **model_kwargs,
            )

    @staticmethod
    def get_guardrail_model(
        temperature: float = 0.1, max_tokens: int = 500, verbosity: str = "low"
    ):
        """Get configured guardrail model - supports both GPT and Gemini"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Flash for cost efficiency (equivalent to GPT-4o-mini)
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # Fast and cost-effective model
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=15,
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # GPT models
            if StudyGuruModels.USE_FALLBACK_MODELS:
                # Fallback to GPT-4o-mini for compatibility
                return ChatOpenAI(
                    model="gpt-4.1-mini",  # Fallback model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=15,
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )
            else:
                # GPT-5 Mini configuration
                return ChatOpenAI(
                    model="gpt-5-mini",  # GPT-5 Mini: 83% more cost-effective than GPT-5
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=15,  # Fast timeout for guardrails
                    # verbosity=verbosity,  # Control response length and detail
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )

    @staticmethod
    def get_complex_reasoning_model(
        temperature: float = 0.1, max_tokens: int = 5000, verbosity: str = "medium"
    ):
        """Get configured model for complex reasoning tasks - supports both GPT and Gemini"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Pro for complex reasoning
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",  # Latest Gemini for complex tasks
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=120,
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # GPT models
            if StudyGuruModels.USE_FALLBACK_MODELS:
                # Fallback to GPT-4o for compatibility
                return ChatOpenAI(
                    model="gpt-4.1",  # Fallback model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=120,
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )
            else:
                # GPT-5 configuration
                return ChatOpenAI(
                    model="gpt-5",  # GPT-5 for complex reasoning
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=150,  # Increased timeout for complex reasoning with high effort
                    # verbosity=verbosity,  # Medium verbosity for complex reasoning tasks
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )

    @staticmethod
    @lru_cache(maxsize=1)
    def get_embeddings_model():
        """Get embeddings model with caching"""
        if StudyGuruModels._is_gemini_model():
            base = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=settings.GOOGLE_API_KEY
            )
        else:
            base = OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
            )

        return CompatibleEmbeddings(base, target_dimension=1536)

    @staticmethod
    def get_title_model(
        temperature: float = 0.3, max_tokens: int = 100, verbosity: str = "low"
    ):
        """Get configured title generation model - supports both GPT and Gemini"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Flash for cost efficiency
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # Fast and cost-effective model
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=10,  # Fast timeout for quick response
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # GPT models
            if StudyGuruModels.USE_FALLBACK_MODELS:
                # Fallback to GPT-4o-mini for compatibility
                return ChatOpenAI(
                    model="gpt-4.1-mini",  # Fallback model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,  # Very low token limit for cost efficiency
                    request_timeout=10,  # Fast timeout for quick response
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )
            else:
                # GPT-5 Mini configuration
                return ChatOpenAI(
                    model="gpt-5-mini",  # GPT-5 Mini: Most cost-effective model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,  # Very low token limit for cost efficiency
                    request_timeout=10,  # Fast timeout for quick response
                    # verbosity=verbosity,  # Control response length and detail
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )

    @staticmethod
    def get_web_search_model(
        temperature: float = 0.2, max_tokens: int = 5000, streaming: bool = True
    ):
        """Get configured model with Google Search tool integration - Gemini only"""
        if not StudyGuruModels._is_gemini_model():
            raise ValueError(
                "Web search functionality is only available with Gemini models"
            )

        # Define the native grounding tool
        grounding_tool = Tool(google_search=GoogleSearch())

        # Initialize the model and pass it the tool
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
            max_output_tokens=max_tokens,
            request_timeout=120,
            cache=cache_manager.get_response_cache(),
            tools=[grounding_tool],  # This enables native "AUTO WEBSEARCH"
            streaming=streaming,
        )

    @staticmethod
    def get_model_with_context_cache(
        model_type: str = "chat",
        temperature: float = 0.2,
        max_tokens: int = 5000,
        cached_content: Optional[Any] = None,
    ):
        """
        Get model with context caching for large documents

        Args:
            model_type: Type of model ("chat", "vision", "guardrail", "reasoning", "title")
            temperature: Model temperature
            max_tokens: Maximum output tokens
            cached_content: Pre-cached content for context caching

        Returns:
            Model instance with context caching enabled
        """
        if StudyGuruModels._is_gemini_model() and cached_content:
            # Use context caching for Gemini models
            if model_type == "chat":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )
            elif model_type == "vision":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    request_timeout=120,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )
            elif model_type == "guardrail":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    request_timeout=15,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )
            elif model_type == "reasoning":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    request_timeout=120,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )
            elif model_type == "title":
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_output_tokens=max_tokens,
                    request_timeout=10,
                    cache=cache_manager.get_response_cache(),
                    cache_context=cached_content,  # Enable context caching
                )

        # Fallback to regular model without context caching
        if model_type == "chat":
            return StudyGuruModels.get_chat_model(temperature, max_tokens)
        elif model_type == "vision":
            return StudyGuruModels.get_vision_model(temperature, max_tokens)
        elif model_type == "guardrail":
            return StudyGuruModels.get_guardrail_model(temperature, max_tokens)
        elif model_type == "reasoning":
            return StudyGuruModels.get_complex_reasoning_model(temperature, max_tokens)
        elif model_type == "title":
            return StudyGuruModels.get_title_model(temperature, max_tokens)
        else:
            return StudyGuruModels.get_chat_model(temperature, max_tokens)


class StudyGuruVectorStore:
    """Vector store configuration for StudyGuru"""

    @staticmethod
    def get_milvus_config() -> Dict[str, Any]:
        """Get Milvus connection configuration"""
        return {
            "uri": settings.ZILLIZ_URI,
            "token": settings.ZILLIZ_TOKEN,
            "secure": True,
        }

    @staticmethod
    def get_collection_config() -> Dict[str, Any]:
        """Get collection configuration - uses single collection with common dimension"""
        # Use single collection with 1536 dimensions (largest) for compatibility
        # Gemini embeddings (768D) will be padded to 1536D
        dimension = 1536  # Common dimension for both models
        collection_name = settings.ZILLIZ_COLLECTION  # Single collection

        return {
            "collection_name": collection_name,
            "dimension": dimension,
            "index_params": {
                "index_type": "IVF_FLAT",
                "metric_type": settings.ZILLIZ_INDEX_METRIC,
                "params": {"nlist": 1024},
            },
        }


class StudyGuruChains:
    """
    Pre-configured chains for StudyGuru operations

    Parser Strategy:
    - MarkdownJsonOutputParser: Used for all chains that might receive mixed content
      (AI responses with explanations + JSON, markdown code blocks, etc.)
    - JsonOutputParser: Reserved for simple, pure JSON responses (currently unused)

    The MarkdownJsonOutputParser is more robust and can handle:
    - Pure JSON responses
    - JSON wrapped in markdown code blocks
    - Mixed content with JSON embedded in text
    - Various formatting issues
    """

    @staticmethod
    def get_document_analysis_chain():
        """Get document analysis chain using GPT-5 with high reasoning effort for better analysis"""
        model = StudyGuruModels.get_vision_model()
        parser = MarkdownJsonOutputParser()  # Use robust parser for document analysis
        return StudyGuruPrompts.DOCUMENT_ANALYSIS | model | parser

    @staticmethod
    def get_guardrail_chain():
        """Get guardrail check chain"""
        model = StudyGuruModels.get_guardrail_model(temperature=0.2, max_tokens=400)
        parser = MarkdownJsonOutputParser()  # Use robust parser for guardrails
        return StudyGuruPrompts.GUARDRAIL_CHECK | model | parser

    @staticmethod
    def get_conversation_chain(has_context: bool = False):
        """Get conversation chain"""
        model = StudyGuruModels.get_chat_model()
        parser = StrOutputParser()

        if has_context:
            return StudyGuruPrompts.CONVERSATION_WITH_CONTEXT | model | parser
        else:
            return StudyGuruPrompts.CONVERSATION_WITHOUT_CONTEXT | model | parser

    @staticmethod
    def get_title_generation_chain():
        """Get title generation chain (cost-optimized) with robust error handling"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Flash for cost efficiency
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # Fast and cost-effective model
                temperature=0.3,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=300,  # Increased to ensure complete JSON response
                request_timeout=20,  # Increased timeout
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # Use GPT-4o-mini for better JSON stability instead of GPT-5-mini
            # GPT-5 models have issues with response_format parameter
            model = ChatOpenAI(
                model="gpt-4.1-mini",  # More reliable for JSON output
                temperature=0.3,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=300,  # Increased to ensure complete JSON response
                request_timeout=20,  # Increased timeout
                # model_kwargs={"response_format": {"type": "json_object"}},
            )
        parser = MarkdownJsonOutputParser()  # Use robust parser for title generation
        return StudyGuruPrompts.TITLE_GENERATION | model | parser

    @staticmethod
    def get_conversation_summarization_chain():
        """Get conversation summarization chain with increased token limits"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Flash for cost efficiency
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # Fast and cost-effective model
                temperature=0.2,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=2000,  # Aggressively increased to handle reasoning + output
                request_timeout=45,  # Increased timeout for longer processing
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # Use GPT-4o-mini for better stability and disable reasoning to reserve tokens for output
            model = ChatOpenAI(
                model="gpt-4.1-mini",  # More reliable for JSON output
                temperature=0.2,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=2000,  # Aggressively increased to handle reasoning + output
                request_timeout=45,  # Increased timeout for longer processing
                # model_kwargs={"response_format": {"type": "json_object"}},
            )
        parser = MarkdownJsonOutputParser()
        return StudyGuruPrompts.CONVERSATION_SUMMARIZATION | model | parser

    @staticmethod
    def get_interaction_summary_update_chain():
        """Get interaction summary update chain with increased token limits"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Flash for cost efficiency
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # Fast and cost-effective model
                temperature=0.2,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=1500,  # Aggressively increased to handle longer updates
                request_timeout=45,  # Increased timeout
                cache=cache_manager.get_response_cache(),  # Enable response caching
            )
        else:
            # Use GPT-4o-mini for better stability and increased token limits
            model = ChatOpenAI(
                model="gpt-4.1-mini",  # More reliable for JSON output
                temperature=0.2,
                openai_api_key=settings.OPENAI_API_KEY,
                max_tokens=1500,  # Aggressively increased to handle longer updates
                request_timeout=45,  # Increased timeout
                # model_kwargs={"response_format": {"type": "json_object"}},
            )
        parser = MarkdownJsonOutputParser()
        return StudyGuruPrompts.INTERACTION_SUMMARY_UPDATE | model | parser

    @staticmethod
    def get_mcq_generation_chain():
        """Get MCQ generation chain using complex reasoning model for better question quality"""
        model = StudyGuruModels.get_complex_reasoning_model(
            temperature=0.3, max_tokens=1200
        )
        parser = MarkdownJsonOutputParser()  # Use robust parser for MCQ generation
        return StudyGuruPrompts.MCQ_GENERATION | model | parser

    @staticmethod
    def get_web_search_chain():
        """Get web search conversation chain with Google Search tool integration"""
        model = StudyGuruModels.get_web_search_model()
        parser = StrOutputParser()
        return StudyGuruPrompts.WEB_SEARCH_CONVERSATION | model | parser


class StudyGuruConfig:
    """Main configuration class for StudyGuru LangChain setup"""

    # Model configurations
    MODELS = StudyGuruModels

    # Prompt templates
    PROMPTS = StudyGuruPrompts

    # Vector store configurations
    VECTOR_STORE = StudyGuruVectorStore

    # Pre-configured chains
    CHAINS = StudyGuruChains

    # Token system configuration
    BASE_TOKENS = 5000  # Base tokens for text-only prompts
    TOKENS_PER_FILE = 5000  # Additional tokens per file
    MAX_TOKENS_LIMIT = (
        30000  # Maximum token limit to prevent excessive usage (increased from 20000)
    )

    # Default settings
    DEFAULT_MAX_TOKENS = 5000
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_TOP_K = 5

    # Points calculation
    POINTS_PER_TOKEN = 100

    @staticmethod
    def calculate_dynamic_tokens(file_count: int = 0) -> int:
        """
        Calculate dynamic token limit based on file count.

        Args:
            file_count: Number of files being processed

        Returns:
            int: Calculated token limit (BASE_TOKENS + TOKENS_PER_FILE * file_count)
        """
        calculated_tokens = StudyGuruConfig.BASE_TOKENS + (
            StudyGuruConfig.TOKENS_PER_FILE * file_count
        )
        return min(calculated_tokens, StudyGuruConfig.MAX_TOKENS_LIMIT)

    @staticmethod
    def calculate_points_cost(tokens_used: int) -> int:
        """Calculate points cost based on tokens used"""
        return max(1, tokens_used // StudyGuruConfig.POINTS_PER_TOKEN)

    @staticmethod
    def get_retrieval_config(user_id: str, top_k: int = None) -> Dict[str, Any]:
        """Get retrieval configuration for a specific user"""
        return {
            "k": top_k or StudyGuruConfig.DEFAULT_TOP_K,
            "filter": f"user_id == '{user_id}'",
        }
