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

from app.config.cache_manager import cache_manager
from app.core.config import settings


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
REJECT: Selfies, social media, inappropriate material, personal photos.

JSON only:
Accept: {{"is_violation": false, "violation_type": null, "reasoning": "Educational: [brief]"}}
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
                """Generate titles for educational content.

Rules: Short title (max 50 chars), summary (max 100 chars), simple language, no special chars, focus on subject.

JSON: {{"title": "short topic", "summary_title": "help provided"}}""",
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


class StudyGuruModels:
    """Model configurations - supports GPT and Gemini"""

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
    ):
        """Get chat model"""
        cache = StudyGuruModels._get_cache()

        if StudyGuruModels._is_gemini_model():
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_output_tokens=max_tokens,
                request_timeout=120,
                cache=cache,
            )

        model_name = "gpt-4o" if StudyGuruModels.USE_FALLBACK_MODELS else "gpt-5"
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
            request_timeout=120,
            streaming=streaming,
            cache=cache,
        )

    @staticmethod
    def get_vision_model(
        temperature: float = 0.3,
        max_tokens: int = 5000,
        verbosity: str = "low",
        streaming: bool = True,
    ):
        """Get configured vision model - supports both GPT and Gemini"""
        if StudyGuruModels._is_gemini_model():
            # Gemini 2.5 Pro with vision capabilities
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",  # Gemini with vision support
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
                    model="gpt-4o",  # Fallback model
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=120,
                    streaming=streaming,
                    cache=cache_manager.get_response_cache(),  # Enable response caching
                )
            else:
                # GPT-5 configuration - optimized for speed
                return ChatOpenAI(
                    model="gpt-5",  # GPT-5 with enhanced vision capabilities
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY,
                    max_tokens=max_tokens,
                    request_timeout=120,  # Reduced timeout for faster processing
                    streaming=streaming,
                    # verbosity="low",  # Low verbosity for faster responses
                    cache=cache_manager.get_response_cache(),  # Enable response caching
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
                    model="gpt-4o-mini",  # Fallback model
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
                    model="gpt-4o",  # Fallback model
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
                    model="gpt-4o-mini",  # Fallback model
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
        # Use GPT-4o-mini for better JSON stability instead of GPT-5-mini
        # GPT-5 models have issues with response_format parameter
        model = ChatOpenAI(
            model="gpt-4o-mini",  # More reliable for JSON output
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
        # Use GPT-4o-mini for better stability and disable reasoning to reserve tokens for output
        model = ChatOpenAI(
            model="gpt-4o-mini",  # More reliable for JSON output
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
        # Use GPT-4o-mini for better stability and increased token limits
        model = ChatOpenAI(
            model="gpt-4o-mini",  # More reliable for JSON output
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
    MAX_TOKENS_LIMIT = 20000  # Maximum token limit to prevent excessive usage

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
