"""
LangChain configuration for StudyGuru Pro
"""

import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings


class StudyGuruPrompts:
    """Centralized prompt templates for StudyGuru"""

    DOCUMENT_ANALYSIS = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are StudyGuru AI analyzing educational content. Analyze the given image/document and provide a structured response:

        CONTENT TYPE DETECTION:
        - "mcq": If you see ANY of these patterns:
          * Questions with lettered sub-parts (a), b), c), d), etc.)
          * Numbered exercises with multiple parts
          * Multiple choice questions with A, B, C, D options
          * Practice problems with sub-questions
          * Exercises that can be broken into separate solvable problems
          * Exercise sheets with multiple problems to solve
          
        - "written": Only if it's pure explanatory text, essays, or single concept explanations
        - "other": For mixed content or unclear format

        IMPORTANT: Most educational exercises with sub-parts should be classified as "mcq" type, even if they don't have traditional A/B/C/D choices.

        1. First, detect the language of the content
        2. Carefully identify the content type using the criteria above
        3. Provide a short, descriptive title for the page/content  
        4. Provide a summary title that describes what you will help the user with
        5. Based on the question type:
           - If MCQ: Extract each question/exercise part as a separate question
           - If written: Provide organized explanatory content

        Respond in the detected language and format your response as JSON with this structure:
        {{
            "type": "mcq" or "written" or "other",
            "language": "detected language",
            "title": "short descriptive title for the content",
            "summary_title": "summary of how you will help the user",
            "_result": {{
                // For MCQ type - extract each sub-part as a question:
                "questions": [
                    {{
                        "question": "question text (e.g., 'Solve |1 - 2x| = 3')",
                        "options": {{"a": "option1", "b": "option2", "c": "option3", "d": "option4"}},
                        "answer": "correct option letter or 'N/A' if no options given",
                        "explanation": "step-by-step solution or brief explanation"
                    }}
                ]
                // For written type:
                "content": "organized explanatory text as you would provide in a chat response"
            }}
        }}
        """,
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
                """
        You are a strict content guardrail for an educational platform. Review all user inputs—including text and any attached images—and determine whether the request violates any of the following rules:

        STRICT VIOLATION RULES:
        1. REJECT any images of people's faces, portraits, selfies, or photographs of individuals (even if they appear to be studying)
        2. REJECT content related to adult, explicit, or inappropriate material
        3. REJECT requests for direct code generation (e.g., "write a Java function"), except when analyzing educational code problems
        4. REJECT any content that is NOT educational, study, or research-related
        5. ACCEPT ONLY: textbooks, notes, diagrams, charts, educational worksheets, practice problems, study materials, equations, or handwritten notes WITHOUT faces

        IMPORTANT FOR IMAGES:
        - If an image contains a human face or portrait: REJECT with violation_type "non_educational_content"
        - If an image is a selfie or photo of a person: REJECT with violation_type "non_educational_content"  
        - If an image is clearly educational content (textbook pages, notes, diagrams): ACCEPT

        You must respond with valid JSON in this exact format:
        {{
            "is_violation": boolean,
            "violation_type": "non_educational_content" | "inappropriate_content" | "code_generation" | null,
            "reasoning": "clear explanation of why this was rejected or accepted"
        }}
        """,
            ),
            ("human", "{content}"),
        ]
    )

    CONVERSATION_WITH_CONTEXT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are StudyGuru AI, an advanced educational assistant. Format your responses for clear readability and educational impact.

Current conversation topic: {interaction_title}
Context summary: {interaction_summary}

FORMATTING GUIDELINES:
1. Use clear section headers with ### for main topics
2. For MCQ content:
   - Number each question (1., 2., etc.)
   - List options clearly (A., B., C., D.)
   - Provide answers in format "Answer: [letter]" (without asterisks)
   - Add explanations with "Explanation: [text]" (without asterisks)
3. Use bullet points with • for lists and key points
4. Use bold sparingly for truly important terms only
5. Structure complex information with clear breaks between sections
6. Avoid LaTeX symbols and mathematical notation in favor of plain text

EDUCATIONAL APPROACH:
- Build explanations step-by-step
- Use examples when helpful
- Connect concepts to real-world applications
- Encourage critical thinking
- Provide clear, concise explanations

Always maintain professional, encouraging tone while being educational and helpful.
                """,
            ),
            ("human", "{content}"),
        ]
    )

    CONVERSATION_WITHOUT_CONTEXT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are StudyGuru AI, an advanced educational assistant. Format your responses for maximum clarity and educational impact.

FORMATTING GUIDELINES:
1. Use clear section headers with ### for main topics
2. For MCQ content:
   - Number each question clearly (1., 2., etc.)
   - List options in format: A. [option], B. [option], etc.
   - Provide answers as "Answer: [letter]" (without asterisks or bold)
   - Include explanations as "Explanation: [detailed explanation]" (without asterisks or bold)
3. Use bullet points with • for lists and key concepts
4. Use plain text formatting - avoid LaTeX, special symbols, or complex markdown
5. Structure information with clear paragraph breaks
6. Write mathematical expressions in plain text (e.g., "x squared" instead of x^2)

EDUCATIONAL APPROACH:
- Provide step-by-step explanations
- Use relevant examples
- Connect theory to practice
- Encourage deeper understanding
- Maintain clarity and precision

Be encouraging, professional, and focused on helping students learn effectively.
                """,
            ),
            ("human", "{content}"),
        ]
    )

    TITLE_GENERATION = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a title generator. Generate a short, descriptive title (max 50 characters) and a summary title (max 100 characters) for the educational content.

Rules:
1. Title should be concise and capture the main topic
2. Summary title should describe what help is being provided
3. Use simple, clear language
4. Avoid special characters or complex formatting
5. Focus on the educational subject matter

Respond in valid JSON format:
{{
    "title": "short descriptive title",
    "summary_title": "what help is being provided"
}}
                """,
            ),
            (
                "human",
                "User message: {message}\nFirst few lines of response: {response_preview}",
            ),
        ]
    )


class StudyGuruModels:
    """Model configurations for StudyGuru"""

    @staticmethod
    def get_chat_model(temperature: float = 0.2, max_tokens: int = 800) -> ChatOpenAI:
        """Get configured chat model - optimized for speed"""
        return ChatOpenAI(
            model="gpt-4o",  # Faster and cheaper model
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
            request_timeout=30,  # Add timeout for faster failure
        )

    @staticmethod
    def get_vision_model(temperature: float = 0.3, max_tokens: int = 800) -> ChatOpenAI:
        """Get configured vision model - optimized for speed"""
        return ChatOpenAI(
            model="gpt-4o",  # Use GPT-4o for vision (faster than GPT-5)
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
            request_timeout=45,  # Vision processing takes longer
        )

    @staticmethod
    def get_guardrail_model(
        temperature: float = 0.1, max_tokens: int = 150
    ) -> ChatOpenAI:
        """Get configured guardrail model (cost-optimized and fast)"""
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
            request_timeout=15,  # Fast timeout for guardrails
        )

    @staticmethod
    def get_embeddings_model() -> OpenAIEmbeddings:
        """Get configured embeddings model"""
        return OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
        )

    @staticmethod
    def get_title_model(temperature: float = 0.3, max_tokens: int = 100) -> ChatOpenAI:
        """Get configured title generation model (ultra cost-optimized)"""
        return ChatOpenAI(
            model="gpt-4o-mini",  # Most cost-effective model
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,  # Very low token limit for cost efficiency
            request_timeout=10,  # Fast timeout for quick response
        )


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
        """Get collection configuration"""
        return {
            "collection_name": settings.ZILLIZ_COLLECTION,
            "dimension": 1536,  # text-embedding-3-small dimension
            "index_params": {
                "index_type": "IVF_FLAT",
                "metric_type": settings.ZILLIZ_INDEX_METRIC,
                "params": {"nlist": 1024},
            },
        }


class StudyGuruChains:
    """Pre-configured chains for StudyGuru operations"""

    @staticmethod
    def get_document_analysis_chain():
        """Get document analysis chain"""
        model = StudyGuruModels.get_vision_model()
        parser = JsonOutputParser()
        return StudyGuruPrompts.DOCUMENT_ANALYSIS | model | parser

    @staticmethod
    def get_guardrail_chain():
        """Get guardrail check chain"""
        model = StudyGuruModels.get_guardrail_model(temperature=0.1, max_tokens=200)
        parser = JsonOutputParser()
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
        """Get title generation chain (cost-optimized)"""
        model = StudyGuruModels.get_title_model(temperature=0.3, max_tokens=100)
        parser = JsonOutputParser()
        return StudyGuruPrompts.TITLE_GENERATION | model | parser


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

    # Default settings
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_TOP_K = 5

    # Points calculation
    POINTS_PER_TOKEN = 100

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
