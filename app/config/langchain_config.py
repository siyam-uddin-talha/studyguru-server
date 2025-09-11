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

        1. First, detect the language of the content
        2. Identify if this contains MCQ (Multiple Choice Questions) or written questions
        3. Provide a short, descriptive title for the page/content
        4. Provide a summary title that describes what you will help the user with
        5. Based on the question type:
           - If MCQ: Extract questions and provide them in the specified JSON format
           - If written: Provide organized explanatory content

        Respond in the detected language and format your response as JSON with this structure:
        {
            "type": "mcq" or "written" or "other",
            "language": "detected language",
            "title": "short descriptive title for the content",
            "summary_title": "summary of how you will help the user",
            "_result": {
                // For MCQ type:
                "questions": [
                    {
                        "question": "question text",
                        "options": {"a": "option1", "b": "option2", "c": "option3", "d": "option4"},
                        "answer": "correct option letter or N/A",
                        "explanation": "brief explanation"
                    }
                ]
                // For written type:
                "content": "organized explanatory text as you would provide in a chat response"
            }
        }
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
        You are required to review all user inputs—including text and any attached images—and determine whether the request violates any of the following rules:

        1. Do not fulfill requests that ask for direct code generation (e.g., "write a Java function"), except when the user is presenting a question from an educational or research context that requires analysis or explanation.
        2. Prohibit content related to adult, explicit, or inappropriate material.
        3. Ensure all requests are strictly for educational, study, or research purposes. Any request outside this scope must be flagged as a violation.

        You must respond with valid JSON in this exact format:
        {{
            "is_violation": boolean,
            "violation_type": "string or null",
            "reasoning": "string"
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
        You are StudyGuru AI, an educational assistant. You have access to the user's learning history and context.
        
        Current conversation topic: {interaction_title}
        Context summary: {interaction_summary}
        
        Instructions:
        1. Use the retrieved context from the user's learning history when relevant to provide better, personalized responses
        2. If the context is not relevant to the current question, answer based on your knowledge
        3. Maintain consistency with the user's learning style and previous interactions
        4. Provide clear, educational explanations that build upon previous knowledge when possible
        5. If this is a follow-up question, reference relevant previous discussions when helpful
        
        Always provide helpful, accurate educational assistance.
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
        You are StudyGuru AI, an educational assistant. Provide helpful, accurate educational assistance based on the user's question and any provided context.
        """,
            ),
            ("human", "{content}"),
        ]
    )


class StudyGuruModels:
    """Model configurations for StudyGuru"""

    @staticmethod
    def get_chat_model(temperature: float = 0.2, max_tokens: int = 1000) -> ChatOpenAI:
        """Get configured chat model"""
        return ChatOpenAI(
            model="gpt-5",
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
        )

    @staticmethod
    def get_vision_model(
        temperature: float = 0.3, max_tokens: int = 1000
    ) -> ChatOpenAI:
        """Get configured vision model"""
        return ChatOpenAI(
            model="gpt-5",
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
        )

    @staticmethod
    def get_guardrail_model(
        temperature: float = 0.1, max_tokens: int = 200
    ) -> ChatOpenAI:
        """Get configured guardrail model (cost-optimized)"""
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
            max_tokens=max_tokens,
        )

    @staticmethod
    def get_embeddings_model() -> OpenAIEmbeddings:
        """Get configured embeddings model"""
        return OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
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
