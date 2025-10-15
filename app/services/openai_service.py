from typing import Dict, Any, Optional, List
import json
from app.core.config import settings
from app.services.langchain_service import langchain_service


class OpenAIService:
    """
    Legacy wrapper for LangChain service to maintain backward compatibility
    """

    @staticmethod
    async def generate_embedding(text: str) -> List[float]:
        """Generate embedding using LangChain"""
        return await langchain_service.embeddings.agenerate([text])

    @staticmethod
    async def upsert_embedding(
        *,
        conv_id: str,
        user_id: str,
        text: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Upsert embedding using LangChain"""
        return await langchain_service.upsert_embedding(
            conv_id=conv_id, user_id=user_id, text=text, title=title, metadata=metadata
        )

    @staticmethod
    async def similarity_search(
        query: str,
        *,
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform similarity search using LangChain"""
        return await langchain_service.similarity_search(
            query=query, user_id=user_id or "", top_k=top_k
        )

    @staticmethod
    async def analyze_document(file_url: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """Analyze document using LangChain"""
        return await langchain_service.analyze_document(
            file_url=file_url, max_tokens=max_tokens
        )

    @staticmethod
    def calculate_points_cost(tokens_used: int) -> int:
        """Calculate points cost using LangChain service"""
        return langchain_service.calculate_points_cost(tokens_used)
