"""
Enhanced context retrieval service for the RAG system
Implements multi-level context retrieval strategy with caching and ranking

DEPRECATION NOTICE:
===================
This service is being replaced by simplified_context_service.py for better performance.
The simplified service uses only 2 context sources (document + vector search) instead of 5,
resulting in ~60% faster retrieval times.

New code should use:
    from app.services.simplified_context_service import simplified_context_service

This file is kept for backwards compatibility but may be removed in a future version.
"""

import asyncio
import hashlib
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.interaction import Interaction, Conversation
from app.models.context import (
    UserLearningProfile,
    DocumentContext,
    ContextUsageLog,
)
from app.models.media import Media
from app.services.langchain_service import langchain_service
from app.services.cache_service import cache_service


class ContextRetrievalService:
    """Enhanced context retrieval service with multi-level strategy"""

    def __init__(self):
        self.cache_ttl = 900  # 15 minutes (increased for better performance)
        self.max_context_length = 10000  # Maximum context length to prevent token overflow (increased from 4000)

    async def get_comprehensive_context(
        self,
        user_id: str,
        interaction_id: Optional[str],
        message: str,
        include_cross_interaction: bool = True,
        max_context_length: int = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive context using multi-level retrieval strategy

        Returns:
            Dict with context from all sources and metadata about retrieval
        """
        start_time = datetime.now()
        max_context_length = max_context_length or self.max_context_length

        # Generate cache key
        cache_key = self._generate_cache_key(user_id, interaction_id, message)

        # Try cache first
        cached_context = await cache_service.get(cache_key)
        if cached_context:
            return cached_context

        context_sources = {
            "semantic_summary": "",
            "vector_search": [],
            "document_content": [],
            "cross_interaction": [],
            "related_conversations": [],
        }

        retrieval_metadata = {
            "sources_used": [],
            "sources_ignored": [],
            "retrieval_times": {},
            "context_lengths": {},
            "relevance_scores": {},
        }

        try:
            # === PARALLEL CONTEXT RETRIEVAL FOR SPEED ===
            # Run all context retrieval levels in parallel for much faster performance
            context_tasks = []

            # Level 1: Semantic summary
            context_tasks.append(
                self._get_semantic_summary_context(
                    user_id, interaction_id, retrieval_metadata
                )
            )

            # Level 2: Vector search
            context_tasks.append(
                self._get_vector_search_context(
                    user_id, interaction_id, message, retrieval_metadata
                )
            )

            # Level 3: Document content
            context_tasks.append(
                self._get_document_context(
                    user_id, interaction_id, message, retrieval_metadata
                )
            )

            # Level 4: Cross-interaction (if enabled)
            if include_cross_interaction:
                context_tasks.append(
                    self._get_cross_interaction_context(
                        user_id, interaction_id, message, retrieval_metadata
                    )
                )

            # Level 5: Related conversations
            context_tasks.append(
                self._get_related_conversations(
                    user_id, interaction_id, message, retrieval_metadata
                )
            )

            # Execute all context retrieval tasks in parallel
            context_results = await asyncio.gather(
                *context_tasks, return_exceptions=True
            )

            # Process results
            context_sources["semantic_summary"] = (
                context_results[0]
                if not isinstance(context_results[0], Exception)
                else {}
            )
            context_sources["vector_search"] = (
                context_results[1]
                if not isinstance(context_results[1], Exception)
                else {}
            )
            context_sources["document_content"] = (
                context_results[2]
                if not isinstance(context_results[2], Exception)
                else {}
            )

            if include_cross_interaction:
                context_sources["cross_interaction"] = (
                    context_results[3]
                    if not isinstance(context_results[3], Exception)
                    else {}
                )

            context_sources["related_conversations"] = (
                context_results[-1]
                if not isinstance(context_results[-1], Exception)
                else {}
            )

            # === CONTEXT RANKING AND OPTIMIZATION ===
            ranked_context = await self._rank_and_optimize_context(
                context_sources, message, max_context_length
            )

            # === BUILD FINAL CONTEXT ===
            final_context = self._build_final_context(ranked_context, message)

            # Calculate total retrieval time
            total_time = (datetime.now() - start_time).total_seconds()
            retrieval_metadata["total_retrieval_time"] = total_time

            # Cache the result
            result = {
                "context": final_context,
                "sources": context_sources,
                "metadata": retrieval_metadata,
                "context_length": len(final_context),
                "timestamp": datetime.now().isoformat(),
            }

            await cache_service.set(cache_key, result, self.cache_ttl)

            return result

        except Exception as e:
            # Log error and return minimal context
            retrieval_metadata["error"] = str(e)
            return {
                "context": "",
                "sources": context_sources,
                "metadata": retrieval_metadata,
                "context_length": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def _get_semantic_summary_context(
        self, user_id: str, interaction_id: Optional[str], metadata: Dict
    ) -> str:
        """
        Get interaction-level semantic summary context

        DEPRECATED: This function is no longer used in the simplified RAG system.
        The semantic_summary field was removed from the Interaction model.
        Vector search now handles all semantic context retrieval.

        This function is kept for backwards compatibility but returns empty string.
        """
        start_time = datetime.now()

        # Return empty - semantic_summary field was removed in RAG streamlining
        # Vector search now handles semantic context retrieval
        metadata["sources_ignored"].append("semantic_summary")
        metadata["retrieval_times"]["semantic_summary"] = (
            datetime.now() - start_time
        ).total_seconds()

        return ""

    async def _get_vector_search_context(
        self, user_id: str, interaction_id: Optional[str], message: str, metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Get context from vector search using enhanced optimization service"""
        start_time = datetime.now()

        try:
            # Use the new vector optimization service for better results
            from app.services.vector_optimization_service import (
                vector_optimization_service,
                SearchQuery,
            )

            # Build enhanced search query
            search_query = SearchQuery(
                query=message,
                user_id=user_id,
                interaction_id=interaction_id,
                top_k=8 if interaction_id else 10,
                use_hybrid_search=True,
                use_query_expansion=True,
                boost_recent=True,
                boost_interaction=bool(interaction_id),
            )

            # Get enhanced search results
            search_results = await vector_optimization_service.hybrid_search(
                search_query
            )

            # Convert SearchResult objects to expected format
            results = []
            for result in search_results:
                # Truncate content for faster processing (increased limit for more context)
                content = result.content
                original_length = len(content)
                if len(content) > 1000:  # Increased from 300 to 1000 for more context
                    content = content[:1000] + "..."
                    print(
                        f"🔍 [CONTEXT SERVICE] Content truncated: {original_length} -> {len(content)} chars"
                    )
                else:
                    print(f"🔍 [CONTEXT SERVICE] Content length: {len(content)} chars")
                print(
                    f"🔍 [CONTEXT SERVICE] Title: {result.title}, Type: {result.content_type}"
                )

                results.append(
                    {
                        "id": result.id,
                        "interaction_id": result.interaction_id,
                        "title": result.title,
                        "content": content,
                        "metadata": result.metadata,
                        "score": result.score,
                        "type": result.content_type,
                        "relevance_score": result.relevance_score,
                        "recency_score": result.recency_score,
                        "importance_score": result.importance_score,
                        "topic_tags": result.topic_tags,
                        "question_numbers": result.question_numbers,
                    }
                )

            # Filter and rank results
            filtered_results = []
            for result in results:
                if result.get("content") and len(result["content"].strip()) > 10:
                    # Add relevance score if not present
                    if "score" not in result:
                        result["score"] = 0.8  # Default score

                    # Boost score for interaction-specific results
                    if result.get("interaction_id") == interaction_id:
                        result["score"] = min(1.0, result["score"] + 0.2)

                    filtered_results.append(result)

            # Sort by score and limit
            filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_results = filtered_results[:5]

            # Update metadata
            metadata["sources_used"].append("vector_search")
            metadata["retrieval_times"]["vector_search"] = (
                datetime.now() - start_time
            ).total_seconds()
            metadata["context_lengths"]["vector_search"] = sum(
                len(r.get("content", "")) for r in final_results
            )
            metadata["relevance_scores"]["vector_search"] = [
                r.get("score", 0) for r in final_results
            ]

            return final_results

        except Exception as e:
            metadata["sources_ignored"].append("vector_search")
            metadata["retrieval_times"]["vector_search"] = (
                datetime.now() - start_time
            ).total_seconds()
            return []

    async def _get_document_context(
        self, user_id: str, interaction_id: Optional[str], message: str, metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Get context from uploaded documents using enhanced document integration service"""
        start_time = datetime.now()

        try:
            # Use the new document integration service for better document search
            from app.services.document_integration_service import (
                document_integration_service,
            )

            # Extract question numbers from the message for targeted search
            question_numbers = self._extract_question_numbers(message)

            # AGGRESSIVE APPROACH: Always try to get recent document content first
            document_results = []

            # 1. If user is asking about specific questions, try to get those directly
            if question_numbers and interaction_id:
                for q_num in question_numbers:
                    doc_content = await document_integration_service.get_document_by_question_number(
                        user_id, interaction_id, q_num
                    )
                    if doc_content:
                        document_results.append(
                            {
                                "id": f"doc_q_{q_num}",
                                "media_id": f"question_{q_num}",
                                "document_type": doc_content.get(
                                    "document_type", "unknown"
                                ),
                                "content": f"Question {q_num}: {doc_content['question_text']}\n\nAnswer: {doc_content.get('answer', 'Not provided')}",
                                "main_topics": doc_content.get("main_topics", []),
                                "total_questions": 1,
                                "relevance_score": 0.95,  # Very high relevance for specific questions
                                "question_number": q_num,
                                "subject_area": doc_content.get(
                                    "subject_area", "unknown"
                                ),
                                "difficulty_level": doc_content.get(
                                    "difficulty_level", "unknown"
                                ),
                                "key_concepts": doc_content.get("key_concepts", []),
                            }
                        )

            # 2. AGGRESSIVE FALLBACK: If no specific questions found OR user is asking general questions,
            # get ALL document content from the current interaction
            if not document_results and interaction_id:
                print(
                    f"🔍 No specific questions found, getting ALL document content for interaction {interaction_id}"
                )

                # Get all document context for this interaction
                async with AsyncSessionLocal() as db:
                    from app.models.context import DocumentContext
                    from sqlalchemy import select, and_

                    result = await db.execute(
                        select(DocumentContext).where(
                            and_(
                                DocumentContext.user_id == user_id,
                                DocumentContext.interaction_id == interaction_id,
                            )
                        )
                    )
                    doc_contexts = result.scalars().all()

                    for doc_ctx in doc_contexts:
                        # Extract all questions from the document
                        if doc_ctx.question_mapping:
                            for (
                                q_num,
                                question_text,
                            ) in doc_ctx.question_mapping.items():
                                answer = doc_ctx.answer_key.get(
                                    q_num, "Answer not provided"
                                )
                                document_results.append(
                                    {
                                        "id": f"doc_all_{q_num}",
                                        "media_id": doc_ctx.media_id,
                                        "document_type": doc_ctx.document_type,
                                        "content": f"Question {q_num}: {question_text}\n\nAnswer: {answer}",
                                        "main_topics": doc_ctx.main_topics or [],
                                        "total_questions": len(
                                            doc_ctx.question_mapping
                                        ),
                                        "relevance_score": 0.9,  # High relevance for all questions
                                        "question_number": q_num,
                                        "subject_area": doc_ctx.subject_area
                                        or "unknown",
                                        "difficulty_level": doc_ctx.difficulty_level
                                        or "unknown",
                                        "key_concepts": doc_ctx.key_concepts or [],
                                    }
                                )
                        else:
                            # If no question mapping, use the full content
                            document_results.append(
                                {
                                    "id": f"doc_full_{doc_ctx.media_id}",
                                    "media_id": doc_ctx.media_id,
                                    "document_type": doc_ctx.document_type,
                                    "content": doc_ctx.full_content
                                    or "Document content not available",
                                    "main_topics": doc_ctx.main_topics or [],
                                    "total_questions": doc_ctx.total_questions or 0,
                                    "relevance_score": 0.8,
                                    "subject_area": doc_ctx.subject_area or "unknown",
                                    "difficulty_level": doc_ctx.difficulty_level
                                    or "unknown",
                                    "key_concepts": doc_ctx.key_concepts or [],
                                }
                            )

            if document_results:
                metadata["sources_used"].append("document_content")
                metadata["retrieval_times"]["document_content"] = (
                    datetime.now() - start_time
                ).total_seconds()
                metadata["context_lengths"]["document_content"] = sum(
                    len(result["content"]) for result in document_results
                )
                print(f"✅ Found {len(document_results)} document results for context")
                return document_results

            # Fallback to document search using the integration service
            search_results = await document_integration_service.search_documents(
                user_id=user_id, query=message, interaction_id=interaction_id, top_k=5
            )

            if search_results:
                # Convert search results to expected format
                document_results = []
                for result in search_results:
                    document_results.append(
                        {
                            "id": result["id"],
                            "media_id": result.get("id", "unknown"),
                            "document_type": result.get("document_type", "unknown"),
                            "content": result["content"],
                            "main_topics": result.get("main_topics", []),
                            "total_questions": 1,  # Each chunk represents part of a document
                            "relevance_score": result.get("score", 0.8),
                            "question_number": result.get("question_number"),
                            "subject_area": result.get("subject_area", "unknown"),
                            "difficulty_level": result.get(
                                "difficulty_level", "unknown"
                            ),
                            "key_concepts": result.get("key_concepts", []),
                        }
                    )

                metadata["sources_used"].append("document_content")
                metadata["retrieval_times"]["document_content"] = (
                    datetime.now() - start_time
                ).total_seconds()
                metadata["context_lengths"]["document_content"] = sum(
                    len(result["content"]) for result in document_results
                )
                return document_results

            # Fallback to original database-based approach
            async with AsyncSessionLocal() as db:
                # Get document context for the interaction
                if interaction_id:
                    result = await db.execute(
                        select(DocumentContext).where(
                            and_(
                                DocumentContext.interaction_id == interaction_id,
                                DocumentContext.user_id == user_id,
                            )
                        )
                    )
                    document_contexts = result.scalars().all()
                else:
                    # Get recent document contexts for the user
                    result = await db.execute(
                        select(DocumentContext)
                        .where(DocumentContext.user_id == user_id)
                        .order_by(desc(DocumentContext.created_at))
                        .limit(3)
                    )
                    document_contexts = result.scalars().all()

                document_results = []
                for doc_ctx in document_contexts:
                    # Check if the message references specific questions in this document
                    question_numbers = self._extract_question_numbers(message)
                    relevant_content = self._extract_relevant_document_content(
                        doc_ctx, message, question_numbers
                    )

                    if relevant_content:
                        document_results.append(
                            {
                                "id": doc_ctx.id,
                                "media_id": doc_ctx.media_id,
                                "document_type": doc_ctx.document_type,
                                "content": relevant_content,
                                "main_topics": doc_ctx.main_topics,
                                "total_questions": doc_ctx.total_questions,
                                "relevance_score": 0.9,  # High relevance for document content
                            }
                        )

                # Update metadata
                metadata["sources_used"].append("document_content")
                metadata["retrieval_times"]["document_content"] = (
                    datetime.now() - start_time
                ).total_seconds()
                metadata["context_lengths"]["document_content"] = sum(
                    len(r.get("content", "")) for r in document_results
                )

                return document_results

        except Exception as e:
            print(f"⚠️ Document context retrieval error: {e}")
            metadata["sources_ignored"].append("document_content")
            metadata["retrieval_times"]["document_content"] = (
                datetime.now() - start_time
            ).total_seconds()
            return []

    async def _get_cross_interaction_context(
        self, user_id: str, interaction_id: Optional[str], message: str, metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Get context from related interactions across the user's history

        DEPRECATED: This function is no longer used in the simplified RAG system.
        The related_interactions field was removed from UserLearningProfile.
        Vector search now handles cross-interaction context via semantic similarity.

        This function is kept for backwards compatibility but returns empty list.
        """
        start_time = datetime.now()

        # Return empty - related_interactions field was removed in RAG streamlining
        # Vector search now handles cross-interaction context via semantic similarity
        metadata["sources_ignored"].append("cross_interaction")
        metadata["retrieval_times"]["cross_interaction"] = (
            datetime.now() - start_time
        ).total_seconds()

        return []

    async def _get_related_conversations(
        self, user_id: str, interaction_id: Optional[str], message: str, metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Get context from related conversations within the same interaction"""
        start_time = datetime.now()

        try:
            if not interaction_id:
                return []

            async with AsyncSessionLocal() as db:
                # Get recent conversations from the same interaction
                result = await db.execute(
                    select(Conversation)
                    .where(
                        and_(
                            Conversation.interaction_id == interaction_id,
                            Conversation.role == "AI",  # Get AI responses for context
                        )
                    )
                    .order_by(desc(Conversation.created_at))
                    .limit(3)
                )
                conversations = result.scalars().all()

                related_conversations = []
                for conv in conversations:
                    if conv.content and isinstance(conv.content, dict):
                        content = conv.content.get("_result", {}).get("note", "")
                        if (
                            content and len(content) > 20
                        ):  # Only include substantial content
                            related_conversations.append(
                                {
                                    "id": conv.id,
                                    "content": content,
                                    "created_at": (
                                        conv.created_at.isoformat()
                                        if conv.created_at
                                        else ""
                                    ),
                                    "relevance_score": 0.8,  # High relevance for same interaction
                                }
                            )

                # Update metadata
                metadata["sources_used"].append("related_conversations")
                metadata["retrieval_times"]["related_conversations"] = (
                    datetime.now() - start_time
                ).total_seconds()
                metadata["context_lengths"]["related_conversations"] = sum(
                    len(r.get("content", "")) for r in related_conversations
                )

                return related_conversations

        except Exception as e:
            metadata["sources_ignored"].append("related_conversations")
            metadata["retrieval_times"]["related_conversations"] = (
                datetime.now() - start_time
            ).total_seconds()
            return []

    def _extract_question_numbers(self, message: str) -> List[int]:
        """Extract question numbers from the message"""
        question_numbers = []
        patterns = [
            r"question\s+(\d+)",
            r"problem\s+(\d+)",
            r"mcq\s+(\d+)",
            r"equation\s+(\d+)",
            r"exercise\s+(\d+)",
            r"task\s+(\d+)",
            r"(\d+)\.\s*$",  # Number at end of line
            r"explain\s+(?:the\s+)?(?:equation|question|problem|mcq)\s+(\d+)",  # "explain equation 6"
            r"what\s+is\s+(?:the\s+)?(?:equation|question|problem|mcq)\s+(\d+)",  # "what is equation 6"
            r"solve\s+(?:the\s+)?(?:equation|question|problem|mcq)\s+(\d+)",  # "solve equation 6"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                try:
                    question_numbers.append(int(match))
                except ValueError:
                    continue

        # Remove duplicates while preserving order
        seen = set()
        unique_question_numbers = []
        for num in question_numbers:
            if num not in seen:
                seen.add(num)
                unique_question_numbers.append(num)

        return unique_question_numbers

    def _extract_relevant_document_content(
        self, doc_ctx: DocumentContext, message: str, question_numbers: List[int]
    ) -> str:
        """Extract relevant content from document context based on message and question numbers"""
        if not doc_ctx.full_content:
            return ""

        # If specific question numbers are mentioned, try to find those questions
        if question_numbers:
            print(
                f"🎯 Context Service: Looking for question numbers: {question_numbers}"
            )
            content_parts = []
            for q_num in question_numbers:
                # Try multiple patterns to find the question
                patterns = [
                    rf"{q_num}\.?\s+.*?(?=\d+\.|$)",  # Standard numbered pattern
                    rf"(?:question|problem|equation|exercise|task)\s+{q_num}.*?(?=\d+\.|$)",  # With type prefix
                    rf"{q_num}\.?\s+.*?(?=\n\d+\.|\n\n|\Z)",  # More flexible ending
                ]

                found_question = False
                for pattern in patterns:
                    match = re.search(
                        pattern, doc_ctx.full_content, re.IGNORECASE | re.DOTALL
                    )
                    if match:
                        content_parts.append(
                            f"Question {q_num}: {match.group(0).strip()}"
                        )
                        print(
                            f"✅ Context Service: Found question {q_num} using pattern"
                        )
                        found_question = True
                        break

                # If no specific pattern found, try to find any content with the number
                if not found_question:
                    # Look for any line containing the number
                    lines = doc_ctx.full_content.split("\n")
                    for line in lines:
                        if re.search(rf"\b{q_num}\b", line):
                            content_parts.append(f"Question {q_num}: {line.strip()}")
                            print(
                                f"✅ Context Service: Found question {q_num} using line search"
                            )
                            break

            if content_parts:
                return "\n\n".join(content_parts)

        # If no specific questions, return summary or key concepts
        if doc_ctx.content_summary:
            return doc_ctx.content_summary
        elif doc_ctx.key_concepts:
            return f"Key concepts: {', '.join(doc_ctx.key_concepts[:5])}"
        else:
            # Return first part of content
            return (
                doc_ctx.full_content[:500] + "..."
                if len(doc_ctx.full_content) > 500
                else doc_ctx.full_content
            )

    def _is_relevant_to_message(self, summary_data: Dict, message: str) -> bool:
        """Check if semantic summary is relevant to the current message"""
        if not summary_data:
            return False

        # Check if any key topics match words in the message
        key_topics = summary_data.get("key_topics", [])
        message_lower = message.lower()

        for topic in key_topics:
            if topic.lower() in message_lower:
                return True

        # Check if the summary contains relevant terms
        summary_text = summary_data.get("updated_summary", "").lower()
        if any(word in summary_text for word in message_lower.split() if len(word) > 3):
            return True

        return False

    async def _rank_and_optimize_context(
        self, context_sources: Dict, message: str, max_length: int
    ) -> Dict[str, Any]:
        """Rank and optimize context based on relevance and length constraints"""
        ranked_context = {
            "semantic_summary": context_sources["semantic_summary"],
            "vector_search": [],
            "document_content": [],
            "cross_interaction": [],
            "related_conversations": [],
        }

        current_length = len(context_sources["semantic_summary"])

        # Prioritize vector search results (most relevant)
        for result in context_sources["vector_search"]:
            if current_length + len(result.get("content", "")) < max_length:
                ranked_context["vector_search"].append(result)
                current_length += len(result.get("content", ""))

        # Add document content (high relevance for specific questions)
        for result in context_sources["document_content"]:
            if current_length + len(result.get("content", "")) < max_length:
                ranked_context["document_content"].append(result)
                current_length += len(result.get("content", ""))

        # Add related conversations (contextual relevance)
        for result in context_sources["related_conversations"]:
            if current_length + len(result.get("content", "")) < max_length:
                ranked_context["related_conversations"].append(result)
                current_length += len(result.get("content", ""))

        # Add cross-interaction context (lower priority)
        for result in context_sources["cross_interaction"]:
            if current_length + len(result.get("summary", "")) < max_length:
                ranked_context["cross_interaction"].append(result)
                current_length += len(result.get("summary", ""))

        return ranked_context

    def _build_final_context(self, ranked_context: Dict, message: str) -> str:
        """Build the final context string from ranked sources"""
        context_parts = []

        # Add semantic summary first (always include if available)
        if ranked_context["semantic_summary"]:
            context_parts.append(ranked_context["semantic_summary"])

        # Add vector search results
        if ranked_context["vector_search"]:
            vector_parts = []
            for result in ranked_context["vector_search"]:
                content = result.get("content", "")
                if content:
                    vector_parts.append(f"**Previous Discussion:** {content}")
            if vector_parts:
                context_parts.append("\n".join(vector_parts))

        # Add document content
        if ranked_context["document_content"]:
            doc_parts = []
            for result in ranked_context["document_content"]:
                content = result.get("content", "")
                if content:
                    doc_parts.append(f"**Document Content:** {content}")
            if doc_parts:
                context_parts.append("\n".join(doc_parts))

        # Add related conversations
        if ranked_context["related_conversations"]:
            conv_parts = []
            for result in ranked_context["related_conversations"]:
                content = result.get("content", "")
                if content:
                    conv_parts.append(f"**Related Discussion:** {content}")
            if conv_parts:
                context_parts.append("\n".join(conv_parts))

        # Add cross-interaction context
        if ranked_context["cross_interaction"]:
            cross_parts = []
            for result in ranked_context["cross_interaction"]:
                summary = result.get("summary", "")
                if summary:
                    cross_parts.append(f"**Related Topic:** {summary}")
            if cross_parts:
                context_parts.append("\n".join(cross_parts))

        return "\n\n".join(context_parts)

    def _generate_cache_key(
        self, user_id: str, interaction_id: Optional[str], message: str
    ) -> str:
        """Generate cache key for context retrieval"""
        key_components = [
            user_id,
            interaction_id or "",
            message[:100],
        ]  # Limit message length
        key_string = "|".join(key_components)
        return f"context:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def log_context_usage(
        self,
        user_id: str,
        interaction_id: str,
        conversation_id: Optional[str],
        context_sources_used: List[str],
        context_sources_ignored: List[str],
        retrieval_time: float,
        user_query: str,
        query_type: str,
        response_quality: Optional[str] = None,
    ):
        """Log context usage for monitoring and optimization"""
        try:
            async with AsyncSessionLocal() as db:
                usage_log = ContextUsageLog(
                    interaction_id=interaction_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    context_sources_used=context_sources_used,
                    context_sources_ignored=context_sources_ignored,
                    context_retrieval_time=retrieval_time,
                    user_query=user_query,
                    query_type=query_type,
                    response_quality=response_quality,
                    created_at=datetime.now(),
                )

                db.add(usage_log)
                await db.commit()

        except Exception as e:
            # Log error but don't fail the main operation
            print(f"Failed to log context usage: {e}")


# Global instance
context_service = ContextRetrievalService()
