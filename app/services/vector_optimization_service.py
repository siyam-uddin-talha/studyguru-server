"""
Enhanced Vector Database Optimization Service

This service provides advanced vector search capabilities including:
- Hybrid search (semantic + keyword)
- Enhanced metadata filtering
- Multi-level retrieval strategies
- Query expansion and optimization
- Performance monitoring and caching
"""

import asyncio
import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from pymilvus import Collection, connections, utility
from pymilvus import DataType, FieldSchema, CollectionSchema

from app.core.config import settings
from app.services.langchain_service import langchain_service

try:
    from app.core.cache import cache_user_context, get_cached_user_context
except ImportError:
    # Fallback if cache module is not available
    async def cache_user_context(key: str, data: dict, ttl: int = 300):
        pass

    async def get_cached_user_context(key: str):
        return None


@dataclass
class SearchResult:
    """Enhanced search result with metadata"""

    id: str
    content: str
    title: str
    score: float
    metadata: Dict[str, Any]
    interaction_id: str
    user_id: str
    content_type: str
    relevance_score: float
    recency_score: float
    importance_score: float
    topic_tags: List[str]
    question_numbers: List[str]


@dataclass
class SearchQuery:
    """Enhanced search query with optimization parameters"""

    query: str
    user_id: str
    interaction_id: Optional[str] = None
    top_k: int = 10
    min_score: float = 0.0
    content_types: List[str] = None
    topic_filters: List[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    use_hybrid_search: bool = True
    use_query_expansion: bool = True
    boost_recent: bool = True
    boost_interaction: bool = True


class VectorOptimizationService:
    """Enhanced vector database service with optimization features"""

    def __init__(self):
        self.vector_store = langchain_service.vector_store
        self.embeddings = langchain_service.embeddings
        self.collection_name = settings.ZILLIZ_COLLECTION
        self._query_cache = {}
        self._cache_ttl = 300  # 5 minutes

    async def hybrid_search(self, search_query: SearchQuery) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic similarity and keyword matching
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"hybrid_search_{hash(str(search_query))}"
            cached_result = await get_cached_user_context(cache_key)
            if cached_result:
                return [
                    SearchResult(**result)
                    for result in cached_result.get("results", [])
                ]

            # Step 1: Query expansion
            expanded_queries = (
                await self._expand_query(search_query)
                if search_query.use_query_expansion
                else [search_query.query]
            )

            # Step 2: Parallel semantic and keyword searches
            semantic_results = await self._semantic_search(
                search_query, expanded_queries
            )
            keyword_results = await self._keyword_search(search_query, expanded_queries)

            # Step 3: Combine and rank results
            combined_results = await self._combine_and_rank_results(
                semantic_results, keyword_results, search_query
            )

            # Step 4: Apply final filters and boosters
            final_results = await self._apply_final_filters(
                combined_results, search_query
            )

            # Cache results
            cache_data = [result.__dict__ for result in final_results]
            await cache_user_context(
                cache_key, {"results": cache_data}, ttl=self._cache_ttl
            )

            search_time = time.time() - start_time
            print(
                f"🔍 Hybrid search completed in {search_time:.2f}s, found {len(final_results)} results"
            )

            return final_results

        except Exception as e:
            print(f"❌ Hybrid search error: {e}")
            return []

    async def _expand_query(self, search_query: SearchQuery) -> List[str]:
        """Expand query with synonyms and related terms"""
        try:
            base_query = search_query.query.lower()
            expanded_queries = [search_query.query]

            # Mathematical term expansions
            math_expansions = {
                "solve": ["find", "calculate", "compute", "determine"],
                "equation": ["formula", "expression", "function"],
                "problem": ["question", "exercise", "challenge"],
                "answer": ["solution", "result", "outcome"],
                "explain": ["describe", "clarify", "elaborate", "detail"],
                "how": ["what", "why", "when", "where"],
                "find": ["locate", "identify", "discover", "determine"],
            }

            # Add expanded terms
            for term, synonyms in math_expansions.items():
                if term in base_query:
                    for synonym in synonyms:
                        expanded_query = base_query.replace(term, synonym)
                        if expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)

            # Number-specific expansions (for question references)
            number_match = re.search(r"(\d+)", base_query)
            if number_match:
                number = number_match.group(1)
                # Add variations like "6.", "problem 6", "question 6"
                variations = [
                    f"{number}.",
                    f"problem {number}",
                    f"question {number}",
                    f"mcq {number}",
                    f"equation {number}",
                ]
                for variation in variations:
                    if variation not in expanded_queries:
                        expanded_queries.append(variation)

            return expanded_queries[:5]  # Limit to 5 expanded queries

        except Exception as e:
            print(f"⚠️ Query expansion error: {e}")
            return [search_query.query]

    async def _semantic_search(
        self, search_query: SearchQuery, expanded_queries: List[str]
    ) -> List[SearchResult]:
        """Perform semantic similarity search"""
        try:
            if not self.vector_store:
                print("⚠️ Vector store not available for semantic search")
                return []

            all_results = []

            for query in expanded_queries:
                # Create retriever with enhanced filters
                filter_expr = self._build_filter_expression(search_query)

                retriever = self.vector_store.as_retriever(
                    search_kwargs={
                        "k": search_query.top_k
                        * 2,  # Get more results for better ranking
                        "expr": filter_expr,
                        "score_threshold": search_query.min_score,
                    }
                )

                # Run search
                loop = asyncio.get_event_loop()
                docs = await loop.run_in_executor(
                    ThreadPoolExecutor(max_workers=1), retriever.invoke, query
                )

                # Convert to SearchResult objects
                for doc in docs:
                    result = self._convert_to_search_result(doc, "semantic")
                    if result:
                        all_results.append(result)

            return all_results

        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return []

    async def _keyword_search(
        self, search_query: SearchQuery, expanded_queries: List[str]
    ) -> List[SearchResult]:
        """Perform keyword-based search using metadata filtering"""
        try:
            all_results = []

            # Extract keywords from queries
            keywords = set()
            for query in expanded_queries:
                # Extract meaningful words (longer than 2 characters, not common words)
                words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
                stop_words = {
                    "the",
                    "and",
                    "for",
                    "are",
                    "but",
                    "not",
                    "you",
                    "all",
                    "can",
                    "had",
                    "her",
                    "was",
                    "one",
                    "our",
                    "out",
                    "day",
                    "get",
                    "has",
                    "him",
                    "his",
                    "how",
                    "its",
                    "may",
                    "new",
                    "now",
                    "old",
                    "see",
                    "two",
                    "way",
                    "who",
                    "boy",
                    "did",
                    "man",
                    "oil",
                    "sit",
                    "try",
                    "use",
                    "she",
                    "put",
                    "end",
                    "why",
                    "let",
                    "ask",
                    "run",
                    "own",
                    "say",
                    "too",
                    "any",
                    "may",
                    "set",
                    "try",
                    "yes",
                    "yet",
                    "big",
                    "few",
                    "got",
                    "lot",
                    "off",
                    "old",
                    "red",
                    "top",
                    "win",
                    "yes",
                }
                keywords.update([word for word in words if word not in stop_words])

            if not keywords:
                return []

            # Build keyword filter expression
            keyword_conditions = []
            for keyword in list(keywords)[:10]:  # Limit to 10 keywords
                keyword_conditions.append(f'text like "%{keyword}%"')

            if keyword_conditions:
                keyword_expr = " || ".join(keyword_conditions)
                filter_expr = self._build_filter_expression(
                    search_query, additional_conditions=[keyword_expr]
                )

                retriever = self.vector_store.as_retriever(
                    search_kwargs={
                        "k": search_query.top_k,
                        "expr": filter_expr,
                    }
                )

                # Run search
                loop = asyncio.get_event_loop()
                docs = await loop.run_in_executor(
                    ThreadPoolExecutor(max_workers=1),
                    retriever.invoke,
                    search_query.query,
                )

                # Convert to SearchResult objects
                for doc in docs:
                    result = self._convert_to_search_result(doc, "keyword")
                    if result:
                        all_results.append(result)

            return all_results

        except Exception as e:
            print(f"❌ Keyword search error: {e}")
            return []

    def _build_filter_expression(
        self, search_query: SearchQuery, additional_conditions: List[str] = None
    ) -> str:
        """Build Milvus filter expression"""
        conditions = [f"user_id == '{search_query.user_id}'"]

        if search_query.interaction_id:
            conditions.append(f"interaction_id == '{search_query.interaction_id}'")

        if search_query.content_types:
            type_conditions = [
                f"type == '{content_type}'"
                for content_type in search_query.content_types
            ]
            conditions.append(f"({' || '.join(type_conditions)})")

        if additional_conditions:
            conditions.extend(additional_conditions)

        return " && ".join(conditions)

    def _convert_to_search_result(
        self, doc: Document, search_type: str
    ) -> Optional[SearchResult]:
        """Convert Document to SearchResult with enhanced metadata"""
        try:
            metadata = doc.metadata or {}

            # Extract enhanced metadata
            content_type = metadata.get("type", "unknown")
            topic_tags = metadata.get("topic_tags", [])
            question_numbers = metadata.get("question_numbers", [])

            # Calculate relevance scores
            base_score = getattr(doc, "score", 0.0)
            relevance_score = self._calculate_relevance_score(doc, search_type)
            recency_score = self._calculate_recency_score(metadata)
            importance_score = self._calculate_importance_score(metadata)

            return SearchResult(
                id=metadata.get("original_id", metadata.get("id", "")),
                content=doc.page_content,
                title=metadata.get("title", ""),
                score=base_score,
                metadata=metadata,
                interaction_id=metadata.get("interaction_id", ""),
                user_id=metadata.get("user_id", ""),
                content_type=content_type,
                relevance_score=relevance_score,
                recency_score=recency_score,
                importance_score=importance_score,
                topic_tags=topic_tags,
                question_numbers=question_numbers,
            )

        except Exception as e:
            print(f"⚠️ Error converting document to search result: {e}")
            return None

    def _calculate_relevance_score(self, doc: Document, search_type: str) -> float:
        """Calculate relevance score based on content and search type"""
        try:
            base_score = getattr(doc, "score", 0.0)

            # Boost scores based on search type
            if search_type == "semantic":
                return base_score * 1.2  # Semantic search gets slight boost
            elif search_type == "keyword":
                return base_score * 1.0  # Keyword search base score
            else:
                return base_score

        except Exception:
            return 0.0

    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score based on creation time"""
        try:
            # Try to extract timestamp from metadata
            created_at = metadata.get("created_at")
            if created_at:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                elif isinstance(created_at, datetime):
                    pass
                else:
                    return 0.5  # Default score if can't parse

                # Calculate days since creation
                days_old = (datetime.now() - created_at).days

                # Score decreases with age (max 1.0 for today, min 0.1 for very old)
                if days_old == 0:
                    return 1.0
                elif days_old <= 7:
                    return 0.9
                elif days_old <= 30:
                    return 0.7
                elif days_old <= 90:
                    return 0.5
                else:
                    return 0.3

            return 0.5  # Default score if no timestamp

        except Exception:
            return 0.5

    def _calculate_importance_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate importance score based on content characteristics"""
        try:
            score = 0.5  # Base score

            # Boost for specific content types
            content_type = metadata.get("type", "")
            if content_type == "ai_response":
                score += 0.2  # AI responses are generally more important
            elif content_type == "user_message":
                score += 0.1  # User messages are moderately important

            # Boost for content with summaries
            if metadata.get("summary"):
                score += 0.2

            # Boost for content with topic tags
            if metadata.get("topic_tags"):
                score += 0.1

            # Boost for content with question numbers (structured content)
            if metadata.get("question_numbers"):
                score += 0.1

            return min(1.0, score)  # Cap at 1.0

        except Exception:
            return 0.5

    async def _combine_and_rank_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        search_query: SearchQuery,
    ) -> List[SearchResult]:
        """Combine and rank results from different search methods"""
        try:
            # Create a dictionary to deduplicate results
            result_dict = {}

            # Add semantic results with higher weight
            for result in semantic_results:
                key = result.id
                if key not in result_dict:
                    result_dict[key] = result
                else:
                    # If duplicate, keep the one with higher score
                    if result.score > result_dict[key].score:
                        result_dict[key] = result

            # Add keyword results
            for result in keyword_results:
                key = result.id
                if key not in result_dict:
                    result_dict[key] = result
                else:
                    # Combine scores for hybrid results
                    existing = result_dict[key]
                    combined_score = (existing.score + result.score) / 2
                    existing.score = combined_score

            # Convert back to list and apply ranking
            combined_results = list(result_dict.values())

            # Calculate composite scores
            for result in combined_results:
                composite_score = (
                    result.score * 0.4  # Base similarity score
                    + result.relevance_score * 0.3  # Relevance
                    + result.recency_score * 0.2  # Recency
                    + result.importance_score * 0.1  # Importance
                )

                # Apply boosters
                if (
                    search_query.boost_interaction
                    and result.interaction_id == search_query.interaction_id
                ):
                    composite_score *= 1.5

                if search_query.boost_recent and result.recency_score > 0.8:
                    composite_score *= 1.2

                result.score = composite_score

            # Sort by composite score
            combined_results.sort(key=lambda x: x.score, reverse=True)

            return combined_results

        except Exception as e:
            print(f"❌ Error combining results: {e}")
            return semantic_results + keyword_results

    async def _apply_final_filters(
        self, results: List[SearchResult], search_query: SearchQuery
    ) -> List[SearchResult]:
        """Apply final filters and limit results"""
        try:
            filtered_results = []

            for result in results:
                # Apply minimum score filter
                if result.score < search_query.min_score:
                    continue

                # Apply topic filters
                if search_query.topic_filters:
                    if not any(
                        topic in result.topic_tags
                        for topic in search_query.topic_filters
                    ):
                        continue

                # Apply time range filter
                if search_query.time_range:
                    # This would require timestamp in metadata
                    # For now, skip this filter
                    pass

                filtered_results.append(result)

            # Limit to top_k results
            return filtered_results[: search_query.top_k]

        except Exception as e:
            print(f"❌ Error applying final filters: {e}")
            return results[: search_query.top_k]

    async def get_enhanced_context(
        self,
        user_id: str,
        interaction_id: Optional[str],
        message: str,
        context_types: List[str] = None,
    ) -> Dict[str, Any]:
        """Get enhanced context using optimized vector search"""
        try:
            if context_types is None:
                context_types = [
                    "semantic_summary",
                    "vector_search",
                    "document_content",
                    "cross_interaction",
                ]

            context_data = {}

            # Vector search context
            if "vector_search" in context_types:
                search_query = SearchQuery(
                    query=message,
                    user_id=user_id,
                    interaction_id=interaction_id,
                    top_k=8,
                    use_hybrid_search=True,
                    use_query_expansion=True,
                    boost_recent=True,
                    boost_interaction=True,
                )

                vector_results = await self.hybrid_search(search_query)
                context_data["vector_search"] = {
                    "results": vector_results,
                    "count": len(vector_results),
                    "search_type": "hybrid",
                }

            return context_data

        except Exception as e:
            print(f"❌ Error getting enhanced context: {e}")
            return {}

    async def optimize_collection_indexes(self) -> bool:
        """Optimize collection indexes for better performance"""
        try:
            if not self.vector_store:
                return False

            # Get collection
            collection = Collection(self.collection_name)

            # Check if collection exists
            if not utility.has_collection(self.collection_name):
                return False

            # Load collection
            collection.load()

            # Create additional indexes for better filtering performance
            try:
                # Index on user_id for faster user filtering
                collection.create_index(
                    field_name="user_id",
                    index_params={
                        "index_type": "STL_SORT",
                        "metric_type": "L2",
                    },
                )
            except Exception:
                pass  # Index might already exist

            try:
                # Index on interaction_id for faster interaction filtering
                collection.create_index(
                    field_name="interaction_id",
                    index_params={
                        "index_type": "STL_SORT",
                        "metric_type": "L2",
                    },
                )
            except Exception:
                pass  # Index might already exist

            print("✅ Collection indexes optimized")
            return True

        except Exception as e:
            print(f"❌ Error optimizing collection indexes: {e}")
            return False

    async def get_search_analytics(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get search analytics for monitoring and optimization"""
        try:
            # This would require additional logging and analytics collection
            # For now, return basic structure
            return {
                "user_id": user_id,
                "period_days": days,
                "total_searches": 0,
                "average_response_time": 0.0,
                "cache_hit_rate": 0.0,
                "most_searched_topics": [],
                "search_success_rate": 0.0,
            }

        except Exception as e:
            print(f"❌ Error getting search analytics: {e}")
            return {}


# Global instance
vector_optimization_service = VectorOptimizationService()
