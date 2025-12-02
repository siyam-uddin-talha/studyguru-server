"""
Cache Manager for StudyGuru - Handles both response and context caching
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import hashlib
import json

from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache, get_llm_cache

from app.core.config import settings


class StudyGuruCacheManager:
    """Enhanced cache manager for StudyGuru with both response and context caching"""

    def __init__(self):
        self.response_cache = None
        self.context_cache_manager = None
        self.cached_contents: Dict[str, Any] = {}
        self._initialize_caches()

    def _initialize_caches(self):
        """Initialize both response and context caches"""
        if settings.ENABLE_MODEL_CACHING:
            # Initialize response cache
            if settings.ENVIRONMENT == "production":
                # Use SQLite cache for production
                self.response_cache = SQLiteCache(database_path="studyguru_cache.db")
            else:
                # Use in-memory cache for development
                self.response_cache = InMemoryCache()

            # Set global cache
            set_llm_cache(self.response_cache)

        if settings.ENABLE_CONTEXT_CACHING:
            # Context caching will be handled by the model's cache_context parameter
            # Context caching is supported for Gemini models when model_name is provided
            # No separate cache manager needed for Gemini context caching
            self.context_cache_manager = None

    def get_response_cache(self):
        """Get the response cache instance"""
        return self.response_cache if settings.ENABLE_MODEL_CACHING else None

    def create_cached_content(
        self, model: str, contents: List[Any], ttl_hours: int = 24
    ) -> Optional[Any]:
        """
        Create cached content for Google AI models

        Args:
            model: The model name (e.g., "gemini-2.5-pro")
            contents: List of content to cache (text, images, etc.)
            ttl_hours: Time to live in hours

        Returns:
            CachedContent object or None if caching is disabled
        """
        if not settings.ENABLE_CONTEXT_CACHING:
            return None

        try:
            # Create cache key based on content hash
            content_hash = self._generate_content_hash(contents)
            cache_key = f"{model}_{content_hash}"

            # Check if already cached
            if cache_key in self.cached_contents:
                cached_item = self.cached_contents[cache_key]
                if self._is_cache_valid(cached_item, ttl_hours):
                    return cached_item["content"]

            # For Gemini models, we'll use the contents directly as cached content
            # The actual context caching will be handled by the model's cache_context parameter
            cached_content = {
                "model": model,
                "contents": contents,
                "cache_key": cache_key,
            }

            # Store in our cache registry
            self.cached_contents[cache_key] = {
                "content": cached_content,
                "created_at": datetime.now(),
                "ttl_hours": ttl_hours,
            }

            return cached_content

        except Exception as e:
            print(f"Warning: Failed to create cached content: {e}")
            return None

    def get_cached_content(self, model: str, contents: List[Any]) -> Optional[Any]:
        """
        Retrieve cached content if available

        Args:
            model: The model name
            contents: List of content to find cache for

        Returns:
            CachedContent object or None if not found/expired
        """
        if not settings.ENABLE_CONTEXT_CACHING:
            return None

        content_hash = self._generate_content_hash(contents)
        cache_key = f"{model}_{content_hash}"

        if cache_key in self.cached_contents:
            cached_item = self.cached_contents[cache_key]
            if self._is_cache_valid(cached_item, cached_item["ttl_hours"]):
                return cached_item["content"]
            else:
                # Remove expired cache
                del self.cached_contents[cache_key]

        return None

    def _generate_content_hash(self, contents: List[Any]) -> str:
        """Generate hash for content to use as cache key"""
        content_str = json.dumps(contents, sort_keys=True, default=str)
        return hashlib.md5(content_str.encode()).hexdigest()

    def _is_cache_valid(self, cached_item: Dict[str, Any], ttl_hours: int) -> bool:
        """Check if cached item is still valid"""
        created_at = cached_item["created_at"]
        expiry_time = created_at + timedelta(hours=ttl_hours)
        return datetime.now() < expiry_time

    def clear_expired_caches(self):
        """Clear expired cached contents"""
        current_time = datetime.now()
        expired_keys = []

        for key, cached_item in self.cached_contents.items():
            if not self._is_cache_valid(cached_item, cached_item["ttl_hours"]):
                expired_keys.append(key)

        for key in expired_keys:
            del self.cached_contents[key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "response_cache_enabled": settings.ENABLE_MODEL_CACHING,
            "context_cache_enabled": settings.ENABLE_CONTEXT_CACHING,
            "cached_contents_count": len(self.cached_contents),
            "cache_ttl_hours": settings.CACHE_TTL // 3600,
            "cache_max_size": settings.CACHE_MAX_SIZE,
        }


# Global cache manager instance
cache_manager = StudyGuruCacheManager()
