"""
Simple in-memory cache service for StudyGuru
"""

import asyncio
import time
from typing import Any, Optional, Dict
from functools import wraps


class SimpleCache:
    """Simple in-memory cache with TTL support"""

    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                return None

            return entry["value"]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            expires_at = time.time() + (ttl or self._default_ttl)
            self._cache[key] = {"value": value, "expires_at": expires_at}

    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self) -> None:
        """Remove expired entries"""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if current_time > entry["expires_at"]
            ]
            for key in expired_keys:
                del self._cache[key]


# Global cache instance
cache_service = SimpleCache(default_ttl=300)


def cached(ttl: int = 300, key_prefix: str = ""):
    """Decorator for caching function results"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{key_prefix}{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Try to get from cache
            cached_result = await cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_service.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator


async def cache_user_context(
    user_id: str, context_data: Dict[str, Any], ttl: int = 600
):
    """Cache user context data"""
    key = f"user_context:{user_id}"
    await cache_service.set(key, context_data, ttl)


async def get_cached_user_context(user_id: str) -> Optional[Dict[str, Any]]:
    """Get cached user context data"""
    key = f"user_context:{user_id}"
    return await cache_service.get(key)


async def cache_interaction_data(
    interaction_id: str, data: Dict[str, Any], ttl: int = 300
):
    """Cache interaction data"""
    key = f"interaction:{interaction_id}"
    await cache_service.set(key, data, ttl)


async def get_cached_interaction_data(interaction_id: str) -> Optional[Dict[str, Any]]:
    """Get cached interaction data"""
    key = f"interaction:{interaction_id}"
    return await cache_service.get(key)


async def invalidate_user_cache(user_id: str):
    """Invalidate all cache entries for a user"""
    # This is a simple implementation - in production you might want more sophisticated invalidation
    await cache_service.delete(f"user_context:{user_id}")


async def invalidate_interaction_cache(interaction_id: str):
    """Invalidate cache entries for an interaction"""
    await cache_service.delete(f"interaction:{interaction_id}")
