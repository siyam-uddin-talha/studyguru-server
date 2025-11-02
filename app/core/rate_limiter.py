"""
High-Performance Rate Limiter for StudyGuru Pro
Supports both Redis (distributed) and in-memory (single instance) backends
Uses sliding window algorithm for accurate rate limiting
"""

import time
import asyncio
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib

try:
    import redis.asyncio as redis  # type: ignore

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore


class RateLimitBackend(ABC):
    """Abstract base class for rate limit backends"""

    @abstractmethod
    async def is_rate_limited(
        self, key: str, limit: int, window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a key is rate limited

        Args:
            key: Unique identifier for the rate limit (e.g., user_id, ip_address)
            limit: Maximum number of requests allowed
            window: Time window in seconds

        Returns:
            Tuple of (is_limited, info_dict)
            info_dict contains: current_count, limit, reset_time, retry_after
        """
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup expired entries"""
        pass


class RedisRateLimitBackend(RateLimitBackend):
    """
    Redis-based rate limiter using sliding window algorithm
    Provides distributed rate limiting across multiple server instances
    """

    def __init__(self, redis_url: str):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for Redis backend")

        self.redis = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
            socket_keepalive=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

    async def is_rate_limited(
        self, key: str, limit: int, window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Sliding window rate limiting using Redis sorted sets
        More accurate than fixed window, prevents burst attacks
        """
        now = time.time()
        window_start = now - window

        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()

        try:
            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)

            # Count requests in current window
            pipe.zcard(key)

            # Add current request with score as timestamp
            pipe.zadd(key, {str(now): now})

            # Set expiry for the key
            pipe.expire(key, window + 1)

            # Execute pipeline
            results = await pipe.execute()

            # Get count before adding current request
            current_count = results[1]

            # Check if rate limit exceeded
            is_limited = current_count >= limit

            # Calculate reset time
            reset_time = int(now + window)
            retry_after = window if is_limited else 0

            return is_limited, {
                "current_count": current_count + 1,
                "limit": limit,
                "reset_time": reset_time,
                "retry_after": retry_after,
                "window": window,
            }

        except Exception as e:
            # If Redis fails, allow the request (fail open)
            return False, {
                "current_count": 0,
                "limit": limit,
                "reset_time": int(now + window),
                "retry_after": 0,
                "window": window,
                "error": str(e),
            }

    async def cleanup(self):
        """Close Redis connection"""
        await self.redis.close()


class InMemoryRateLimitBackend(RateLimitBackend):
    """
    In-memory rate limiter using sliding window algorithm
    Suitable for single-instance deployments
    Optimized for high performance with minimal memory overhead
    """

    def __init__(self):
        # Dictionary to store request timestamps
        # Format: {key: [(timestamp1, timestamp2, ...)]}
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()

        # Start cleanup task
        self._cleanup_task = None

    async def is_rate_limited(
        self, key: str, limit: int, window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Sliding window rate limiting using in-memory storage
        """
        now = time.time()
        window_start = now - window

        async with self._lock:
            # Get existing requests
            requests = self._requests[key]

            # Remove old entries outside the window
            # Use list comprehension for better performance
            requests = [ts for ts in requests if ts > window_start]
            self._requests[key] = requests

            # Count requests in current window
            current_count = len(requests)

            # Check if rate limit exceeded
            is_limited = current_count >= limit

            # Add current request if not limited
            if not is_limited:
                requests.append(now)

            # Calculate reset time
            reset_time = int(now + window)
            retry_after = window if is_limited else 0

            return is_limited, {
                "current_count": current_count + (0 if is_limited else 1),
                "limit": limit,
                "reset_time": reset_time,
                "retry_after": retry_after,
                "window": window,
            }

    async def cleanup(self):
        """Remove expired entries from memory"""
        now = time.time()

        async with self._lock:
            # Remove empty keys and old entries
            keys_to_remove = []

            for key, requests in self._requests.items():
                # Remove requests older than 1 hour (max window + buffer)
                cutoff = now - 3600
                requests = [ts for ts in requests if ts > cutoff]

                if not requests:
                    keys_to_remove.append(key)
                else:
                    self._requests[key] = requests

            for key in keys_to_remove:
                del self._requests[key]


class RateLimiter:
    """
    High-performance rate limiter with multiple strategies
    """

    def __init__(
        self,
        backend: RateLimitBackend,
        default_limit: int = 100,
        default_window: int = 60,
    ):
        """
        Initialize rate limiter

        Args:
            backend: Rate limit backend (Redis or In-Memory)
            default_limit: Default requests per window
            default_window: Default time window in seconds
        """
        self.backend = backend
        self.default_limit = default_limit
        self.default_window = default_window

        # Route-specific limits
        self._route_limits: Dict[str, Tuple[int, int]] = {}

        # Start periodic cleanup for in-memory backend
        if isinstance(backend, InMemoryRateLimitBackend):
            self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task for in-memory backend"""

        async def cleanup_loop():
            while True:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self.backend.cleanup()

        # Create background task
        asyncio.create_task(cleanup_loop())

    def set_route_limit(self, route: str, limit: int, window: int):
        """
        Set custom rate limit for a specific route

        Args:
            route: Route pattern (e.g., "/api/app/interactions")
            limit: Maximum requests allowed
            window: Time window in seconds
        """
        self._route_limits[route] = (limit, window)

    def _get_route_limit(self, route: str) -> Tuple[int, int]:
        """Get limit and window for a specific route"""
        # Check for exact match
        if route in self._route_limits:
            return self._route_limits[route]

        # Check for pattern match (e.g., /api/*)
        for pattern, (limit, window) in self._route_limits.items():
            if pattern.endswith("/*") and route.startswith(pattern[:-2]):
                return limit, window

        return self.default_limit, self.default_window

    async def check_rate_limit(
        self,
        identifier: str,
        route: str = "",
        limit: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request should be rate limited

        Args:
            identifier: Unique identifier (user_id, ip_address, etc.)
            route: Route path for route-specific limits
            limit: Override default limit
            window: Override default window

        Returns:
            Tuple of (is_limited, info_dict)
        """
        # Get limit and window
        if limit is None or window is None:
            if route:
                limit, window = self._get_route_limit(route)
            else:
                limit, window = self.default_limit, self.default_window

        # Create rate limit key
        key_parts = ["rate_limit", identifier]
        if route:
            # Hash route to keep key size manageable
            route_hash = hashlib.md5(route.encode()).hexdigest()[:8]
            key_parts.append(route_hash)

        key = ":".join(key_parts)

        # Check rate limit
        return await self.backend.is_rate_limited(key, limit, window)

    async def cleanup(self):
        """Cleanup backend resources"""
        await self.backend.cleanup()


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def init_rate_limiter(
    redis_url: Optional[str] = None,
    default_limit: int = 100,
    default_window: int = 60,
) -> RateLimiter:
    """
    Initialize global rate limiter

    Args:
        redis_url: Redis URL (if None, uses in-memory backend)
        default_limit: Default requests per window
        default_window: Default time window in seconds

    Returns:
        RateLimiter instance
    """
    global _rate_limiter

    # Choose backend
    if redis_url and REDIS_AVAILABLE:
        backend = RedisRateLimitBackend(redis_url)
    else:
        backend = InMemoryRateLimitBackend()

    _rate_limiter = RateLimiter(
        backend=backend,
        default_limit=default_limit,
        default_window=default_window,
    )

    return _rate_limiter


def get_rate_limiter() -> Optional[RateLimiter]:
    """Get the global rate limiter instance"""
    return _rate_limiter
