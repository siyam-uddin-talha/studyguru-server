"""
Rate Limiting Middleware for FastAPI
Provides flexible rate limiting with minimal performance overhead
"""

import time
from typing import Optional, Callable, Awaitable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.rate_limiter import get_rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI

    Features:
    - Per-user and per-IP rate limiting
    - Route-specific limits
    - Informative headers (X-RateLimit-*)
    - Graceful error handling
    """

    def __init__(
        self,
        app: ASGIApp,
        exempt_paths: Optional[list] = None,
        identifier_func: Optional[Callable] = None,
    ):
        """
        Initialize rate limit middleware

        Args:
            app: FastAPI application
            exempt_paths: List of paths to exempt from rate limiting
            identifier_func: Custom function to extract identifier from request
        """
        super().__init__(app)
        self.exempt_paths = exempt_paths or ["/", "/docs", "/redoc", "/openapi.json"]
        self.identifier_func = identifier_func or self._default_identifier

    def _default_identifier(self, request: Request) -> str:
        """
        Default identifier extraction
        Priority: user_id > API key > IP address
        """
        # Try to get user_id from request state (set by auth middleware)
        if hasattr(request.state, "user_id") and request.state.user_id:
            return f"user:{request.state.user_id}"

        # Try to get API key from headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key}"

        # Fall back to IP address
        # Handle proxied requests
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    def _is_exempt(self, path: str) -> bool:
        """Check if path is exempt from rate limiting"""
        # Exact match
        if path in self.exempt_paths:
            return True

        # Pattern match (e.g., /health/*)
        for exempt_path in self.exempt_paths:
            if exempt_path.endswith("/*") and path.startswith(exempt_path[:-2]):
                return True

        return False

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Process request with rate limiting
        """
        # Check if path is exempt
        if self._is_exempt(request.url.path):
            return await call_next(request)

        # Get rate limiter
        rate_limiter = get_rate_limiter()
        if not rate_limiter:
            # Rate limiter not initialized, allow request
            return await call_next(request)

        # Get identifier
        try:
            identifier = self.identifier_func(request)
        except Exception:
            # If identifier extraction fails, allow request
            return await call_next(request)

        # Check rate limit
        try:
            is_limited, info = await rate_limiter.check_rate_limit(
                identifier=identifier,
                route=request.url.path,
            )
        except Exception:
            # If rate limiting fails, allow request (fail open)
            return await call_next(request)

        # Add rate limit headers to response
        headers = {
            "X-RateLimit-Limit": str(info["limit"]),
            "X-RateLimit-Remaining": str(max(0, info["limit"] - info["current_count"])),
            "X-RateLimit-Reset": str(info["reset_time"]),
        }

        if is_limited:
            # Request is rate limited
            headers["Retry-After"] = str(info["retry_after"])

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                    "limit": info["limit"],
                    "window": info["window"],
                    "retry_after": info["retry_after"],
                    "reset_time": info["reset_time"],
                },
                headers=headers,
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to successful response
        for key, value in headers.items():
            response.headers[key] = value

        return response


# Decorator for route-specific rate limiting
def rate_limit(limit: int, window: int = 60):
    """
    Decorator for route-specific rate limiting

    Usage:
        @app.get("/api/expensive-operation")
        @rate_limit(limit=10, window=60)  # 10 requests per minute
        async def expensive_operation():
            ...

    Args:
        limit: Maximum requests allowed
        window: Time window in seconds
    """

    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            rate_limiter = get_rate_limiter()
            if not rate_limiter:
                return await func(request, *args, **kwargs)

            # Get identifier
            if hasattr(request.state, "user_id") and request.state.user_id:
                identifier = f"user:{request.state.user_id}"
            else:
                forwarded = request.headers.get("X-Forwarded-For")
                if forwarded:
                    ip = forwarded.split(",")[0].strip()
                else:
                    ip = request.client.host if request.client else "unknown"
                identifier = f"ip:{ip}"

            # Check rate limit
            is_limited, info = await rate_limiter.check_rate_limit(
                identifier=identifier,
                route=request.url.path,
                limit=limit,
                window=window,
            )

            if is_limited:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": "Too many requests. Please try again later.",
                        "limit": info["limit"],
                        "window": info["window"],
                        "retry_after": info["retry_after"],
                        "reset_time": info["reset_time"],
                    },
                    headers={
                        "Retry-After": str(info["retry_after"]),
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(info["reset_time"]),
                    },
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


# Utility function to get identifier from request
def get_request_identifier(request: Request) -> str:
    """
    Get rate limit identifier from request
    Priority: user_id > API key > IP address
    """
    # Try to get user_id from request state
    if hasattr(request.state, "user_id") and request.state.user_id:
        return f"user:{request.state.user_id}"

    # Try to get API key from headers
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"

    # Fall back to IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"

    return f"ip:{ip}"
