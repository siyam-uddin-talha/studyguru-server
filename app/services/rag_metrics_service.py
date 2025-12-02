"""
RAG System Metrics Service

Simple metrics tracking for monitoring and optimization of the streamlined RAG system.

Key Metrics:
1. Retrieval Latency - Should be < 1.5 seconds
2. Context Relevance - User satisfaction scores
3. Response Quality - Accuracy on test questions
4. Cache Hit Rate - Should be > 40%
5. Error Rate - Should be < 1%
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

from app.services.cache_service import cache_service


@dataclass
class RetrievalMetric:
    """Single retrieval metric record"""

    retrieval_latency_ms: float
    results_count: int
    context_chars: int
    query_type: str  # "question_specific" or "general"
    cache_hit: bool
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RAGMetricsService:
    """
    Simple metrics tracking for RAG system optimization

    Collects:
    - Retrieval latency (target: < 1500ms)
    - Cache hit rate (target: > 40%)
    - Error rate (target: < 1%)
    - Context relevance scores
    """

    def __init__(self):
        self._metrics_buffer: List[RetrievalMetric] = []
        self._buffer_max_size = 1000
        self._cache_key_prefix = "rag_metrics"

        # In-memory aggregates for quick stats
        self._total_requests = 0
        self._cache_hits = 0
        self._errors = 0
        self._total_latency_ms = 0.0

    async def record_retrieval(
        self,
        retrieval_time: float,
        num_results: int,
        context_length: int,
        query_type: str,
        cache_hit: bool = False,
        success: bool = True,
    ) -> None:
        """
        Record a retrieval operation metric

        Args:
            retrieval_time: Time taken in seconds
            num_results: Number of results returned
            context_length: Total context length in characters
            query_type: "question_specific" or "general"
            cache_hit: Whether result was from cache
            success: Whether retrieval was successful
        """
        latency_ms = retrieval_time * 1000

        metric = RetrievalMetric(
            retrieval_latency_ms=latency_ms,
            results_count=num_results,
            context_chars=context_length,
            query_type=query_type,
            cache_hit=cache_hit,
            success=success,
        )

        # Update in-memory aggregates
        self._total_requests += 1
        self._total_latency_ms += latency_ms
        if cache_hit:
            self._cache_hits += 1
        if not success:
            self._errors += 1

        # Add to buffer
        self._metrics_buffer.append(metric)

        # Flush if buffer is full
        if len(self._metrics_buffer) >= self._buffer_max_size:
            await self._flush_metrics()

        # Log slow retrievals
        if latency_ms > 1500:
            print(
                f"⚠️ [RAG METRICS] Slow retrieval: {latency_ms:.0f}ms (target: <1500ms)"
            )

    async def _flush_metrics(self) -> None:
        """Flush metrics buffer to cache for persistence"""
        if not self._metrics_buffer:
            return

        try:
            # Get current hour for bucketing
            hour_key = datetime.utcnow().strftime("%Y%m%d_%H")
            cache_key = f"{self._cache_key_prefix}:{hour_key}"

            # Get existing metrics for this hour
            existing = await cache_service.get(cache_key) or {"metrics": []}

            # Add new metrics
            for metric in self._metrics_buffer:
                existing["metrics"].append(
                    {
                        "latency_ms": metric.retrieval_latency_ms,
                        "results": metric.results_count,
                        "chars": metric.context_chars,
                        "type": metric.query_type,
                        "cache_hit": metric.cache_hit,
                        "success": metric.success,
                        "ts": metric.timestamp.isoformat(),
                    }
                )

            # Save with 24-hour TTL
            await cache_service.set(cache_key, existing, ttl=86400)

            # Clear buffer
            self._metrics_buffer.clear()

        except Exception as e:
            print(f"⚠️ [RAG METRICS] Flush error: {e}")

    def get_quick_stats(self) -> Dict[str, Any]:
        """
        Get quick in-memory stats (no async, for debugging)

        Returns:
            Dict with current session stats
        """
        if self._total_requests == 0:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0,
                "cache_hit_rate": 0,
                "error_rate": 0,
                "status": "no_data",
            }

        avg_latency = self._total_latency_ms / self._total_requests
        cache_hit_rate = (self._cache_hits / self._total_requests) * 100
        error_rate = (self._errors / self._total_requests) * 100

        # Determine health status
        status = "healthy"
        if avg_latency > 1500:
            status = "slow"
        if error_rate > 1:
            status = "degraded"
        if cache_hit_rate < 40:
            status = "cache_miss_high"

        return {
            "total_requests": self._total_requests,
            "avg_latency_ms": round(avg_latency, 2),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "error_rate": round(error_rate, 2),
            "status": status,
        }

    async def get_hourly_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get aggregated stats for the last N hours

        Args:
            hours: Number of hours to look back

        Returns:
            Dict with aggregated metrics
        """
        try:
            all_metrics = []
            now = datetime.utcnow()

            for i in range(hours):
                hour = now - timedelta(hours=i)
                hour_key = hour.strftime("%Y%m%d_%H")
                cache_key = f"{self._cache_key_prefix}:{hour_key}"

                data = await cache_service.get(cache_key)
                if data and "metrics" in data:
                    all_metrics.extend(data["metrics"])

            if not all_metrics:
                return self.get_quick_stats()

            # Aggregate metrics
            total = len(all_metrics)
            total_latency = sum(m["latency_ms"] for m in all_metrics)
            cache_hits = sum(1 for m in all_metrics if m.get("cache_hit", False))
            errors = sum(1 for m in all_metrics if not m.get("success", True))
            question_specific = sum(
                1 for m in all_metrics if m.get("type") == "question_specific"
            )

            avg_latency = total_latency / total if total > 0 else 0
            cache_hit_rate = (cache_hits / total) * 100 if total > 0 else 0
            error_rate = (errors / total) * 100 if total > 0 else 0

            return {
                "period_hours": hours,
                "total_requests": total,
                "avg_latency_ms": round(avg_latency, 2),
                "cache_hit_rate": round(cache_hit_rate, 2),
                "error_rate": round(error_rate, 2),
                "question_specific_rate": (
                    round((question_specific / total) * 100, 2) if total > 0 else 0
                ),
                "targets": {
                    "latency": "< 1500ms",
                    "cache_hit_rate": "> 40%",
                    "error_rate": "< 1%",
                },
                "status": self._calculate_health_status(
                    avg_latency, cache_hit_rate, error_rate
                ),
            }

        except Exception as e:
            print(f"⚠️ [RAG METRICS] Stats error: {e}")
            return self.get_quick_stats()

    def _calculate_health_status(
        self,
        avg_latency: float,
        cache_hit_rate: float,
        error_rate: float,
    ) -> str:
        """Calculate overall health status"""
        issues = []

        if avg_latency > 1500:
            issues.append("slow_retrieval")
        if cache_hit_rate < 40:
            issues.append("low_cache_hit")
        if error_rate > 1:
            issues.append("high_error_rate")

        if not issues:
            return "healthy"
        elif len(issues) == 1:
            return issues[0]
        else:
            return "degraded"

    def reset_stats(self) -> None:
        """Reset in-memory stats (for testing)"""
        self._total_requests = 0
        self._cache_hits = 0
        self._errors = 0
        self._total_latency_ms = 0.0
        self._metrics_buffer.clear()


# Global instance
rag_metrics_service = RAGMetricsService()


async def track_retrieval_metrics(
    retrieval_time: float,
    num_results: int,
    context_length: int,
    query_type: str,
    cache_hit: bool = False,
    success: bool = True,
) -> None:
    """
    Convenience function for tracking retrieval metrics

    Usage:
        await track_retrieval_metrics(
            retrieval_time=1.2,
            num_results=5,
            context_length=3500,
            query_type="question_specific"
        )
    """
    await rag_metrics_service.record_retrieval(
        retrieval_time=retrieval_time,
        num_results=num_results,
        context_length=context_length,
        query_type=query_type,
        cache_hit=cache_hit,
        success=success,
    )
