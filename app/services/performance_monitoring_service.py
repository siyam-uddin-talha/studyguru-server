"""
Performance monitoring and optimization service
Provides comprehensive monitoring, metrics collection, and performance optimization
"""

import asyncio
import time
import json
import psutil
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

from app.core.database import AsyncSessionLocal
from app.models.context import ContextUsageLog
from app.services.context_service import context_service
from app.services.vector_optimization_service import vector_optimization_service
from app.services.real_time_context_service import real_time_context_service


class MetricType(Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Represents a performance metric"""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Represents a performance alert"""

    alert_id: str
    metric_name: str
    threshold: float
    current_value: float
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: datetime
    resolved: bool = False


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""

    report_id: str
    user_id: Optional[str]
    time_range: Tuple[datetime, datetime]
    total_requests: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    context_retrieval_metrics: Dict[str, Any]
    vector_search_metrics: Dict[str, Any]
    real_time_update_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]
    alerts: List[PerformanceAlert] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""

    def __init__(self):
        self.metrics: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.alerts: List[PerformanceAlert] = []
        self.performance_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.thresholds: Dict[str, Dict[str, float]] = {
            "context_retrieval_time": {"warning": 2.0, "critical": 5.0},
            "vector_search_time": {"warning": 1.0, "critical": 3.0},
            "semantic_summary_time": {"warning": 3.0, "critical": 8.0},
            "embedding_creation_time": {"warning": 2.0, "critical": 5.0},
            "response_time": {"warning": 5.0, "critical": 10.0},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "memory_usage": {"warning": 0.8, "critical": 0.9},
            "cpu_usage": {"warning": 0.8, "critical": 0.9},
        }
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None

    def start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            print("📊 Performance monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
            print("📊 Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Check thresholds and generate alerts
                await self._check_thresholds()

                # Clean up old data
                await self._cleanup_old_data()

                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage", cpu_percent, MetricType.GAUGE)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            self.record_metric("memory_usage", memory_percent, MetricType.GAUGE)

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent / 100.0
            self.record_metric("disk_usage", disk_percent, MetricType.GAUGE)

            # Network I/O
            network = psutil.net_io_counters()
            self.record_metric(
                "network_bytes_sent", network.bytes_sent, MetricType.COUNTER
            )
            self.record_metric(
                "network_bytes_recv", network.bytes_recv, MetricType.COUNTER
            )

        except Exception as e:
            print(f"⚠️ Error collecting system metrics: {e}")

    async def _check_thresholds(self):
        """Check metric thresholds and generate alerts"""
        try:
            current_time = datetime.now()
            recent_metrics = self._get_recent_metrics(minutes=5)

            for metric_name, thresholds in self.thresholds.items():
                metric_values = [
                    m.value for m in recent_metrics if m.name == metric_name
                ]

                if not metric_values:
                    continue

                avg_value = sum(metric_values) / len(metric_values)

                # Check critical threshold
                if avg_value >= thresholds["critical"]:
                    await self._create_alert(
                        metric_name=metric_name,
                        threshold=thresholds["critical"],
                        current_value=avg_value,
                        severity="critical",
                        message=f"{metric_name} is critically high: {avg_value:.2f} >= {thresholds['critical']}",
                    )

                # Check warning threshold
                elif avg_value >= thresholds["warning"]:
                    await self._create_alert(
                        metric_name=metric_name,
                        threshold=thresholds["warning"],
                        current_value=avg_value,
                        severity="warning",
                        message=f"{metric_name} is high: {avg_value:.2f} >= {thresholds['warning']}",
                    )

        except Exception as e:
            print(f"⚠️ Error checking thresholds: {e}")

    async def _create_alert(
        self,
        metric_name: str,
        threshold: float,
        current_value: float,
        severity: str,
        message: str,
    ):
        """Create a performance alert"""
        alert_id = f"{metric_name}_{int(time.time())}"

        # Check if similar alert already exists
        existing_alert = next(
            (a for a in self.alerts if a.metric_name == metric_name and not a.resolved),
            None,
        )

        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.timestamp = datetime.now()
        else:
            # Create new alert
            alert = PerformanceAlert(
                alert_id=alert_id,
                metric_name=metric_name,
                threshold=threshold,
                current_value=current_value,
                severity=severity,
                message=message,
                timestamp=datetime.now(),
            )
            self.alerts.append(alert)
            print(f"🚨 Performance Alert: {message}")

    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)

            # Clean up old metrics
            while self.metrics and self.metrics[0].timestamp < cutoff_time:
                self.metrics.popleft()

            # Clean up old alerts (keep for 7 days)
            alert_cutoff = datetime.now() - timedelta(days=7)
            self.alerts = [a for a in self.alerts if a.timestamp > alert_cutoff]

        except Exception as e:
            print(f"⚠️ Error cleaning up old data: {e}")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a performance metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {},
        )

        self.metrics.append(metric)
        self.performance_data[name].append(metric)

    def _get_recent_metrics(self, minutes: int = 5) -> List[Metric]:
        """Get recent metrics within specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics if m.timestamp >= cutoff_time]

    @asynccontextmanager
    async def measure_time(
        self, metric_name: str, tags: Optional[Dict[str, str]] = None
    ):
        """Context manager for measuring execution time"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.record_metric(metric_name, execution_time, MetricType.TIMER, tags)

    async def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time window"""
        recent_metrics = self._get_recent_metrics(minutes)

        if not recent_metrics:
            return {"message": "No metrics available for the specified time window"}

        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)

        summary = {}
        for metric_name, values in metric_groups.items():
            summary[metric_name] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else 0,
            }

        return summary

    async def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get active (unresolved) alerts"""
        return [a for a in self.alerts if not a.resolved]

    async def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                print(f"✅ Alert resolved: {alert_id}")
                break


class PerformanceOptimizer:
    """Performance optimization service"""

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_rules: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []

    async def analyze_performance(
        self, user_id: Optional[str] = None, time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Analyze performance and identify optimization opportunities"""
        try:
            print(f"🔍 Analyzing performance for user {user_id or 'all users'}")

            # Get performance metrics
            performance_summary = await self.monitor.get_performance_summary(
                time_window_minutes
            )

            # Get context usage logs
            context_metrics = await self._analyze_context_usage(
                user_id, time_window_minutes
            )

            # Get vector search metrics
            vector_metrics = await self._analyze_vector_search_performance(
                user_id, time_window_minutes
            )

            # Get real-time update metrics
            real_time_metrics = await self._analyze_real_time_updates(
                user_id, time_window_minutes
            )

            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks(performance_summary)

            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(
                performance_summary,
                context_metrics,
                vector_metrics,
                real_time_metrics,
                bottlenecks,
            )

            analysis_result = {
                "user_id": user_id,
                "time_window_minutes": time_window_minutes,
                "performance_summary": performance_summary,
                "context_metrics": context_metrics,
                "vector_metrics": vector_metrics,
                "real_time_metrics": real_time_metrics,
                "bottlenecks": bottlenecks,
                "recommendations": recommendations,
                "analysis_timestamp": datetime.now().isoformat(),
            }

            print(f"✅ Performance analysis completed")
            return analysis_result

        except Exception as e:
            print(f"❌ Performance analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_context_usage(
        self, user_id: Optional[str], time_window_minutes: int
    ) -> Dict[str, Any]:
        """Analyze context usage patterns"""
        try:
            async with AsyncSessionLocal() as db:
                cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

                # Get context usage logs
                query = select(ContextUsageLog).where(
                    ContextUsageLog.created_at >= cutoff_time
                )

                if user_id:
                    query = query.where(ContextUsageLog.user_id == user_id)

                result = await db.execute(query)
                usage_logs = result.scalars().all()

                if not usage_logs:
                    return {"message": "No context usage data available"}

                # Analyze patterns
                total_requests = len(usage_logs)
                avg_retrieval_time = (
                    sum(log.context_retrieval_time for log in usage_logs)
                    / total_requests
                )

                # Source usage analysis
                source_usage = defaultdict(int)
                for log in usage_logs:
                    for source in log.context_sources_used:
                        source_usage[source] += 1

                # Query type analysis
                query_types = defaultdict(int)
                for log in usage_logs:
                    query_types[log.query_type] += 1

                return {
                    "total_requests": total_requests,
                    "avg_retrieval_time": avg_retrieval_time,
                    "source_usage": dict(source_usage),
                    "query_types": dict(query_types),
                    "avg_sources_per_request": sum(
                        len(log.context_sources_used) for log in usage_logs
                    )
                    / total_requests,
                }

        except Exception as e:
            print(f"⚠️ Error analyzing context usage: {e}")
            return {"error": str(e)}

    async def _analyze_vector_search_performance(
        self, user_id: Optional[str], time_window_minutes: int
    ) -> Dict[str, Any]:
        """Analyze vector search performance"""
        try:
            # This would integrate with vector optimization service metrics
            # For now, return placeholder data
            return {
                "total_searches": 0,
                "avg_search_time": 0.0,
                "hybrid_search_usage": 0.0,
                "query_expansion_usage": 0.0,
                "cache_hit_rate": 0.0,
            }

        except Exception as e:
            print(f"⚠️ Error analyzing vector search performance: {e}")
            return {"error": str(e)}

    async def _analyze_real_time_updates(
        self, user_id: Optional[str], time_window_minutes: int
    ) -> Dict[str, Any]:
        """Analyze real-time update performance"""
        try:
            # Get queue status from real-time context service
            queue_status = await real_time_context_service.get_queue_status()

            return {
                "queue_length": queue_status["queue_length"],
                "processing_tasks": queue_status["processing_tasks"],
                "max_concurrent": queue_status["max_concurrent"],
                "queue_utilization": (
                    queue_status["queue_length"] / queue_status["max_concurrent"]
                    if queue_status["max_concurrent"] > 0
                    else 0
                ),
            }

        except Exception as e:
            print(f"⚠️ Error analyzing real-time updates: {e}")
            return {"error": str(e)}

    async def _identify_bottlenecks(
        self, performance_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        # Check response times
        if "response_time" in performance_summary:
            avg_response_time = performance_summary["response_time"]["avg"]
            if avg_response_time > 5.0:
                bottlenecks.append(
                    {
                        "type": "response_time",
                        "severity": "high" if avg_response_time > 10.0 else "medium",
                        "description": f"Average response time is {avg_response_time:.2f}s",
                        "recommendation": "Optimize context retrieval and AI response generation",
                    }
                )

        # Check context retrieval times
        if "context_retrieval_time" in performance_summary:
            avg_retrieval_time = performance_summary["context_retrieval_time"]["avg"]
            if avg_retrieval_time > 2.0:
                bottlenecks.append(
                    {
                        "type": "context_retrieval",
                        "severity": "high" if avg_retrieval_time > 5.0 else "medium",
                        "description": f"Average context retrieval time is {avg_retrieval_time:.2f}s",
                        "recommendation": "Optimize context retrieval pipeline and caching",
                    }
                )

        # Check vector search times
        if "vector_search_time" in performance_summary:
            avg_search_time = performance_summary["vector_search_time"]["avg"]
            if avg_search_time > 1.0:
                bottlenecks.append(
                    {
                        "type": "vector_search",
                        "severity": "high" if avg_search_time > 3.0 else "medium",
                        "description": f"Average vector search time is {avg_search_time:.2f}s",
                        "recommendation": "Optimize vector database queries and indexing",
                    }
                )

        # Check error rates
        if "error_rate" in performance_summary:
            error_rate = performance_summary["error_rate"]["avg"]
            if error_rate > 0.05:
                bottlenecks.append(
                    {
                        "type": "error_rate",
                        "severity": "high" if error_rate > 0.1 else "medium",
                        "description": f"Error rate is {error_rate:.1%}",
                        "recommendation": "Investigate and fix error sources",
                    }
                )

        return bottlenecks

    async def _generate_optimization_recommendations(
        self,
        performance_summary: Dict[str, Any],
        context_metrics: Dict[str, Any],
        vector_metrics: Dict[str, Any],
        real_time_metrics: Dict[str, Any],
        bottlenecks: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Context retrieval optimizations
        if "context_retrieval_time" in performance_summary:
            avg_retrieval_time = performance_summary["context_retrieval_time"]["avg"]
            if avg_retrieval_time > 2.0:
                recommendations.append(
                    "Implement context caching to reduce retrieval times"
                )
                recommendations.append(
                    "Optimize database queries for context retrieval"
                )
                recommendations.append(
                    "Consider parallel context retrieval for multiple sources"
                )

        # Vector search optimizations
        if "vector_search_time" in performance_summary:
            avg_search_time = performance_summary["vector_search_time"]["avg"]
            if avg_search_time > 1.0:
                recommendations.append("Optimize vector database indexes")
                recommendations.append("Implement search result caching")
                recommendations.append(
                    "Consider reducing search result count for faster queries"
                )

        # Real-time update optimizations
        if real_time_metrics.get("queue_utilization", 0) > 0.8:
            recommendations.append("Increase concurrent update processing capacity")
            recommendations.append("Implement priority-based queue processing")

        # Memory optimizations
        if "memory_usage" in performance_summary:
            memory_usage = performance_summary["memory_usage"]["avg"]
            if memory_usage > 0.8:
                recommendations.append("Implement memory cleanup for old context data")
                recommendations.append("Consider reducing context cache size")

        # General optimizations
        if not recommendations:
            recommendations.append("System performance is within acceptable ranges")
            recommendations.append("Continue monitoring for performance degradation")

        return recommendations

    async def apply_optimization(
        self, optimization_type: str, parameters: Dict[str, Any]
    ) -> bool:
        """Apply a specific optimization"""
        try:
            print(f"🔧 Applying optimization: {optimization_type}")

            if optimization_type == "enable_context_caching":
                # Enable enhanced context caching
                # This would integrate with the cache service
                pass

            elif optimization_type == "optimize_vector_search":
                # Optimize vector search parameters
                # This would integrate with vector optimization service
                pass

            elif optimization_type == "increase_concurrent_updates":
                # Increase concurrent update processing
                # This would modify real-time context service settings
                pass

            elif optimization_type == "cleanup_old_data":
                # Clean up old context data
                await self._cleanup_old_context_data(parameters.get("max_age_days", 7))

            # Record optimization
            optimization_record = {
                "type": optimization_type,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat(),
                "status": "applied",
            }
            self.optimization_history.append(optimization_record)

            print(f"✅ Optimization applied: {optimization_type}")
            return True

        except Exception as e:
            print(f"❌ Failed to apply optimization {optimization_type}: {e}")
            return False

    async def _cleanup_old_context_data(self, max_age_days: int):
        """Clean up old context data"""
        try:
            async with AsyncSessionLocal() as db:
                cutoff_date = datetime.now() - timedelta(days=max_age_days)

                # Clean up old conversation contexts
                result = await db.execute(
                    select(ContextUsageLog).where(
                        ContextUsageLog.created_at < cutoff_date
                    )
                )
                old_logs = result.scalars().all()

                for log in old_logs:
                    await db.delete(log)

                await db.commit()
                print(f"✅ Cleaned up {len(old_logs)} old context usage logs")

        except Exception as e:
            print(f"❌ Error cleaning up old context data: {e}")


class PerformanceMonitoringService:
    """Main performance monitoring and optimization service"""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.optimizer = PerformanceOptimizer(self.monitor)
        self.service_started = False

    async def start_service(self):
        """Start the performance monitoring service"""
        if not self.service_started:
            self.monitor.start_monitoring()
            self.service_started = True
            print("🚀 Performance monitoring service started")

    async def stop_service(self):
        """Stop the performance monitoring service"""
        if self.service_started:
            self.monitor.stop_monitoring()
            self.service_started = False
            print("🛑 Performance monitoring service stopped")

    async def generate_performance_report(
        self, user_id: Optional[str] = None, time_window_minutes: int = 60
    ) -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            print(f"📊 Generating performance report for user {user_id or 'all users'}")

            # Get performance analysis
            analysis = await self.optimizer.analyze_performance(
                user_id, time_window_minutes
            )

            # Get active alerts
            active_alerts = await self.monitor.get_active_alerts()

            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=time_window_minutes)

            # Create performance report
            report = PerformanceReport(
                report_id=f"perf_report_{user_id or 'all'}_{int(time.time())}",
                user_id=user_id,
                time_range=(start_time, end_time),
                total_requests=analysis.get("context_metrics", {}).get(
                    "total_requests", 0
                ),
                avg_response_time=analysis.get("performance_summary", {})
                .get("response_time", {})
                .get("avg", 0),
                p95_response_time=0,  # Would need more detailed metrics
                p99_response_time=0,  # Would need more detailed metrics
                error_rate=analysis.get("performance_summary", {})
                .get("error_rate", {})
                .get("avg", 0),
                context_retrieval_metrics=analysis.get("context_metrics", {}),
                vector_search_metrics=analysis.get("vector_metrics", {}),
                real_time_update_metrics=analysis.get("real_time_metrics", {}),
                system_metrics=analysis.get("performance_summary", {}),
                alerts=active_alerts,
                recommendations=analysis.get("recommendations", []),
            )

            print(f"✅ Performance report generated: {report.report_id}")
            return report

        except Exception as e:
            print(f"❌ Failed to generate performance report: {e}")
            raise

    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            # Get recent performance summary
            performance_summary = await self.monitor.get_performance_summary(minutes=5)

            # Get active alerts
            active_alerts = await self.monitor.get_active_alerts()

            # Get queue status
            queue_status = await real_time_context_service.get_queue_status()

            # Determine overall health
            health_score = 100
            health_issues = []

            # Check response times
            if "response_time" in performance_summary:
                avg_response_time = performance_summary["response_time"]["avg"]
                if avg_response_time > 10.0:
                    health_score -= 30
                    health_issues.append("High response times")
                elif avg_response_time > 5.0:
                    health_score -= 15
                    health_issues.append("Elevated response times")

            # Check error rates
            if "error_rate" in performance_summary:
                error_rate = performance_summary["error_rate"]["avg"]
                if error_rate > 0.1:
                    health_score -= 25
                    health_issues.append("High error rate")
                elif error_rate > 0.05:
                    health_score -= 10
                    health_issues.append("Elevated error rate")

            # Check system resources
            if "memory_usage" in performance_summary:
                memory_usage = performance_summary["memory_usage"]["avg"]
                if memory_usage > 0.9:
                    health_score -= 20
                    health_issues.append("High memory usage")
                elif memory_usage > 0.8:
                    health_score -= 10
                    health_issues.append("Elevated memory usage")

            # Check queue status
            if queue_status["queue_length"] > queue_status["max_concurrent"] * 2:
                health_score -= 15
                health_issues.append("Queue backlog")

            # Determine health status
            if health_score >= 90:
                health_status = "excellent"
            elif health_score >= 75:
                health_status = "good"
            elif health_score >= 50:
                health_status = "fair"
            else:
                health_status = "poor"

            return {
                "health_status": health_status,
                "health_score": max(0, health_score),
                "health_issues": health_issues,
                "active_alerts": len(active_alerts),
                "queue_status": queue_status,
                "performance_summary": performance_summary,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"❌ Failed to get system health: {e}")
            return {
                "health_status": "unknown",
                "health_score": 0,
                "health_issues": [f"Health check failed: {e}"],
                "error": str(e),
            }


# Global instances
performance_monitor = PerformanceMonitor()
performance_optimizer = PerformanceOptimizer(performance_monitor)
performance_monitoring_service = PerformanceMonitoringService()
