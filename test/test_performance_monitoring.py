#!/usr/bin/env python3
"""
Test script for Performance Monitoring Service
Tests the comprehensive performance monitoring and optimization system
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.performance_monitoring_service import (
    performance_monitoring_service,
    performance_monitor,
    performance_optimizer,
    MetricType,
)


async def test_metric_recording():
    """Test metric recording functionality"""
    print("📊 Testing Metric Recording")
    print("=" * 50)

    try:
        # Test 1: Record various types of metrics
        print("\n1. Testing Metric Recording")

        # Record counter metric
        performance_monitor.record_metric("test_counter", 1, MetricType.COUNTER)
        print("   ✅ Counter metric recorded")

        # Record gauge metric
        performance_monitor.record_metric("test_gauge", 75.5, MetricType.GAUGE)
        print("   ✅ Gauge metric recorded")

        # Record timer metric
        performance_monitor.record_metric("test_timer", 1.25, MetricType.TIMER)
        print("   ✅ Timer metric recorded")

        # Record histogram metric
        performance_monitor.record_metric("test_histogram", 42, MetricType.HISTOGRAM)
        print("   ✅ Histogram metric recorded")

        # Test 2: Record metrics with tags and metadata
        print("\n2. Testing Metrics with Tags and Metadata")
        performance_monitor.record_metric(
            "test_tagged_metric",
            100,
            MetricType.GAUGE,
            tags={"user_id": "test_user", "service": "test_service"},
            metadata={"version": "1.0", "environment": "test"},
        )
        print("   ✅ Tagged metric recorded")

        # Test 3: Check metric storage
        print("\n3. Testing Metric Storage")
        recent_metrics = performance_monitor._get_recent_metrics(minutes=1)
        print(f"   ✅ {len(recent_metrics)} metrics stored in recent time window")

        for metric in recent_metrics[-3:]:  # Show last 3 metrics
            print(f"      {metric.name}: {metric.value} ({metric.metric_type.value})")

        print("\n✅ Metric recording tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Metric recording test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_time_measurement():
    """Test time measurement functionality"""
    print("\n⏱️ Testing Time Measurement")
    print("=" * 50)

    try:
        # Test 1: Basic time measurement
        print("\n1. Testing Basic Time Measurement")
        async with performance_monitor.measure_time("test_operation"):
            await asyncio.sleep(0.1)  # Simulate work
        print("   ✅ Basic time measurement completed")

        # Test 2: Time measurement with tags
        print("\n2. Testing Time Measurement with Tags")
        async with performance_monitor.measure_time(
            "test_tagged_operation", {"user": "test_user"}
        ):
            await asyncio.sleep(0.05)  # Simulate work
        print("   ✅ Tagged time measurement completed")

        # Test 3: Multiple time measurements
        print("\n3. Testing Multiple Time Measurements")
        for i in range(3):
            async with performance_monitor.measure_time(f"test_operation_{i}"):
                await asyncio.sleep(0.02 * (i + 1))  # Varying work times
        print("   ✅ Multiple time measurements completed")

        # Test 4: Check recorded timing metrics
        print("\n4. Testing Timing Metrics Storage")
        timing_metrics = [
            m
            for m in performance_monitor.metrics
            if m.name.startswith("test_operation")
        ]
        print(f"   ✅ {len(timing_metrics)} timing metrics recorded")

        for metric in timing_metrics:
            print(f"      {metric.name}: {metric.value:.3f}s")

        print("\n✅ Time measurement tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Time measurement test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_performance_summary():
    """Test performance summary functionality"""
    print("\n📈 Testing Performance Summary")
    print("=" * 50)

    try:
        # Generate some test metrics
        print("\n1. Generating Test Metrics")
        for i in range(10):
            performance_monitor.record_metric(
                "test_response_time", 0.5 + (i * 0.1), MetricType.TIMER
            )
            performance_monitor.record_metric(
                "test_memory_usage", 0.6 + (i * 0.02), MetricType.GAUGE
            )
            performance_monitor.record_metric(
                "test_request_count", 1, MetricType.COUNTER
            )

        print("   ✅ Test metrics generated")

        # Test 2: Get performance summary
        print("\n2. Testing Performance Summary Generation")
        summary = await performance_monitor.get_performance_summary(minutes=1)
        print(f"   ✅ Performance summary generated")

        # Display summary
        for metric_name, stats in summary.items():
            print(f"      {metric_name}:")
            print(f"         Count: {stats['count']}")
            print(f"         Average: {stats['avg']:.3f}")
            print(f"         Min: {stats['min']:.3f}")
            print(f"         Max: {stats['max']:.3f}")
            print(f"         Latest: {stats['latest']:.3f}")

        print("\n✅ Performance summary tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Performance summary test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_alert_system():
    """Test alert system functionality"""
    print("\n🚨 Testing Alert System")
    print("=" * 50)

    try:
        # Test 1: Generate metrics that should trigger alerts
        print("\n1. Generating High-Value Metrics for Alerts")

        # Generate high response time (should trigger alert)
        for i in range(5):
            performance_monitor.record_metric("response_time", 6.0, MetricType.TIMER)

        # Generate high memory usage (should trigger alert)
        for i in range(5):
            performance_monitor.record_metric("memory_usage", 0.85, MetricType.GAUGE)

        print("   ✅ High-value metrics generated")

        # Test 2: Check for alerts
        print("\n2. Testing Alert Detection")
        await performance_monitor._check_thresholds()

        active_alerts = await performance_monitor.get_active_alerts()
        print(f"   ✅ {len(active_alerts)} active alerts detected")

        for alert in active_alerts:
            print(f"      Alert: {alert.metric_name}")
            print(f"         Severity: {alert.severity}")
            print(f"         Message: {alert.message}")
            print(f"         Current Value: {alert.current_value:.2f}")
            print(f"         Threshold: {alert.threshold:.2f}")

        # Test 3: Resolve alerts
        print("\n3. Testing Alert Resolution")
        if active_alerts:
            alert_id = active_alerts[0].alert_id
            await performance_monitor.resolve_alert(alert_id)
            print(f"   ✅ Alert resolved: {alert_id}")

        print("\n✅ Alert system tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Alert system test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_performance_analysis():
    """Test performance analysis functionality"""
    print("\n🔍 Testing Performance Analysis")
    print("=" * 50)

    try:
        # Test 1: Run performance analysis
        print("\n1. Testing Performance Analysis")
        analysis = await performance_optimizer.analyze_performance(
            user_id="test_user_123", time_window_minutes=60
        )

        print(f"   ✅ Performance analysis completed")
        print(f"      Analysis timestamp: {analysis.get('analysis_timestamp', 'N/A')}")

        # Test 2: Check analysis components
        print("\n2. Testing Analysis Components")

        performance_summary = analysis.get("performance_summary", {})
        print(f"      Performance summary: {len(performance_summary)} metrics")

        context_metrics = analysis.get("context_metrics", {})
        print(f"      Context metrics: {len(context_metrics)} items")

        vector_metrics = analysis.get("vector_metrics", {})
        print(f"      Vector metrics: {len(vector_metrics)} items")

        real_time_metrics = analysis.get("real_time_metrics", {})
        print(f"      Real-time metrics: {len(real_time_metrics)} items")

        bottlenecks = analysis.get("bottlenecks", [])
        print(f"      Bottlenecks identified: {len(bottlenecks)}")

        recommendations = analysis.get("recommendations", [])
        print(f"      Recommendations generated: {len(recommendations)}")

        # Test 3: Display recommendations
        print("\n3. Testing Recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. {rec}")

        print("\n✅ Performance analysis tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Performance analysis test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_system_health():
    """Test system health monitoring"""
    print("\n💚 Testing System Health Monitoring")
    print("=" * 50)

    try:
        # Test 1: Get system health
        print("\n1. Testing System Health Check")
        health = await performance_monitoring_service.get_system_health()

        print(f"   ✅ System health check completed")
        print(f"      Health status: {health.get('health_status', 'unknown')}")
        print(f"      Health score: {health.get('health_score', 0)}")
        print(f"      Active alerts: {health.get('active_alerts', 0)}")

        # Test 2: Check health issues
        print("\n2. Testing Health Issues Detection")
        health_issues = health.get("health_issues", [])
        if health_issues:
            print(f"      Health issues detected: {len(health_issues)}")
            for issue in health_issues:
                print(f"         - {issue}")
        else:
            print("      No health issues detected")

        # Test 3: Check queue status
        print("\n3. Testing Queue Status")
        queue_status = health.get("queue_status", {})
        print(f"      Queue length: {queue_status.get('queue_length', 0)}")
        print(f"      Processing tasks: {queue_status.get('processing_tasks', 0)}")
        print(f"      Max concurrent: {queue_status.get('max_concurrent', 0)}")

        print("\n✅ System health monitoring tests completed successfully!")

    except Exception as e:
        print(f"\n❌ System health monitoring test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_performance_report():
    """Test performance report generation"""
    print("\n📋 Testing Performance Report Generation")
    print("=" * 50)

    try:
        # Test 1: Generate performance report
        print("\n1. Testing Performance Report Generation")
        report = await performance_monitoring_service.generate_performance_report(
            user_id="test_user_123", time_window_minutes=60
        )

        print(f"   ✅ Performance report generated: {report.report_id}")
        print(f"      User ID: {report.user_id}")
        print(f"      Time range: {report.time_range[0]} to {report.time_range[1]}")
        print(f"      Total requests: {report.total_requests}")
        print(f"      Average response time: {report.avg_response_time:.3f}s")
        print(f"      Error rate: {report.error_rate:.1%}")

        # Test 2: Check report components
        print("\n2. Testing Report Components")
        print(
            f"      Context retrieval metrics: {len(report.context_retrieval_metrics)} items"
        )
        print(f"      Vector search metrics: {len(report.vector_search_metrics)} items")
        print(
            f"      Real-time update metrics: {len(report.real_time_update_metrics)} items"
        )
        print(f"      System metrics: {len(report.system_metrics)} items")
        print(f"      Active alerts: {len(report.alerts)}")
        print(f"      Recommendations: {len(report.recommendations)}")

        # Test 3: Display recommendations
        print("\n3. Testing Report Recommendations")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"      {i}. {rec}")

        print("\n✅ Performance report generation tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Performance report generation test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_optimization_application():
    """Test optimization application"""
    print("\n🔧 Testing Optimization Application")
    print("=" * 50)

    try:
        # Test 1: Apply optimization
        print("\n1. Testing Optimization Application")
        success = await performance_optimizer.apply_optimization(
            "cleanup_old_data", {"max_age_days": 1}
        )

        print(f"   ✅ Optimization application result: {success}")

        # Test 2: Check optimization history
        print("\n2. Testing Optimization History")
        history = performance_optimizer.optimization_history
        print(f"      Optimization history: {len(history)} records")

        for opt in history:
            print(f"         {opt['type']}: {opt['status']} at {opt['timestamp']}")

        print("\n✅ Optimization application tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Optimization application test failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests"""
    print("🚀 Starting Performance Monitoring Service Tests")
    print("=" * 60)

    # Start the monitoring service
    await performance_monitoring_service.start_service()

    try:
        await test_metric_recording()
        await test_time_measurement()
        await test_performance_summary()
        await test_alert_system()
        await test_performance_analysis()
        await test_system_health()
        await test_performance_report()
        await test_optimization_application()

    finally:
        # Stop the monitoring service
        await performance_monitoring_service.stop_service()

    print("\n🎉 All performance monitoring service tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
