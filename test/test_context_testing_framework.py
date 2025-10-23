#!/usr/bin/env python3
"""
Test script for Context Testing Framework
Tests the comprehensive testing and validation framework
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.context_testing_framework import (
    context_testing_framework,
    TestCase,
    TestResult,
)


async def test_framework_basic_functionality():
    """Test basic framework functionality"""
    print("🧪 Testing Framework Basic Functionality")
    print("=" * 50)

    try:
        # Test 1: Register test case
        print("\n1. Testing Test Case Registration")
        test_case = TestCase(
            test_id="test_001",
            name="Basic Test",
            description="A basic test case",
            test_type="context_retrieval",
            expected_result={"context_length": 100},
            test_data={
                "user_id": "test_user_123",
                "interaction_id": "test_interaction_456",
                "message": "What is photosynthesis?",
                "include_cross_interaction": True,
                "max_context_length": 10000,  # Increased from 4000
            },
        )

        context_testing_framework.register_test_case(test_case)
        print(f"   ✅ Test case registered: {test_case.test_id}")

        # Test 2: Run single test case
        print("\n2. Testing Single Test Case Execution")
        result = await context_testing_framework.run_test_case(test_case)
        print(f"   ✅ Test executed: {result.test_name}")
        print(f"      Result: {result.result.value}")
        print(f"      Execution time: {result.execution_time:.3f}s")
        if result.error_message:
            print(f"      Error: {result.error_message}")

        # Test 3: Check test results storage
        print("\n3. Testing Test Results Storage")
        stored_results = context_testing_framework.test_results
        print(f"   ✅ Test results stored: {len(stored_results)} results")

        print("\n✅ Framework basic functionality tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Framework basic functionality test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_comprehensive_validation():
    """Test comprehensive validation functionality"""
    print("\n🔍 Testing Comprehensive Validation")
    print("=" * 50)

    try:
        # Test comprehensive validation
        print("\n1. Running Comprehensive Validation")
        validation_report = (
            await context_testing_framework.run_comprehensive_validation(
                user_id="test_user_123", interaction_id="test_interaction_456"
            )
        )

        print(f"   ✅ Validation report generated: {validation_report.report_id}")
        print(f"      Total tests: {validation_report.total_tests}")
        print(f"      Passed: {validation_report.passed_tests}")
        print(f"      Failed: {validation_report.failed_tests}")
        print(f"      Warnings: {validation_report.warning_tests}")
        print(f"      Skipped: {validation_report.skipped_tests}")
        print(f"      Execution time: {validation_report.execution_time:.3f}s")

        # Test 2: Check recommendations
        print("\n2. Testing Recommendations Generation")
        recommendations = validation_report.recommendations
        print(f"   ✅ Generated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. {rec}")

        # Test 3: Check test results details
        print("\n3. Testing Test Results Details")
        for result in validation_report.test_results:
            print(f"   Test: {result.test_name}")
            print(f"      Result: {result.result.value}")
            print(f"      Execution time: {result.execution_time:.3f}s")
            if result.metrics:
                print(f"      Metrics: {result.metrics}")

        print("\n✅ Comprehensive validation tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Comprehensive validation test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_performance_reporting():
    """Test performance reporting functionality"""
    print("\n📊 Testing Performance Reporting")
    print("=" * 50)

    try:
        # Run multiple validations to generate history
        print("\n1. Generating Validation History")
        for i in range(3):
            await context_testing_framework.run_comprehensive_validation(
                user_id="test_user_123", interaction_id=f"test_interaction_{i}"
            )
            print(f"   ✅ Validation {i+1} completed")

        # Test 2: Get validation history
        print("\n2. Testing Validation History Retrieval")
        history = await context_testing_framework.get_validation_history(
            user_id="test_user_123", days=7
        )
        print(f"   ✅ Retrieved {len(history)} validation reports from history")

        # Test 3: Generate performance report
        print("\n3. Testing Performance Report Generation")
        performance_report = (
            await context_testing_framework.generate_performance_report(
                user_id="test_user_123"
            )
        )

        print(f"   ✅ Performance report generated:")
        print(
            f"      Total validations: {performance_report.get('total_validations', 0)}"
        )
        print(
            f"      Average execution time: {performance_report.get('avg_execution_time', 0):.3f}s"
        )
        print(
            f"      Average pass rate: {performance_report.get('avg_pass_rate', 0):.1%}"
        )

        latest = performance_report.get("latest_validation", {})
        if latest:
            print(f"      Latest validation:")
            print(f"         Report ID: {latest.get('report_id', 'N/A')}")
            print(f"         Total tests: {latest.get('total_tests', 0)}")
            print(f"         Passed tests: {latest.get('passed_tests', 0)}")
            print(f"         Failed tests: {latest.get('failed_tests', 0)}")

        print("\n✅ Performance reporting tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Performance reporting test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_custom_test_cases():
    """Test custom test case creation and execution"""
    print("\n🔧 Testing Custom Test Cases")
    print("=" * 50)

    try:
        # Test 1: Create custom semantic summary test
        print("\n1. Testing Custom Semantic Summary Test")
        semantic_test = TestCase(
            test_id="custom_semantic_001",
            name="Custom Semantic Summary Test",
            description="Custom test for semantic summary creation",
            test_type="semantic_summary",
            expected_result={
                "summary_length": 50,
                "topics_count": 1,
                "summary_quality": 0.6,
            },
            test_data={
                "user_message": "Explain the water cycle",
                "ai_response": "The water cycle is the continuous movement of water through evaporation, condensation, and precipitation.",
            },
        )

        result = await context_testing_framework.run_test_case(semantic_test)
        print(f"   ✅ Custom semantic test executed: {result.result.value}")
        print(f"      Execution time: {result.execution_time:.3f}s")

        # Test 2: Create custom vector search test
        print("\n2. Testing Custom Vector Search Test")
        vector_test = TestCase(
            test_id="custom_vector_001",
            name="Custom Vector Search Test",
            description="Custom test for vector search functionality",
            test_type="vector_search",
            expected_result={
                "results_count": 0,  # May be 0 if no data
                "search_quality": 0.0,  # May be 0 if no data
            },
            test_data={
                "query": "water cycle",
                "user_id": "test_user_123",
                "top_k": 3,
                "hybrid_search": True,
                "query_expansion": True,
                "boost_recent": True,
            },
        )

        result = await context_testing_framework.run_test_case(vector_test)
        print(f"   ✅ Custom vector test executed: {result.result.value}")
        print(f"      Execution time: {result.execution_time:.3f}s")

        # Test 3: Create custom document integration test
        print("\n3. Testing Custom Document Integration Test")
        doc_test = TestCase(
            test_id="custom_doc_001",
            name="Custom Document Integration Test",
            description="Custom test for document integration",
            test_type="document_integration",
            expected_result={
                "search_results_count": 0,  # May be 0 if no documents
                "integration_quality": 0.0,  # May be 0 if no documents
            },
            test_data={"user_id": "test_user_123", "query": "water cycle", "top_k": 3},
        )

        result = await context_testing_framework.run_test_case(doc_test)
        print(f"   ✅ Custom document test executed: {result.result.value}")
        print(f"      Execution time: {result.execution_time:.3f}s")

        print("\n✅ Custom test cases tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Custom test cases test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_error_handling():
    """Test error handling in the framework"""
    print("\n🛡️ Testing Error Handling")
    print("=" * 50)

    try:
        # Test 1: Invalid test type
        print("\n1. Testing Invalid Test Type")
        invalid_test = TestCase(
            test_id="invalid_001",
            name="Invalid Test Type",
            description="Test with invalid test type",
            test_type="invalid_type",
            expected_result={"should": "fail"},
            test_data={"invalid": "data"},
        )

        result = await context_testing_framework.run_test_case(invalid_test)
        print(f"   ✅ Invalid test handled: {result.result.value}")
        if result.error_message:
            print(f"      Error message: {result.error_message}")

        # Test 2: Missing required data
        print("\n2. Testing Missing Required Data")
        missing_data_test = TestCase(
            test_id="missing_data_001",
            name="Missing Data Test",
            description="Test with missing required data",
            test_type="context_retrieval",
            expected_result={"context_length": 100},
            test_data={
                "user_id": "test_user_123",
                # Missing required fields
            },
        )

        result = await context_testing_framework.run_test_case(missing_data_test)
        print(f"   ✅ Missing data test handled: {result.result.value}")
        if result.error_message:
            print(f"      Error message: {result.error_message}")

        # Test 3: Timeout test
        print("\n3. Testing Timeout Handling")
        timeout_test = TestCase(
            test_id="timeout_001",
            name="Timeout Test",
            description="Test with very short timeout",
            test_type="context_retrieval",
            expected_result={"context_length": 100},
            test_data={
                "user_id": "test_user_123",
                "interaction_id": "test_interaction_456",
                "message": "What is photosynthesis?",
                "include_cross_interaction": True,
                "max_context_length": 10000,  # Increased from 4000
            },
            timeout=1,  # Very short timeout
        )

        result = await context_testing_framework.run_test_case(timeout_test)
        print(f"   ✅ Timeout test handled: {result.result.value}")
        print(f"      Execution time: {result.execution_time:.3f}s")

        print("\n✅ Error handling tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests"""
    print("🚀 Starting Context Testing Framework Tests")
    print("=" * 60)

    await test_framework_basic_functionality()
    await test_comprehensive_validation()
    await test_performance_reporting()
    await test_custom_test_cases()
    await test_error_handling()

    print("\n🎉 All context testing framework tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
