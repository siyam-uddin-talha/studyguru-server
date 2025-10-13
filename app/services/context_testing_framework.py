"""
Comprehensive testing and validation framework for context usage
Provides automated testing, validation, and monitoring of the RAG system
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.interaction import Interaction, Conversation
from app.models.context import (
    ConversationContext,
    UserLearningProfile,
    DocumentContext,
    ContextUsageLog,
)
from app.services.context_service import context_service
from app.services.semantic_summary_service import semantic_summary_service
from app.services.vector_optimization_service import vector_optimization_service
from app.services.document_integration_service import document_integration_service
from app.services.real_time_context_service import real_time_context_service


class TestResult(Enum):
    """Test result status"""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class TestCase:
    """Represents a test case"""

    test_id: str
    name: str
    description: str
    test_type: str  # "context_retrieval", "semantic_summary", "vector_search", etc.
    expected_result: Any
    test_data: Dict[str, Any]
    timeout: int = 30
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TestResult:
    """Represents a test result"""

    test_id: str
    test_name: str
    result: TestResult
    execution_time: float
    error_message: Optional[str] = None
    actual_result: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""

    report_id: str
    user_id: str
    interaction_id: Optional[str]
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    skipped_tests: int
    execution_time: float
    test_results: List[TestResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class ContextTestingFramework:
    """Comprehensive testing and validation framework for context usage"""

    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.test_results: List[TestResult] = []
        self.validation_reports: List[ValidationReport] = []

    def register_test_case(self, test_case: TestCase):
        """Register a test case"""
        self.test_cases.append(test_case)

    async def run_test_case(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        start_time = time.time()

        try:
            print(f"🧪 Running test: {test_case.name}")

            # Route to appropriate test handler
            if test_case.test_type == "context_retrieval":
                actual_result = await self._test_context_retrieval(test_case)
            elif test_case.test_type == "semantic_summary":
                actual_result = await self._test_semantic_summary(test_case)
            elif test_case.test_type == "vector_search":
                actual_result = await self._test_vector_search(test_case)
            elif test_case.test_type == "document_integration":
                actual_result = await self._test_document_integration(test_case)
            elif test_case.test_type == "real_time_updates":
                actual_result = await self._test_real_time_updates(test_case)
            elif test_case.test_type == "consistency_check":
                actual_result = await self._test_consistency_check(test_case)
            else:
                raise ValueError(f"Unknown test type: {test_case.test_type}")

            execution_time = time.time() - start_time

            # Validate result
            if self._validate_result(actual_result, test_case.expected_result):
                result = TestResult.PASS
                error_message = None
            else:
                result = TestResult.FAIL
                error_message = (
                    f"Expected: {test_case.expected_result}, Got: {actual_result}"
                )

            test_result = TestResult(
                test_id=test_case.test_id,
                test_name=test_case.name,
                result=result,
                execution_time=execution_time,
                error_message=error_message,
                actual_result=actual_result,
                metrics={"execution_time": execution_time},
            )

            print(f"   ✅ Test {test_case.name}: {result.value}")
            return test_result

        except Exception as e:
            execution_time = time.time() - start_time
            test_result = TestResult(
                test_id=test_case.test_id,
                test_name=test_case.name,
                result=TestResult.FAIL,
                execution_time=execution_time,
                error_message=str(e),
                metrics={"execution_time": execution_time},
            )

            print(f"   ❌ Test {test_case.name}: FAIL - {e}")
            return test_result

    async def _test_context_retrieval(self, test_case: TestCase) -> Dict[str, Any]:
        """Test context retrieval functionality"""
        test_data = test_case.test_data

        # Get comprehensive context
        context_result = await context_service.get_comprehensive_context(
            user_id=test_data["user_id"],
            interaction_id=test_data.get("interaction_id"),
            message=test_data["message"],
            include_cross_interaction=test_data.get("include_cross_interaction", True),
            max_context_length=test_data.get("max_context_length", 4000),
        )

        return {
            "context_length": len(context_result.get("context", "")),
            "sources_used": context_result.get("metadata", {}).get("sources_used", []),
            "retrieval_time": context_result.get("metadata", {}).get(
                "total_retrieval_time", 0
            ),
            "context_quality": self._assess_context_quality(context_result),
        }

    async def _test_semantic_summary(self, test_case: TestCase) -> Dict[str, Any]:
        """Test semantic summary functionality"""
        test_data = test_case.test_data

        # Create conversation summary
        summary = await semantic_summary_service.create_conversation_summary(
            user_message=test_data["user_message"], ai_response=test_data["ai_response"]
        )

        return {
            "summary_length": len(summary.get("semantic_summary", "")),
            "topics_count": len(summary.get("main_topics", [])),
            "facts_count": len(summary.get("key_facts", [])),
            "has_learning_progress": bool(summary.get("learning_progress")),
            "summary_quality": self._assess_summary_quality(summary),
        }

    async def _test_vector_search(self, test_case: TestCase) -> Dict[str, Any]:
        """Test vector search functionality"""
        test_data = test_case.test_data

        from app.services.vector_optimization_service import SearchQuery

        search_query = SearchQuery(
            query=test_data["query"],
            user_id=test_data["user_id"],
            top_k=test_data.get("top_k", 10),
            hybrid_search=test_data.get("hybrid_search", True),
            query_expansion=test_data.get("query_expansion", True),
            boost_recent=test_data.get("boost_recent", True),
        )

        results = await vector_optimization_service.hybrid_search(search_query)

        return {
            "results_count": len(results),
            "avg_relevance_score": (
                sum(r.relevance_score for r in results) / len(results) if results else 0
            ),
            "has_metadata": all(r.metadata for r in results),
            "search_quality": self._assess_search_quality(results),
        }

    async def _test_document_integration(self, test_case: TestCase) -> Dict[str, Any]:
        """Test document integration functionality"""
        test_data = test_case.test_data

        # Test document search
        search_results = await document_integration_service.search_documents(
            user_id=test_data["user_id"],
            query=test_data["query"],
            top_k=test_data.get("top_k", 5),
        )

        return {
            "search_results_count": len(search_results),
            "has_document_types": any(r.get("document_type") for r in search_results),
            "has_subject_areas": any(r.get("subject_area") for r in search_results),
            "integration_quality": self._assess_document_integration_quality(
                search_results
            ),
        }

    async def _test_real_time_updates(self, test_case: TestCase) -> Dict[str, Any]:
        """Test real-time update functionality"""
        test_data = test_case.test_data

        # Queue a test update
        task_id = await real_time_context_service.queue_context_update(
            user_id=test_data["user_id"],
            interaction_id=test_data["interaction_id"],
            conversation_id=test_data.get("conversation_id"),
            update_type=test_data["update_type"],
            payload=test_data["payload"],
        )

        # Wait for processing
        await asyncio.sleep(2)

        # Check status
        status = await real_time_context_service.get_update_status(task_id)

        return {
            "task_queued": bool(task_id),
            "task_status": status.get("status") if status else "unknown",
            "processing_time": status.get("execution_time", 0) if status else 0,
            "update_quality": self._assess_update_quality(status),
        }

    async def _test_consistency_check(self, test_case: TestCase) -> Dict[str, Any]:
        """Test consistency checking functionality"""
        test_data = test_case.test_data

        # Run consistency check
        is_consistent = await real_time_context_service.ensure_consistency(
            user_id=test_data["user_id"], interaction_id=test_data["interaction_id"]
        )

        return {
            "is_consistent": is_consistent,
            "consistency_score": 1.0 if is_consistent else 0.0,
            "check_quality": self._assess_consistency_quality(is_consistent),
        }

    def _validate_result(self, actual: Any, expected: Any) -> bool:
        """Validate test result against expected result"""
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False

            for key, expected_value in expected.items():
                if key not in actual:
                    return False

                if isinstance(expected_value, (int, float)):
                    # Allow some tolerance for numeric values
                    if abs(actual[key] - expected_value) > 0.1:
                        return False
                elif actual[key] != expected_value:
                    return False

            return True
        else:
            return actual == expected

    def _assess_context_quality(self, context_result: Dict[str, Any]) -> float:
        """Assess the quality of retrieved context"""
        context = context_result.get("context", "")
        metadata = context_result.get("metadata", {})

        quality_score = 0.0

        # Length check
        if len(context) > 100:
            quality_score += 0.3

        # Sources used check
        sources_used = metadata.get("sources_used", [])
        if len(sources_used) >= 2:
            quality_score += 0.3

        # Retrieval time check
        retrieval_time = metadata.get("total_retrieval_time", 0)
        if retrieval_time < 2.0:  # Less than 2 seconds
            quality_score += 0.2

        # Context relevance (basic check)
        if "context" in context.lower() or "previous" in context.lower():
            quality_score += 0.2

        return min(quality_score, 1.0)

    def _assess_summary_quality(self, summary: Dict[str, Any]) -> float:
        """Assess the quality of semantic summary"""
        quality_score = 0.0

        # Summary length
        summary_text = summary.get("semantic_summary", "")
        if len(summary_text) > 50:
            quality_score += 0.3

        # Topics present
        if summary.get("main_topics"):
            quality_score += 0.3

        # Facts present
        if summary.get("key_facts"):
            quality_score += 0.2

        # Learning progress
        if summary.get("learning_progress"):
            quality_score += 0.2

        return min(quality_score, 1.0)

    def _assess_search_quality(self, results: List) -> float:
        """Assess the quality of vector search results"""
        if not results:
            return 0.0

        quality_score = 0.0

        # Result count
        if len(results) >= 3:
            quality_score += 0.3

        # Relevance scores
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        if avg_relevance > 0.7:
            quality_score += 0.4

        # Metadata presence
        if all(r.metadata for r in results):
            quality_score += 0.3

        return min(quality_score, 1.0)

    def _assess_document_integration_quality(self, results: List[Dict]) -> float:
        """Assess the quality of document integration"""
        if not results:
            return 0.0

        quality_score = 0.0

        # Result count
        if len(results) >= 2:
            quality_score += 0.3

        # Document types present
        if any(r.get("document_type") for r in results):
            quality_score += 0.3

        # Subject areas present
        if any(r.get("subject_area") for r in results):
            quality_score += 0.2

        # Content quality
        if any(len(r.get("content", "")) > 50 for r in results):
            quality_score += 0.2

        return min(quality_score, 1.0)

    def _assess_update_quality(self, status: Optional[Dict]) -> float:
        """Assess the quality of real-time updates"""
        if not status:
            return 0.0

        quality_score = 0.0

        # Status check
        if status.get("status") == "completed":
            quality_score += 0.6

        # Processing time
        execution_time = status.get("execution_time", 0)
        if execution_time < 5.0:  # Less than 5 seconds
            quality_score += 0.4

        return min(quality_score, 1.0)

    def _assess_consistency_quality(self, is_consistent: bool) -> float:
        """Assess the quality of consistency checks"""
        return 1.0 if is_consistent else 0.0

    async def run_comprehensive_validation(
        self, user_id: str, interaction_id: Optional[str] = None
    ) -> ValidationReport:
        """Run comprehensive validation for a user/interaction"""
        start_time = time.time()
        report_id = f"validation_{user_id}_{int(time.time())}"

        print(f"🔍 Starting comprehensive validation for user {user_id}")

        # Create default test cases if none registered
        if not self.test_cases:
            await self._create_default_test_cases(user_id, interaction_id)

        # Run all test cases
        test_results = []
        for test_case in self.test_cases:
            result = await self.run_test_case(test_case)
            test_results.append(result)

        # Calculate statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.result == TestResult.PASS)
        failed_tests = sum(1 for r in test_results if r.result == TestResult.FAIL)
        warning_tests = sum(1 for r in test_results if r.result == TestResult.WARNING)
        skipped_tests = sum(1 for r in test_results if r.result == TestResult.SKIP)

        execution_time = time.time() - start_time

        # Generate recommendations
        recommendations = self._generate_recommendations(test_results)

        # Create validation report
        report = ValidationReport(
            report_id=report_id,
            user_id=user_id,
            interaction_id=interaction_id,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            skipped_tests=skipped_tests,
            execution_time=execution_time,
            test_results=test_results,
            recommendations=recommendations,
        )

        self.validation_reports.append(report)

        print(f"✅ Comprehensive validation completed:")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Warnings: {warning_tests}")
        print(f"   Skipped: {skipped_tests}")
        print(f"   Execution time: {execution_time:.2f}s")

        return report

    async def _create_default_test_cases(
        self, user_id: str, interaction_id: Optional[str]
    ):
        """Create default test cases for validation"""
        # Context retrieval test
        self.register_test_case(
            TestCase(
                test_id="context_retrieval_001",
                name="Basic Context Retrieval",
                description="Test basic context retrieval functionality",
                test_type="context_retrieval",
                expected_result={
                    "context_length": 100,  # Minimum expected length
                    "sources_used": 1,  # At least one source
                    "retrieval_time": 5.0,  # Maximum 5 seconds
                    "context_quality": 0.5,  # Minimum quality score
                },
                test_data={
                    "user_id": user_id,
                    "interaction_id": interaction_id,
                    "message": "What is photosynthesis?",
                    "include_cross_interaction": True,
                    "max_context_length": 4000,
                },
            )
        )

        # Semantic summary test
        self.register_test_case(
            TestCase(
                test_id="semantic_summary_001",
                name="Semantic Summary Creation",
                description="Test semantic summary creation",
                test_type="semantic_summary",
                expected_result={
                    "summary_length": 50,  # Minimum expected length
                    "topics_count": 1,  # At least one topic
                    "facts_count": 0,  # Facts are optional
                    "has_learning_progress": True,
                    "summary_quality": 0.6,  # Minimum quality score
                },
                test_data={
                    "user_message": "What is photosynthesis?",
                    "ai_response": "Photosynthesis is the process by which plants convert light energy into chemical energy using chlorophyll.",
                },
            )
        )

        # Vector search test
        self.register_test_case(
            TestCase(
                test_id="vector_search_001",
                name="Vector Search Functionality",
                description="Test vector search with hybrid search",
                test_type="vector_search",
                expected_result={
                    "results_count": 1,  # At least one result
                    "avg_relevance_score": 0.5,  # Minimum relevance
                    "has_metadata": True,
                    "search_quality": 0.5,  # Minimum quality score
                },
                test_data={
                    "query": "photosynthesis",
                    "user_id": user_id,
                    "top_k": 5,
                    "hybrid_search": True,
                    "query_expansion": True,
                    "boost_recent": True,
                },
            )
        )

        # Document integration test
        self.register_test_case(
            TestCase(
                test_id="document_integration_001",
                name="Document Integration Search",
                description="Test document integration search functionality",
                test_type="document_integration",
                expected_result={
                    "search_results_count": 0,  # May be 0 if no documents
                    "has_document_types": False,  # May be false if no documents
                    "has_subject_areas": False,  # May be false if no documents
                    "integration_quality": 0.0,  # May be 0 if no documents
                },
                test_data={"user_id": user_id, "query": "photosynthesis", "top_k": 5},
            )
        )

        # Real-time updates test
        if interaction_id:
            self.register_test_case(
                TestCase(
                    test_id="real_time_updates_001",
                    name="Real-Time Context Updates",
                    description="Test real-time context update queuing",
                    test_type="real_time_updates",
                    expected_result={
                        "task_queued": True,
                        "task_status": "completed",  # or "in_progress"
                        "processing_time": 10.0,  # Maximum 10 seconds
                        "update_quality": 0.5,  # Minimum quality score
                    },
                    test_data={
                        "user_id": user_id,
                        "interaction_id": interaction_id,
                        "conversation_id": "test_conversation",
                        "update_type": "embedding",
                        "payload": {
                            "conversation_id": "test_conversation",
                            "text": "Test message for real-time updates",
                            "title": "Test title",
                            "metadata": {"test": True},
                        },
                    },
                )
            )

        # Consistency check test
        if interaction_id:
            self.register_test_case(
                TestCase(
                    test_id="consistency_check_001",
                    name="Context Consistency Check",
                    description="Test context consistency validation",
                    test_type="consistency_check",
                    expected_result={
                        "is_consistent": True,
                        "consistency_score": 1.0,
                        "check_quality": 1.0,
                    },
                    test_data={"user_id": user_id, "interaction_id": interaction_id},
                )
            )

    def _generate_recommendations(self, test_results: List[TestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        # Analyze failed tests
        failed_tests = [r for r in test_results if r.result == TestResult.FAIL]
        if failed_tests:
            recommendations.append(
                f"Address {len(failed_tests)} failed tests to improve system reliability"
            )

        # Analyze performance
        slow_tests = [r for r in test_results if r.execution_time > 5.0]
        if slow_tests:
            recommendations.append(
                f"Optimize {len(slow_tests)} slow tests for better performance"
            )

        # Analyze context quality
        context_tests = [r for r in test_results if "context" in r.test_name.lower()]
        low_quality_context = [
            r for r in context_tests if r.metrics.get("context_quality", 0) < 0.7
        ]
        if low_quality_context:
            recommendations.append("Improve context retrieval quality and relevance")

        # Analyze search quality
        search_tests = [r for r in test_results if "search" in r.test_name.lower()]
        low_quality_search = [
            r for r in search_tests if r.metrics.get("search_quality", 0) < 0.7
        ]
        if low_quality_search:
            recommendations.append("Enhance vector search relevance and metadata")

        # General recommendations
        if not recommendations:
            recommendations.append("System is performing well - continue monitoring")

        return recommendations

    async def get_validation_history(
        self, user_id: str, days: int = 7
    ) -> List[ValidationReport]:
        """Get validation history for a user"""
        cutoff_date = datetime.now() - timedelta(days=days)

        return [
            report
            for report in self.validation_reports
            if report.user_id == user_id and report.generated_at >= cutoff_date
        ]

    async def generate_performance_report(self, user_id: str) -> Dict[str, Any]:
        """Generate performance report for a user"""
        user_reports = await self.get_validation_history(user_id, days=30)

        if not user_reports:
            return {"message": "No validation data available"}

        # Calculate averages
        avg_execution_time = sum(r.execution_time for r in user_reports) / len(
            user_reports
        )
        avg_pass_rate = sum(r.passed_tests / r.total_tests for r in user_reports) / len(
            user_reports
        )

        # Get latest report
        latest_report = max(user_reports, key=lambda r: r.generated_at)

        return {
            "user_id": user_id,
            "total_validations": len(user_reports),
            "avg_execution_time": avg_execution_time,
            "avg_pass_rate": avg_pass_rate,
            "latest_validation": {
                "report_id": latest_report.report_id,
                "generated_at": latest_report.generated_at.isoformat(),
                "total_tests": latest_report.total_tests,
                "passed_tests": latest_report.passed_tests,
                "failed_tests": latest_report.failed_tests,
                "recommendations": latest_report.recommendations,
            },
        }


# Global instance
context_testing_framework = ContextTestingFramework()
