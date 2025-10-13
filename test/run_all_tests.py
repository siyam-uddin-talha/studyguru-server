#!/usr/bin/env python3
"""
Test runner for all enhanced RAG system tests

This script runs all test files in the test directory to validate
the enhanced RAG system functionality.
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_test_file(test_file: str) -> bool:
    """Run a single test file and return success status"""
    print(f"\n🧪 Running {test_file}...")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"✅ {test_file} - PASSED")
            return True
        else:
            print(f"❌ {test_file} - FAILED")
            print(f"Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ {test_file} - ERROR: {e}")
        return False


def main():
    """Run all test files"""
    print("🚀 Enhanced RAG System - Test Suite Runner")
    print("=" * 60)

    # Get all test files
    test_dir = Path(__file__).parent
    test_files = [
        "test_basic.py",
        "test_langchain.py",
        "test_vector_db.py",
        "test_vector_optimization.py",
        "test_document_integration.py",
        "test_real_time_context.py",
        "test_context_testing_framework.py",
        "test_performance_monitoring.py",
    ]

    # Filter to only existing files
    existing_tests = [f for f in test_files if (test_dir / f).exists()]

    print(f"📋 Found {len(existing_tests)} test files to run")

    # Run all tests
    passed = 0
    failed = 0

    for test_file in existing_tests:
        if run_test_file(test_file):
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📋 Total: {passed + failed}")

    if failed == 0:
        print("\n🎉 All tests passed! Enhanced RAG system is ready.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
