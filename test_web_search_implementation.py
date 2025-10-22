#!/usr/bin/env python3
"""
Test script for StudyGuru Web Search implementation
"""

import os
import sys
from app.config.langchain_config import StudyGuruConfig


def test_imports():
    """Test if all required imports are available"""
    print("🔍 Testing imports...")

    try:
        from google.generativeai.types import Tool
        from google.generativeai.protos import GoogleSearchRetrieval

        print("✅ Native Google Search tool imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Run 'pip install google-generativeai' to get native tool support")
        return False


def test_model_creation():
    """Test model creation with web search"""
    print("\n🔍 Testing model creation...")

    # Check if we're using Gemini
    if not StudyGuruConfig.MODELS._is_gemini_model():
        print("❌ Not using Gemini model - web search not available")
        return False

    try:
        # Test chat model with web search enabled
        chat_model = StudyGuruConfig.MODELS.get_chat_model(web_search=True)
        print("✅ Chat model with web search created successfully")

        # Test chat model with web search disabled
        chat_model_no_search = StudyGuruConfig.MODELS.get_chat_model(web_search=False)
        print("✅ Chat model without web search created successfully")

        # Test vision model with web search enabled
        vision_model = StudyGuruConfig.MODELS.get_vision_model(web_search=True)
        print("✅ Vision model with web search created successfully")

        # Test dedicated web search model
        web_search_model = StudyGuruConfig.MODELS.get_web_search_model()
        print("✅ Dedicated web search model created successfully")

        return True

    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def test_chain_creation():
    """Test chain creation"""
    print("\n🔍 Testing chain creation...")

    try:
        # Test web search chain
        web_search_chain = StudyGuruConfig.CHAINS.get_web_search_chain()
        print("✅ Web search chain created successfully")

        return True

    except Exception as e:
        print(f"❌ Chain creation error: {e}")
        return False


def test_environment():
    """Test environment configuration"""
    print("\n🔍 Testing environment configuration...")

    # Check required environment variables
    required_vars = ["GOOGLE_API_KEY", "LLM_MODEL"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False

    print("✅ Environment variables configured")

    # Check if using Gemini
    if os.getenv("LLM_MODEL", "").lower() == "gemini":
        print("✅ Using Gemini model - web search available")
    else:
        print("⚠️  Not using Gemini model - web search not available")

    return True


def main():
    """Run all tests"""
    print("🚀 StudyGuru Web Search Implementation Test")
    print("=" * 60)

    tests = [
        ("Environment Configuration", test_environment),
        ("Import Availability", test_imports),
        ("Model Creation", test_model_creation),
        ("Chain Creation", test_chain_creation),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Web search implementation is ready.")
    else:
        print("⚠️  Some tests failed. Please check the configuration.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
