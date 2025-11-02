#!/usr/bin/env python3
"""
Test script for Gemini integration in StudyGuru Pro
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set LLM_MODEL to gemini for testing
os.environ["LLM_MODEL"] = "gemini"

from app.config.langchain_config import StudyGuruModels, StudyGuruConfig
from app.services.langchain_service import LangChainService


async def test_gemini_models():
    """Test Gemini model configurations"""
    print("🧪 Testing Gemini Model Integration")
    print("=" * 50)

    # Test model instantiation
    try:
        print("1. Testing Chat Model...")
        chat_model = StudyGuruModels.get_chat_model()
        print(f"   ✅ Chat model created: {type(chat_model).__name__}")
        print(f"   Model: {getattr(chat_model, 'model', 'N/A')}")

        print("\n2. Testing Vision Model...")
        vision_model = StudyGuruModels.get_vision_model()
        print(f"   ✅ Vision model created: {type(vision_model).__name__}")
        print(f"   Model: {getattr(vision_model, 'model', 'N/A')}")

        print("\n3. Testing Guardrail Model...")
        guardrail_model = StudyGuruModels.get_guardrail_model()
        print(f"   ✅ Guardrail model created: {type(guardrail_model).__name__}")
        print(f"   Model: {getattr(guardrail_model, 'model', 'N/A')}")

        print("\n4. Testing Complex Reasoning Model...")
        reasoning_model = StudyGuruModels.get_complex_reasoning_model()
        print(f"   ✅ Reasoning model created: {type(reasoning_model).__name__}")
        print(f"   Model: {getattr(reasoning_model, 'model', 'N/A')}")

        print("\n5. Testing Embeddings Model...")
        embeddings_model = StudyGuruModels.get_embeddings_model()
        print(f"   ✅ Embeddings model created: {type(embeddings_model).__name__}")
        print(f"   Model: {getattr(embeddings_model, 'model', 'N/A')}")

        print("\n6. Testing Title Model...")
        title_model = StudyGuruModels.get_title_model()
        print(f"   ✅ Title model created: {type(title_model).__name__}")
        print(f"   Model: {getattr(title_model, 'model', 'N/A')}")

        print("\n7. Testing Vector Store Configuration...")
        vector_config = StudyGuruConfig.VECTOR_STORE.get_collection_config()
        print(f"   ✅ Vector store config created")
        print(f"   Dimension: {vector_config['dimension']}")
        print(f"   Collection: {vector_config['collection_name']}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    return True


async def test_gemini_chat():
    """Test actual Gemini chat functionality"""
    print("\n🤖 Testing Gemini Chat Functionality")
    print("=" * 50)

    try:
        # Initialize LangChain service
        service = LangChainService()

        # Test simple conversation
        print("1. Testing simple conversation...")
        response, input_tokens, output_tokens, total_tokens = (
            await service.generate_conversation_response(
                message="What is 2+2?", max_tokens=100
            )
        )

        print(f"   ✅ Response received: {response[:100]}...")
        print(f"   Input tokens: {input_tokens}")
        print(f"   Output tokens: {output_tokens}")
        print(f"   Total tokens: {total_tokens}")

        # Test MCQ generation
        print("\n2. Testing MCQ generation...")
        mcq_result = await service.generate_mcq_questions("Basic math addition")

        print(f"   ✅ MCQ generation completed")
        print(f"   Type: {mcq_result.get('type', 'N/A')}")
        print(f"   Language: {mcq_result.get('language', 'N/A')}")
        print(f"   Title: {mcq_result.get('title', 'N/A')}")

        # Test title generation
        print("\n3. Testing title generation...")
        title, summary = await service.generate_interaction_title(
            "Help me with calculus derivatives",
            "I'll help you understand calculus derivatives step by step...",
        )

        print(f"   ✅ Title generation completed")
        print(f"   Title: {title}")
        print(f"   Summary: {summary}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


async def test_gpt_fallback():
    """Test GPT fallback when LLM_MODEL is not gemini"""
    print("\n🔄 Testing GPT Fallback")
    print("=" * 50)

    # Temporarily change to GPT
    original_model = os.environ.get("LLM_MODEL")
    os.environ["LLM_MODEL"] = "gpt"

    try:
        print("1. Testing GPT model instantiation...")
        chat_model = StudyGuruModels.get_chat_model()
        print(f"   ✅ GPT model created: {type(chat_model).__name__}")
        print(f"   Model: {getattr(chat_model, 'model', 'N/A')}")

        print("\n2. Testing GPT embeddings...")
        embeddings_model = StudyGuruModels.get_embeddings_model()
        print(f"   ✅ GPT embeddings created: {type(embeddings_model).__name__}")
        print(f"   Model: {getattr(embeddings_model, 'model', 'N/A')}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        # Restore original model setting
        if original_model:
            os.environ["LLM_MODEL"] = original_model
        else:
            os.environ.pop("LLM_MODEL", None)

    return True


async def main():
    """Main test function"""
    print("🚀 StudyGuru Pro - Gemini Integration Test")
    print("=" * 60)

    # Check if Google API key is set
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("⚠️  Warning: GOOGLE_API_KEY not set. Some tests may fail.")
        print("   Set GOOGLE_API_KEY in your .env file to test Gemini functionality.")
    else:
        print(f"✅ Google API key found: {google_api_key[:10]}...")

    print(f"📋 Current LLM_MODEL: {os.getenv('LLM_MODEL', 'gpt')}")

    # Run tests
    tests = [
        ("Model Configuration", test_gemini_models),
        ("Chat Functionality", test_gemini_chat),
        ("GPT Fallback", test_gpt_fallback),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print("=" * 60)

        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Gemini integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
