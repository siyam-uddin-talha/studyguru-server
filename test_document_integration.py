#!/usr/bin/env python3
"""
Test script for Document Integration Service

This script tests the enhanced document processing capabilities including:
- Comprehensive document analysis
- Document chunking and indexing
- Document search and retrieval
- Document analytics
"""

import asyncio
import sys
import os

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.services.document_integration_service import (
    document_integration_service,
    DocumentChunk,
    DocumentAnalysis,
)


async def test_document_integration():
    """Test the document integration service"""
    print("🧪 Testing Document Integration Service")
    print("=" * 50)

    try:
        # Test 1: Document chunking
        print("\n1. Testing Document Chunking")
        test_content = {
            "raw_content": "Question 1: What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6\nAnswer: B\n\nQuestion 2: What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Madrid\nAnswer: B",
            "structured_questions": [
                {
                    "question_number": "1",
                    "question_text": "What is 2+2?",
                    "options": {"a": "3", "b": "4", "c": "5", "d": "6"},
                    "answer": "B",
                    "explanation": "2+2 equals 4",
                    "difficulty": "easy",
                    "topic": "mathematics",
                },
                {
                    "question_number": "2",
                    "question_text": "What is the capital of France?",
                    "options": {
                        "a": "London",
                        "b": "Paris",
                        "c": "Berlin",
                        "d": "Madrid",
                    },
                    "answer": "B",
                    "explanation": "Paris is the capital of France",
                    "difficulty": "easy",
                    "topic": "geography",
                },
            ],
            "document_type": "mcq",
            "language": "english",
            "title": "Math and Geography Quiz",
            "summary": "Basic math and geography questions",
        }

        chunks = await document_integration_service._create_document_chunks(
            test_content
        )
        print(f"   ✅ Created {len(chunks)} document chunks")

        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {chunk.chunk_type} - {chunk.chunk_id}")
            print(f"      Content: {chunk.content[:100]}...")
            print(f"      Question Number: {chunk.question_number}")
            print(f"      Metadata: {chunk.metadata}")

        # Test 2: Document structure analysis
        print("\n2. Testing Document Structure Analysis")
        document_analysis = (
            await document_integration_service._analyze_document_structure(
                test_content, chunks, {"type": "mcq", "language": "english"}
            )
        )

        print(f"   ✅ Document analysis completed:")
        print(f"      Document Type: {document_analysis.document_type}")
        print(f"      Total Questions: {document_analysis.total_questions}")
        print(f"      Main Topics: {document_analysis.main_topics}")
        print(f"      Difficulty Level: {document_analysis.difficulty_level}")
        print(f"      Subject Area: {document_analysis.subject_area}")
        print(f"      Key Concepts: {document_analysis.key_concepts}")
        print(f"      Chunks: {len(document_analysis.chunks)}")

        # Test 3: Document search
        print("\n3. Testing Document Search")
        search_results = await document_integration_service.search_documents(
            user_id="test_user_123", query="What is 2+2?", top_k=5
        )

        print(f"   ✅ Document search completed: {len(search_results)} results")
        for i, result in enumerate(search_results[:3]):
            print(
                f"   Result {i+1}: {result.get('document_type', 'unknown')} (Score: {result.get('score', 0):.3f})"
            )
            print(f"      Content: {result.get('content', '')[:100]}...")
            print(f"      Subject: {result.get('subject_area', 'unknown')}")
            print(f"      Difficulty: {result.get('difficulty_level', 'unknown')}")

        # Test 4: Document analytics
        print("\n4. Testing Document Analytics")
        analytics = await document_integration_service.get_document_analytics(
            user_id="test_user_123", days=30
        )

        print(f"   ✅ Document analytics retrieved:")
        for key, value in analytics.items():
            print(f"      {key}: {value}")

        print("\n✅ All document integration tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_document_chunking():
    """Test document chunking functionality"""
    print("\n🔧 Testing Document Chunking")
    print("=" * 50)

    try:
        # Test text chunking
        long_text = "This is a long document with multiple paragraphs. " * 50
        chunks = await document_integration_service._chunk_text_content(long_text)

        print(f"   ✅ Text chunking completed: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            print(f"   Chunk {i+1}: {len(chunk.content)} characters")
            print(f"      Type: {chunk.chunk_type}")
            print(f"      Content: {chunk.content[:100]}...")

    except Exception as e:
        print(f"   ⚠️ Document chunking test failed: {e}")


async def main():
    """Main test function"""
    print("🚀 Starting Document Integration Service Tests")
    print("=" * 60)

    await test_document_integration()
    await test_document_chunking()

    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
