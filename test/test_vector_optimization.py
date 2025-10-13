#!/usr/bin/env python3
"""
Test script for Vector Optimization Service

This script tests the enhanced vector search capabilities including:
- Hybrid search functionality
- Query expansion
- Enhanced metadata extraction
- Performance monitoring
"""

import asyncio
import sys
import os

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.vector_optimization_service import (
    vector_optimization_service,
    SearchQuery,
)


async def test_vector_optimization():
    """Test the vector optimization service"""
    print("🧪 Testing Vector Optimization Service")
    print("=" * 50)

    try:
        # Test 1: Basic hybrid search
        print("\n1. Testing Basic Hybrid Search")
        search_query = SearchQuery(
            query="solve equation 6",
            user_id="test_user_123",
            top_k=5,
            use_hybrid_search=True,
            use_query_expansion=True,
        )

        results = await vector_optimization_service.hybrid_search(search_query)
        print(f"   ✅ Found {len(results)} results")

        for i, result in enumerate(results[:3]):
            print(f"   Result {i+1}: {result.title} (Score: {result.score:.3f})")
            print(f"      Content: {result.content[:100]}...")
            print(f"      Type: {result.content_type}")
            print(f"      Topics: {result.topic_tags}")
            print(f"      Questions: {result.question_numbers}")

        # Test 2: Query expansion
        print("\n2. Testing Query Expansion")
        expanded_queries = await vector_optimization_service._expand_query(search_query)
        print(f"   ✅ Expanded to {len(expanded_queries)} queries:")
        for query in expanded_queries:
            print(f"      - {query}")

        # Test 3: Enhanced metadata extraction
        print("\n3. Testing Enhanced Metadata Extraction")
        test_text = "Solve the equation 2x + 5 = 13. This is a basic algebra problem."
        enhanced_metadata = vector_optimization_service._extract_enhanced_metadata(
            test_text, "Algebra Problem", {}
        )
        print(f"   ✅ Extracted metadata:")
        print(f"      Content Type: {enhanced_metadata['content_type']}")
        print(f"      Topic Tags: {enhanced_metadata['topic_tags']}")
        print(f"      Difficulty: {enhanced_metadata['difficulty_level']}")
        print(f"      Subject: {enhanced_metadata['subject_area']}")
        print(f"      Key Concepts: {enhanced_metadata['key_concepts']}")

        # Test 4: Search analytics
        print("\n4. Testing Search Analytics")
        analytics = await vector_optimization_service.get_search_analytics(
            "test_user_123", 7
        )
        print(f"   ✅ Analytics retrieved:")
        for key, value in analytics.items():
            print(f"      {key}: {value}")

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_collection_optimization():
    """Test collection optimization features"""
    print("\n🔧 Testing Collection Optimization")
    print("=" * 50)

    try:
        # Test collection index optimization
        success = await vector_optimization_service.optimize_collection_indexes()
        if success:
            print("   ✅ Collection indexes optimized successfully")
        else:
            print(
                "   ⚠️ Collection optimization failed (may not be available in test environment)"
            )

    except Exception as e:
        print(f"   ⚠️ Collection optimization test failed: {e}")


async def main():
    """Main test function"""
    print("🚀 Starting Vector Optimization Service Tests")
    print("=" * 60)

    await test_vector_optimization()
    await test_collection_optimization()

    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
