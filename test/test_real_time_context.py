#!/usr/bin/env python3
"""
Test script for Real-Time Context Service
Tests the enhanced context update and consistency mechanisms
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.real_time_context_service import real_time_context_service


async def test_context_update_queue():
    """Test the context update queue functionality"""
    print("🧪 Testing Context Update Queue")
    print("=" * 50)

    try:
        # Test 1: Queue semantic summary update
        print("\n1. Testing Semantic Summary Update")
        semantic_task_id = await real_time_context_service.queue_context_update(
            user_id="test_user_123",
            interaction_id="test_interaction_456",
            conversation_id="test_conversation_789",
            update_type="semantic_summary",
            payload={
                "user_message": "What is photosynthesis?",
                "ai_response": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
            },
            priority=1,
        )
        print(f"   ✅ Queued semantic summary update: {semantic_task_id}")

        # Test 2: Queue embedding update
        print("\n2. Testing Embedding Update")
        embedding_task_id = await real_time_context_service.queue_context_update(
            user_id="test_user_123",
            interaction_id="test_interaction_456",
            conversation_id="test_conversation_789",
            update_type="embedding",
            payload={
                "conversation_id": "test_conversation_789",
                "text": "What is photosynthesis?",
                "title": "User question about photosynthesis",
                "metadata": {
                    "topic": "biology",
                    "difficulty": "beginner",
                    "subject": "science",
                },
            },
            priority=2,
        )
        print(f"   ✅ Queued embedding update: {embedding_task_id}")

        # Test 3: Queue document context update
        print("\n3. Testing Document Context Update")
        document_task_id = await real_time_context_service.queue_context_update(
            user_id="test_user_123",
            interaction_id="test_interaction_456",
            conversation_id="test_conversation_789",
            update_type="document_context",
            payload={
                "media_id": "test_media_123",
                "document_analysis": {
                    "document_type": "mcq",
                    "total_questions": 10,
                    "main_topics": ["photosynthesis", "plant biology"],
                    "difficulty_level": "intermediate",
                    "subject_area": "biology",
                },
            },
            priority=3,
        )
        print(f"   ✅ Queued document context update: {document_task_id}")

        # Test 4: Check queue status
        print("\n4. Testing Queue Status")
        await asyncio.sleep(1)  # Let some processing happen
        queue_status = await real_time_context_service.get_queue_status()
        print(f"   ✅ Queue status retrieved:")
        print(f"      Queue length: {queue_status['queue_length']}")
        print(f"      Processing tasks: {queue_status['processing_tasks']}")
        print(f"      Max concurrent: {queue_status['max_concurrent']}")

        # Test 5: Check individual task status
        print("\n5. Testing Task Status Check")
        semantic_status = await real_time_context_service.get_update_status(
            semantic_task_id
        )
        if semantic_status:
            print(
                f"   ✅ Semantic task status: {semantic_status.get('status', 'unknown')}"
            )
        else:
            print(f"   ⚠️ Semantic task not found (may have completed)")

        embedding_status = await real_time_context_service.get_update_status(
            embedding_task_id
        )
        if embedding_status:
            print(
                f"   ✅ Embedding task status: {embedding_status.get('status', 'unknown')}"
            )
        else:
            print(f"   ⚠️ Embedding task not found (may have completed)")

        print("\n✅ Context update queue tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Context update queue test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_consistency_checks():
    """Test consistency checking functionality"""
    print("\n🔍 Testing Consistency Checks")
    print("=" * 50)

    try:
        # Test consistency check for a user/interaction
        print("\n1. Testing Consistency Check")
        consistency_result = await real_time_context_service.ensure_consistency(
            user_id="test_user_123", interaction_id="test_interaction_456"
        )
        print(f"   ✅ Consistency check result: {consistency_result}")

        # Test cleanup of expired context
        print("\n2. Testing Context Cleanup")
        cleanup_count = await real_time_context_service.cleanup_expired_context(
            max_age_hours=1
        )
        print(f"   ✅ Cleaned up {cleanup_count} expired context entries")

        print("\n✅ Consistency check tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Consistency check test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_concurrent_updates():
    """Test concurrent update processing"""
    print("\n⚡ Testing Concurrent Updates")
    print("=" * 50)

    try:
        # Queue multiple updates simultaneously
        print("\n1. Queuing Multiple Concurrent Updates")
        task_ids = []

        for i in range(5):
            task_id = await real_time_context_service.queue_context_update(
                user_id=f"test_user_{i}",
                interaction_id=f"test_interaction_{i}",
                conversation_id=f"test_conversation_{i}",
                update_type="embedding",
                payload={
                    "conversation_id": f"test_conversation_{i}",
                    "text": f"Test message {i}",
                    "title": f"Test title {i}",
                    "metadata": {"test_index": i},
                },
                priority=2,
            )
            task_ids.append(task_id)
            print(f"   ✅ Queued update {i+1}: {task_id}")

        # Wait for processing
        print("\n2. Waiting for Processing")
        await asyncio.sleep(3)

        # Check status of all tasks
        print("\n3. Checking Task Statuses")
        for i, task_id in enumerate(task_ids):
            status = await real_time_context_service.get_update_status(task_id)
            if status:
                print(f"   Task {i+1} status: {status.get('status', 'unknown')}")
            else:
                print(f"   Task {i+1}: Completed or not found")

        # Check final queue status
        final_status = await real_time_context_service.get_queue_status()
        print(f"\n   Final queue length: {final_status['queue_length']}")
        print(f"   Final processing tasks: {final_status['processing_tasks']}")

        print("\n✅ Concurrent update tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Concurrent update test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_error_handling():
    """Test error handling and retry mechanisms"""
    print("\n🛡️ Testing Error Handling")
    print("=" * 50)

    try:
        # Test with invalid data to trigger errors
        print("\n1. Testing Error Handling with Invalid Data")
        error_task_id = await real_time_context_service.queue_context_update(
            user_id="test_user_error",
            interaction_id="nonexistent_interaction",
            conversation_id="test_conversation_error",
            update_type="semantic_summary",
            payload={
                "user_message": "",  # Empty message should cause issues
                "ai_response": "",  # Empty response should cause issues
            },
            priority=1,
        )
        print(f"   ✅ Queued error test task: {error_task_id}")

        # Wait for processing and retries
        print("\n2. Waiting for Error Processing and Retries")
        await asyncio.sleep(5)

        # Check status
        error_status = await real_time_context_service.get_update_status(error_task_id)
        if error_status:
            print(f"   ✅ Error task status: {error_status.get('status', 'unknown')}")
            print(f"   Retry count: {error_status.get('retry_count', 0)}")
            if error_status.get("error_message"):
                print(f"   Error message: {error_status['error_message']}")
        else:
            print(f"   ⚠️ Error task not found (may have failed completely)")

        print("\n✅ Error handling tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests"""
    print("🚀 Starting Real-Time Context Service Tests")
    print("=" * 60)

    await test_context_update_queue()
    await test_consistency_checks()
    await test_concurrent_updates()
    await test_error_handling()

    print("\n🎉 All real-time context service tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
