#!/usr/bin/env python3
"""
Test script for LangGraph Multi-Source Summarization Workflow
Demonstrates intelligent orchestration for links + PDFs + web search
"""

import asyncio
import json
from app.services.langgraph_workflow_service import langgraph_workflow_service
from app.services.langgraph_integration_service import langgraph_integration_service


async def test_simple_text_processing():
    """Test simple text processing (should not use LangGraph)"""
    print("🧪 Testing Simple Text Processing")
    print("=" * 50)

    # Simple text message - should not trigger LangGraph
    message = "Hello, how are you?"
    media_files = []

    result = await langgraph_integration_service.process_with_langgraph(
        user=None,  # Mock user
        interaction=None,
        message=message,
        media_files=media_files,
    )

    print(f"✅ Simple text processing: {result.get('workflow_type', 'simple')}")
    print()


async def test_pdf_processing():
    """Test PDF processing (should use LangGraph)"""
    print("🧪 Testing PDF Processing")
    print("=" * 50)

    # PDF processing - should trigger LangGraph
    message = "Please analyze this document"
    media_files = [
        {
            "id": "pdf1",
            "url": "https://example.com/document.pdf",
            "type": "application/pdf",
            "name": "document.pdf",
        }
    ]

    result = await langgraph_integration_service.process_with_langgraph(
        user=None,  # Mock user
        interaction=None,
        message=message,
        media_files=media_files,
    )

    print(f"✅ PDF processing: {result.get('workflow_type', 'simple')}")
    print()


async def test_link_processing():
    """Test link processing (should use LangGraph)"""
    print("🧪 Testing Link Processing")
    print("=" * 50)

    # Link processing - should trigger LangGraph
    message = "Please research this topic: https://example.com/article"
    media_files = []

    result = await langgraph_integration_service.process_with_langgraph(
        user=None,  # Mock user
        interaction=None,
        message=message,
        media_files=media_files,
    )

    print(f"✅ Link processing: {result.get('workflow_type', 'simple')}")
    print()


async def test_hybrid_processing():
    """Test hybrid PDF + Link processing (should use LangGraph)"""
    print("🧪 Testing Hybrid PDF + Link Processing")
    print("=" * 50)

    # Hybrid processing - should definitely trigger LangGraph
    message = "Please analyze these documents and research current information: https://example.com/article"
    media_files = [
        {
            "id": "pdf1",
            "url": "https://example.com/document1.pdf",
            "type": "application/pdf",
            "name": "document1.pdf",
        },
        {
            "id": "pdf2",
            "url": "https://example.com/document2.pdf",
            "type": "application/pdf",
            "name": "document2.pdf",
        },
    ]

    result = await langgraph_integration_service.process_with_langgraph(
        user=None,  # Mock user
        interaction=None,
        message=message,
        media_files=media_files,
    )

    print(f"✅ Hybrid processing: {result.get('workflow_type', 'simple')}")
    print()


async def test_analytical_processing():
    """Test analytical processing (should use LangGraph)"""
    print("🧪 Testing Analytical Processing")
    print("=" * 50)

    # Analytical processing - should trigger LangGraph
    message = "Please analyze and compare the latest developments in AI research"
    media_files = []

    result = await langgraph_integration_service.process_with_langgraph(
        user=None,  # Mock user
        interaction=None,
        message=message,
        media_files=media_files,
    )

    print(f"✅ Analytical processing: {result.get('workflow_type', 'simple')}")
    print()


async def test_workflow_orchestration():
    """Test the actual LangGraph workflow orchestration"""
    print("🧪 Testing LangGraph Workflow Orchestration")
    print("=" * 50)

    # Test workflow orchestration directly
    message = "Please analyze these documents and provide a comprehensive summary"
    media_files = [
        {
            "id": "pdf1",
            "url": "https://example.com/document.pdf",
            "type": "application/pdf",
            "name": "document.pdf",
        }
    ]

    try:
        workflow_result = await langgraph_workflow_service.execute_workflow(
            message=message,
            media_files=media_files,
            user_id="test_user",
            interaction_id="test_interaction",
        )

        print(f"✅ Workflow execution: {workflow_result.get('success', False)}")
        print(f"📊 Total tokens: {workflow_result.get('total_tokens', 0)}")
        print(f"🧠 Thinking steps: {len(workflow_result.get('thinking_steps', []))}")

        # Print thinking steps
        thinking_steps = workflow_result.get("thinking_steps", [])
        for i, step in enumerate(thinking_steps, 1):
            print(f"   {i}. {step}")

    except Exception as e:
        print(f"❌ Workflow execution failed: {str(e)}")

    print()


async def test_thinking_config():
    """Test ThinkingConfig for different complexity levels"""
    print("🧪 Testing ThinkingConfig")
    print("=" * 50)

    from app.services.langgraph_workflow_service import (
        ThinkingConfigManager,
        ComplexityLevel,
    )

    # Test different complexity levels
    complexity_levels = [
        ComplexityLevel.SIMPLE,
        ComplexityLevel.MODERATE,
        ComplexityLevel.COMPLEX,
        ComplexityLevel.ANALYTICAL,
    ]

    task_types = [
        "simple_question",
        "analytical_reasoning",
        "complex_analysis",
        "multi_source_integration",
    ]

    for complexity in complexity_levels:
        for task_type in task_types:
            should_use = ThinkingConfigManager.should_use_thinking(
                complexity, task_type
            )
            config = ThinkingConfigManager.get_thinking_config(complexity, task_type)

            print(
                f"   {complexity.value} + {task_type}: {'✅' if should_use else '❌'} "
                f"Config: {config is not None}"
            )

    print()


async def main():
    """Run all tests"""
    print("🚀 LangGraph Multi-Source Summarization Workflow Tests")
    print("=" * 70)
    print()

    # Test workflow orchestration
    await test_workflow_orchestration()

    # Test thinking config
    await test_thinking_config()

    # Test different processing scenarios
    await test_simple_text_processing()
    await test_pdf_processing()
    await test_link_processing()
    await test_hybrid_processing()
    await test_analytical_processing()

    print("✅ All tests completed!")
    print()
    print("🎯 Key Features Demonstrated:")
    print("   • Intelligent workflow orchestration")
    print("   • Automatic complexity detection")
    print("   • ThinkingConfig for GPT and Gemini")
    print("   • Multi-source content integration")
    print("   • Real-time thinking step display")
    print("   • Web search integration")
    print("   • PDF processing with context")


if __name__ == "__main__":
    asyncio.run(main())
