#!/usr/bin/env python3
"""
Example usage of StudyGuru Web Search functionality with Gemini
Demonstrates both dedicated web search chain and web_search parameter usage
"""

import asyncio
from app.config.langchain_config import StudyGuruConfig


async def main():
    """Example of using web search functionality"""

    # Check if we're using Gemini model
    if not StudyGuruConfig.MODELS._is_gemini_model():
        print("❌ Web search functionality is only available with Gemini models")
        print("Please set LLM_MODEL=gemini in your environment variables")
        return

    print("🚀 StudyGuru Web Search Example")
    print("=" * 50)

    # Get the web search chain
    web_search_chain = StudyGuruConfig.CHAINS.get_web_search_chain()

    # Example questions that would benefit from web search
    questions = [
        "What are the latest developments in quantum computing in 2024?",
        "What is the current status of renewable energy adoption worldwide?",
        "Explain the latest research on artificial intelligence in education",
        "What are the current trends in machine learning for 2024?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 50)

        try:
            # Invoke the chain with the question
            response = await web_search_chain.ainvoke({"question": question})
            print(f"🤖 Response: {response}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

        print("\n" + "=" * 50)


def sync_example():
    """Synchronous example for testing"""

    # Check if we're using Gemini model
    if not StudyGuruConfig.MODELS._is_gemini_model():
        print("❌ Web search functionality is only available with Gemini models")
        print("Please set LLM_MODEL=gemini in your environment variables")
        return

    print("🚀 StudyGuru Web Search Example (Sync)")
    print("=" * 50)

    # Example 1: Using dedicated web search chain
    print("\n📋 Example 1: Dedicated Web Search Chain")
    print("-" * 50)

    web_search_chain = StudyGuruConfig.CHAINS.get_web_search_chain()
    question = "What are the latest developments in quantum computing in 2024?"

    print(f"📝 Question: {question}")
    print("-" * 30)

    try:
        response = web_search_chain.invoke({"question": question})
        print(f"🤖 Response: {response[:200]}...")  # Truncate for display
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Example 2: Using web_search parameter with chat model
    print("\n📋 Example 2: Chat Model with Web Search Enabled")
    print("-" * 50)

    try:
        # Get chat model with web search enabled (default)
        chat_model = StudyGuruConfig.MODELS.get_chat_model(web_search=True)

        # Create a simple prompt
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

        # Create chain
        chain = prompt | chat_model

        question2 = "What is the current status of renewable energy adoption worldwide?"
        print(f"📝 Question: {question2}")
        print("-" * 30)

        response2 = chain.invoke({"question": question2})
        print(f"🤖 Response: {response2.content[:200]}...")  # Truncate for display

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Example 3: Using vision model with web search
    print("\n📋 Example 3: Vision Model with Web Search Enabled")
    print("-" * 50)

    try:
        # Get vision model with web search enabled
        vision_model = StudyGuruConfig.MODELS.get_vision_model(web_search=True)

        question3 = (
            "Explain the latest research on artificial intelligence in education"
        )
        print(f"📝 Question: {question3}")
        print("-" * 30)

        response3 = vision_model.invoke([{"role": "user", "content": question3}])
        print(f"🤖 Response: {response3.content[:200]}...")  # Truncate for display

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Example 4: Disabling web search
    print("\n📋 Example 4: Chat Model with Web Search Disabled")
    print("-" * 50)

    try:
        # Get chat model with web search disabled
        chat_model_no_search = StudyGuruConfig.MODELS.get_chat_model(web_search=False)

        question4 = "What are the current trends in machine learning for 2024?"
        print(f"📝 Question: {question4}")
        print("-" * 30)

        response4 = chat_model_no_search.invoke(
            [{"role": "user", "content": question4}]
        )
        print(f"🤖 Response: {response4.content[:200]}...")  # Truncate for display

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    # Run synchronous example
    sync_example()

    # Uncomment to run async example
    # asyncio.run(main())
