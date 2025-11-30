#!/usr/bin/env python3
"""
Test script for Serper API integration
Tests both search and scrape functionality
"""

import os
import sys
import json
import asyncio
import requests
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")


def get_serper_api_key() -> Optional[str]:
    """Get Serper API key from environment"""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("❌ SERPER_API_KEY not found in environment variables")
        print("   Set it with: export SERPER_API_KEY='your-api-key'")
        return None
    return api_key


def test_serper_search(api_key: str, query: str = "What is machine learning?") -> bool:
    """Test Serper search API"""
    print(f"\n🔍 Testing Serper Search API...")
    print(f"   Query: {query}")

    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 5})
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        response = requests.post(url, headers=headers, data=payload, timeout=30)

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # Print answer box if available
            if "answerBox" in result:
                answer_box = result["answerBox"]
                print(f"\n   📦 Answer Box:")
                if "answer" in answer_box:
                    print(f"      Answer: {answer_box['answer'][:200]}...")
                elif "snippet" in answer_box:
                    print(f"      Snippet: {answer_box['snippet'][:200]}...")

            # Print knowledge graph if available
            if "knowledgeGraph" in result:
                kg = result["knowledgeGraph"]
                print(f"\n   📚 Knowledge Graph:")
                if "title" in kg:
                    print(f"      Title: {kg['title']}")
                if "description" in kg:
                    print(f"      Description: {kg['description'][:200]}...")

            # Print organic results
            organic = result.get("organic", [])
            print(f"\n   📋 Organic Results: {len(organic)} found")
            for i, item in enumerate(organic[:3]):
                print(f"      {i+1}. {item.get('title', 'No title')}")
                print(f"         {item.get('snippet', 'No snippet')[:100]}...")
                print(f"         URL: {item.get('link', 'No link')}")

            print("\n   ✅ Search API test PASSED")
            return True
        else:
            print(f"   ❌ Search failed: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"   ❌ Search error: {e}")
        return False


def test_serper_scrape(
    api_key: str,
    url: str = "https://en.wikipedia.org/wiki/Python_(programming_language)",
) -> bool:
    """Test Serper scrape API"""
    print(f"\n🔗 Testing Serper Scrape API...")
    print(f"   URL: {url}")

    try:
        scrape_url = "https://scrape.serper.dev"
        payload = json.dumps({"url": url})
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        response = requests.post(scrape_url, headers=headers, data=payload, timeout=30)

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # Print available keys
            print(f"   Response keys: {list(result.keys())}")

            # Print title
            title = result.get("title", "")
            print(f"   Title: {title[:100] if title else 'None'}")

            # Print text content
            text_content = result.get("text", "")
            print(f"   Content length: {len(text_content)} characters")

            if text_content:
                print(f"   Content preview: {text_content[:300]}...")
                print("\n   ✅ Scrape API test PASSED")
                return True
            else:
                print("   ⚠️ No text content returned")
                # Check for other content types
                if "html" in result:
                    print(f"   HTML length: {len(result.get('html', ''))} characters")
                if "markdown" in result:
                    print(
                        f"   Markdown length: {len(result.get('markdown', ''))} characters"
                    )
                return False
        else:
            print(f"   ❌ Scrape failed: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"   ❌ Scrape error: {e}")
        return False


async def test_async_scrape(
    api_key: str, url: str = "https://en.wikipedia.org/wiki/Artificial_intelligence"
) -> bool:
    """Test async scraping (simulating the workflow service)"""
    print(f"\n⚡ Testing Async Scrape (simulating workflow)...")
    print(f"   URL: {url}")

    try:
        scrape_url = "https://scrape.serper.dev"
        payload = json.dumps({"url": url})
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                scrape_url, headers=headers, data=payload, timeout=30
            ),
        )

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            text_content = result.get("text", "")
            title = result.get("title", "")

            print(f"   Title: {title[:100] if title else 'None'}")
            print(f"   Content length: {len(text_content)} characters")

            if text_content:
                print("\n   ✅ Async scrape test PASSED")
                return True
            else:
                print("   ⚠️ No text content returned")
                return False
        else:
            print(f"   ❌ Async scrape failed: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"   ❌ Async scrape error: {e}")
        return False


def test_research_url(api_key: str) -> bool:
    """Test scraping a research/academic URL (like the user's ScienceDirect link)"""
    print(f"\n📚 Testing Research URL Scrape...")

    # Test with a simpler academic page first
    test_urls = [
        "https://arxiv.org/abs/2301.00234",  # ArXiv abstract page
        "https://www.nature.com/articles/d41586-023-00191-1",  # Nature article
    ]

    for url in test_urls:
        print(f"\n   Testing: {url}")

        try:
            scrape_url = "https://scrape.serper.dev"
            payload = json.dumps({"url": url})
            headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

            response = requests.post(
                scrape_url, headers=headers, data=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                text_content = result.get("text", "")
                title = result.get("title", "")

                if text_content and len(text_content) > 100:
                    print(f"   ✅ Success! Title: {title[:80]}...")
                    print(f"      Content: {len(text_content)} chars")
                else:
                    print(f"   ⚠️ Limited content: {len(text_content)} chars")
            else:
                print(f"   ❌ Failed with status {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return True


def main():
    """Run all Serper API tests"""
    print("=" * 60)
    print("🚀 Serper API Integration Test")
    print("=" * 60)

    # Get API key
    api_key = get_serper_api_key()
    if not api_key:
        print("\n❌ Cannot run tests without SERPER_API_KEY")
        return False

    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")

    results = []

    # Test 1: Search API
    results.append(("Search API", test_serper_search(api_key)))

    # Test 2: Scrape API
    results.append(("Scrape API", test_serper_scrape(api_key)))

    # Test 3: Async Scrape
    results.append(("Async Scrape", asyncio.run(test_async_scrape(api_key))))

    # Test 4: Research URLs
    results.append(("Research URLs", test_research_url(api_key)))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Serper API is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
