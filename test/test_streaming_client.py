#!/usr/bin/env python3
"""
Test client for the StudyGuru streaming API
"""

import asyncio
import aiohttp
import json
import sys
from typing import Optional


class StudyGuruStreamingClient:
    """Client for testing the StudyGuru streaming API"""

    def __init__(
        self, base_url: str = "http://localhost:8000", token: Optional[str] = None
    ):
        self.base_url = base_url
        self.token = token
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def stream_conversation(
        self,
        message: str,
        interaction_id: Optional[str] = None,
        media_files: Optional[list] = None,
        max_tokens: Optional[int] = None,
    ):
        """Stream a conversation using the full API"""

        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.base_url}/api/stream/stream-conversation"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {
            "message": message,
            "interaction_id": interaction_id,
            "media_files": media_files,
            "max_tokens": max_tokens,
        }

        print(f"🚀 Starting streaming conversation...")
        print(f"📝 Message: {message}")
        print(f"🔗 URL: {url}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
        print("-" * 50)

        try:
            async with self.session.post(
                url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Error {response.status}: {error_text}")
                    return

                print("✅ Connected to streaming endpoint")
                print("📡 Streaming response:")
                print("-" * 50)

                full_response = ""
                async for line in response.content:
                    line = line.decode("utf-8").strip()

                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove 'data: ' prefix

                        try:
                            data = json.loads(data_str)
                            event_type = data.get("type", "unknown")

                            if event_type == "metadata":
                                print(f"📋 Metadata: {data}")
                            elif event_type == "token":
                                content = data.get("content", "")
                                print(content, end="", flush=True)
                                full_response += content
                            elif event_type == "complete":
                                print(f"\n\n✅ Complete: {data}")
                                break
                            elif event_type == "error":
                                print(
                                    f"\n❌ Error: {data.get('error', 'Unknown error')}"
                                )
                                break
                            else:
                                print(f"\n🔍 Unknown event: {data}")

                        except json.JSONDecodeError:
                            print(f"⚠️  Invalid JSON: {data_str}")

                print(f"\n\n📄 Full response: {full_response}")
                return full_response

        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

    async def stream_simple(self, message: str, interaction_id: Optional[str] = None):
        """Stream a simple conversation using the simplified API"""

        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.base_url}/api/stream/stream-simple"

        headers = {
            "Accept": "text/event-stream",
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        params = {
            "message": message,
        }

        if interaction_id:
            params["interaction_id"] = interaction_id

        print(f"🚀 Starting simple streaming conversation...")
        print(f"📝 Message: {message}")
        print(f"🔗 URL: {url}")
        print("-" * 50)

        try:
            async with self.session.get(
                url, params=params, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Error {response.status}: {error_text}")
                    return

                print("✅ Connected to simple streaming endpoint")
                print("📡 Streaming response:")
                print("-" * 50)

                full_response = ""
                async for line in response.content:
                    line = line.decode("utf-8").strip()

                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove 'data: ' prefix

                        try:
                            data = json.loads(data_str)
                            event_type = data.get("type", "unknown")

                            if event_type == "start":
                                print(f"🚀 {data.get('message', 'Starting...')}")
                            elif event_type == "token":
                                content = data.get("content", "")
                                print(content, end="", flush=True)
                                full_response += content
                            elif event_type == "complete":
                                print(f"\n\n✅ Complete!")
                                break
                            elif event_type == "error":
                                print(
                                    f"\n❌ Error: {data.get('error', 'Unknown error')}"
                                )
                                break
                            else:
                                print(f"\n🔍 Unknown event: {data}")

                        except json.JSONDecodeError:
                            print(f"⚠️  Invalid JSON: {data_str}")

                print(f"\n\n📄 Full response: {full_response}")
                return full_response

        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

    async def health_check(self):
        """Check if the streaming service is healthy"""

        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.base_url}/api/stream/health"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False


async def main():
    """Main test function"""

    # You can set a token here for authenticated requests
    # token = "your-jwt-token-here"
    token = None

    print("🧪 StudyGuru Streaming API Test Client")
    print("=" * 50)

    async with StudyGuruStreamingClient(token=token) as client:
        # Test health check
        print("1. Testing health check...")
        health_ok = await client.health_check()
        print()

        if not health_ok:
            print("❌ Service is not healthy. Exiting.")
            return

        # Test simple streaming
        print("2. Testing simple streaming...")
        await client.stream_simple("What is 2+2? Please explain briefly.")
        print("\n")

        # Test full streaming (if you have a token)
        if token:
            print("3. Testing full streaming...")
            await client.stream_conversation(
                message="Hello, can you help me with a math problem?", max_tokens=1000
            )
        else:
            print("3. Skipping full streaming test (no token provided)")
            print("   To test full streaming, set a valid JWT token in the script")

    print("\n🎉 Test completed!")


if __name__ == "__main__":
    # Run the test
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
