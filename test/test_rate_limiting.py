"""
Test Rate Limiting Implementation
Run this script to verify rate limiting is working correctly
"""

import asyncio
import httpx
import time
from datetime import datetime


async def test_basic_rate_limiting():
    """Test basic rate limiting functionality"""
    print("=" * 60)
    print("TEST 1: Basic Rate Limiting")
    print("=" * 60)

    base_url = "http://localhost:8000"
    endpoint = "/api/app/interactions"

    # Note: This requires authentication - replace with your token
    headers = {
        "Authorization": "Bearer YOUR_TOKEN_HERE",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        success_count = 0
        rate_limited_count = 0

        print(f"\nMaking 110 requests to {endpoint}")
        print("-" * 60)

        for i in range(110):
            try:
                response = await client.get(
                    f"{base_url}{endpoint}",
                    headers=headers,
                    timeout=5.0,
                )

                # Get rate limit headers
                limit = response.headers.get("X-RateLimit-Limit", "N/A")
                remaining = response.headers.get("X-RateLimit-Remaining", "N/A")
                reset = response.headers.get("X-RateLimit-Reset", "N/A")

                if response.status_code == 200:
                    success_count += 1
                    if i % 10 == 0:  # Print every 10th request
                        print(
                            f"Request {i+1:3d}: ✅ SUCCESS "
                            f"(Limit: {limit}, Remaining: {remaining})"
                        )

                elif response.status_code == 429:
                    rate_limited_count += 1
                    retry_after = response.headers.get("Retry-After", "N/A")
                    print(
                        f"Request {i+1:3d}: ⛔ RATE LIMITED "
                        f"(Retry after: {retry_after}s)"
                    )
                    break

                else:
                    print(f"Request {i+1:3d}: ⚠️  Status {response.status_code}")

            except Exception as e:
                print(f"Request {i+1:3d}: ❌ ERROR - {e}")
                break

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.01)

        print("-" * 60)
        print(f"\n📊 Results:")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ⛔ Rate Limited: {rate_limited_count}")
        print()


async def test_route_specific_limits():
    """Test different limits for different routes"""
    print("=" * 60)
    print("TEST 2: Route-Specific Limits")
    print("=" * 60)

    base_url = "http://localhost:8000"

    routes = [
        ("/graphql", "GraphQL Endpoint", 60),
        ("/api/stream/interactions", "Streaming Endpoint", 30),
    ]

    print("\nTesting different route limits:")
    print("-" * 60)

    for route, name, expected_limit in routes:
        async with httpx.AsyncClient() as client:
            try:
                # Make a single request to check headers
                response = await client.get(
                    f"{base_url}{route}",
                    timeout=5.0,
                )

                limit = response.headers.get("X-RateLimit-Limit", "N/A")
                remaining = response.headers.get("X-RateLimit-Remaining", "N/A")

                print(f"\n{name}:")
                print(f"   Route: {route}")
                print(f"   Expected Limit: {expected_limit}/min")
                print(f"   Actual Limit: {limit}/min")
                print(f"   Remaining: {remaining}")

                if str(limit) == str(expected_limit):
                    print(f"   Status: ✅ PASS")
                else:
                    print(f"   Status: ⚠️  DIFFERENT (may be configured differently)")

            except Exception as e:
                print(f"   Status: ❌ ERROR - {e}")

    print()


async def test_rate_limit_headers():
    """Test rate limit headers in responses"""
    print("=" * 60)
    print("TEST 3: Rate Limit Headers")
    print("=" * 60)

    base_url = "http://localhost:8000"
    endpoint = "/"  # Root endpoint

    print(f"\nChecking headers on {endpoint}")
    print("-" * 60)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}{endpoint}", timeout=5.0)

            # Check for rate limit headers
            headers_to_check = [
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
            ]

            print("\n📋 Response Headers:")
            for header in headers_to_check:
                value = response.headers.get(header)
                if value:
                    print(f"   ✅ {header}: {value}")

                    # Parse reset time if available
                    if header == "X-RateLimit-Reset" and value != "N/A":
                        try:
                            reset_time = datetime.fromtimestamp(int(value))
                            print(
                                f"      (Resets at: {reset_time.strftime('%H:%M:%S')})"
                            )
                        except:
                            pass
                else:
                    print(f"   ⚠️  {header}: Not present (endpoint may be exempt)")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    print()


async def test_performance():
    """Test rate limiter performance overhead"""
    print("=" * 60)
    print("TEST 4: Performance Test")
    print("=" * 60)

    base_url = "http://localhost:8000"
    endpoint = "/"

    print(f"\nMeasuring latency for {endpoint}")
    print("-" * 60)

    async with httpx.AsyncClient() as client:
        latencies = []

        # Warm-up
        for _ in range(5):
            await client.get(f"{base_url}{endpoint}", timeout=5.0)

        # Measure
        print("\nMaking 50 requests...")
        for i in range(50):
            start = time.perf_counter()
            try:
                response = await client.get(f"{base_url}{endpoint}", timeout=5.0)
                end = time.perf_counter()

                if response.status_code == 200:
                    latency_ms = (end - start) * 1000
                    latencies.append(latency_ms)

            except Exception:
                pass

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            print(f"\n📊 Performance Results:")
            print(f"   Requests: {len(latencies)}")
            print(f"   Average Latency: {avg_latency:.2f}ms")
            print(f"   Min Latency: {min_latency:.2f}ms")
            print(f"   Max Latency: {max_latency:.2f}ms")

            if avg_latency < 50:
                print(f"   Status: ✅ EXCELLENT (<50ms)")
            elif avg_latency < 100:
                print(f"   Status: ✅ GOOD (<100ms)")
            else:
                print(f"   Status: ⚠️  NEEDS OPTIMIZATION (>100ms)")

    print()


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RATE LIMITING TEST SUITE")
    print("=" * 60)
    print("\nMake sure your server is running on http://localhost:8000")
    print("Press Ctrl+C to cancel\n")

    try:
        # Check if server is running
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8000/", timeout=5.0)
                print("✅ Server is running\n")
            except Exception as e:
                print(f"❌ Server is not running: {e}")
                print("Please start the server first: python -m app.main\n")
                return

        # Run tests
        await test_rate_limit_headers()
        await test_performance()

        # Note: These tests require authentication
        # await test_basic_rate_limiting()
        # await test_route_specific_limits()

        print("=" * 60)
        print("TESTS COMPLETED")
        print("=" * 60)
        print("\nNote: Some tests require authentication and are commented out.")
        print("Uncomment and add your token to test authenticated endpoints.\n")

    except KeyboardInterrupt:
        print("\n\nTests cancelled by user\n")


if __name__ == "__main__":
    asyncio.run(main())
