#!/usr/bin/env python3
"""
Benchmark script for Qwen3 Local System

Tests performance metrics:
- Latency by model tier
- Cache hit rates
- Throughput
- Resource utilization
"""

import asyncio
import time
import statistics
from typing import List, Dict
import httpx

BASE_URL = "http://localhost:8000"


async def benchmark_latency(queries: List[str]) -> Dict:
    """Benchmark latency for various queries."""
    latencies = []

    async with httpx.AsyncClient() as client:
        for query in queries:
            start = time.time()

            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.7
                }
            )

            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)

            if response.status_code == 200:
                print(f"✓ Query completed in {latency:.1f}ms")
            else:
                print(f"✗ Query failed: {response.status_code}")

    return {
        "avg_latency_ms": statistics.mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
    }


async def benchmark_cache():
    """Benchmark cache effectiveness."""
    # Send same query twice
    query = "What is the capital of France?"

    async with httpx.AsyncClient() as client:
        # First request (miss)
        start1 = time.time()
        await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": query}]}
        )
        latency1 = (time.time() - start1) * 1000

        # Wait a bit
        await asyncio.sleep(0.5)

        # Second request (should hit cache)
        start2 = time.time()
        await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": query}]}
        )
        latency2 = (time.time() - start2) * 1000

        speedup = latency1 / latency2 if latency2 > 0 else 0

        return {
            "first_request_ms": latency1,
            "cached_request_ms": latency2,
            "speedup": f"{speedup:.1f}x"
        }


async def main():
    """Run benchmarks."""
    print("=" * 60)
    print("Qwen3 Local System - Benchmark")
    print("=" * 60)
    print()

    # Test queries
    simple_queries = [
        "What is 2+2?",
        "List 3 colors",
        "What day comes after Monday?"
    ]

    complex_queries = [
        "Explain the theory of relativity in simple terms",
        "Compare and contrast Python and JavaScript",
        "Design a database schema for an e-commerce system"
    ]

    # Latency benchmark
    print("Testing simple queries...")
    simple_results = await benchmark_latency(simple_queries)

    print("\nTesting complex queries...")
    complex_results = await benchmark_latency(complex_queries)

    # Cache benchmark
    print("\nTesting cache effectiveness...")
    cache_results = await benchmark_cache()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nSimple Queries:")
    for key, value in simple_results.items():
        print(f"  {key}: {value:.1f}")

    print("\nComplex Queries:")
    for key, value in complex_results.items():
        print(f"  {key}: {value:.1f}")

    print("\nCache Performance:")
    for key, value in cache_results.items():
        print(f"  {key}: {value}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
