"""
benchmark.py — Phase 2 performance benchmarking

Measures against 2026 inference engineer targets:
  - TTFT (Time to First Token) : < 15ms
  - ITL  (Inter-Token Latency) : < 8ms
  - Throughput                 : tokens/sec

Usage:
    python benchmark.py
    python benchmark.py --url http://<POD_IP>:8000 --requests 50
"""

import os
import sys
import time
import json
import statistics
import argparse
import requests
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Targets ───────────────────────────────────────────────────────────────────
TTFT_TARGET_MS  = 15.0    # Time to First Token < 15ms
ITL_TARGET_MS   = 8.0     # Inter-Token Latency < 8ms
POWER_TARGET_W  = 0.5     # Watts per 1k tokens (requires manual measurement)

DEFAULT_URL     = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
DEFAULT_MODEL   = "sql-genie"

# ── SQL Benchmark Prompts ─────────────────────────────────────────────────────
BENCHMARK_PROMPTS = [
    {
        "name"  : "simple_count",
        "prompt": "How many orders were placed in Q3 2024?",
        "schema": "orders(order_id, customer_id, amount, status, created_at)"
    },
    {
        "name"  : "join_query",
        "prompt": "Get the total revenue per customer, only for customers who spent more than $1000.",
        "schema": "orders(order_id, customer_id, amount, created_at), customers(customer_id, name, email)"
    },
    {
        "name"  : "window_function",
        "prompt": "Rank employees by salary within each department.",
        "schema": "employees(emp_id, name, dept_id, salary), departments(dept_id, dept_name)"
    },
    {
        "name"  : "subquery",
        "prompt": "Find products that have never been ordered.",
        "schema": "products(product_id, name, price), order_items(item_id, order_id, product_id, quantity)"
    },
    {
        "name"  : "cte_complex",
        "prompt": "Using a CTE, find the month-over-month revenue growth for the last 6 months.",
        "schema": "orders(order_id, amount, created_at)"
    },
    {
        "name"  : "multi_join",
        "prompt": "List the top 10 customers by total spend, with their most recent order date and number of orders.",
        "schema": "orders(order_id, customer_id, amount, created_at), customers(customer_id, name, city, country)"
    },
]


# ── Result Dataclass ──────────────────────────────────────────────────────────
@dataclass
class RequestResult:
    name          : str
    prompt        : str
    ttft_ms       : float         # Time to first token (ms)
    total_time_ms : float         # Total generation time (ms)
    output_tokens : int           # Number of tokens generated
    itl_ms        : float         # Average inter-token latency (ms)
    tokens_per_sec: float
    success       : bool
    error         : Optional[str] = None
    output        : str           = ""


# ── Single Streaming Request ──────────────────────────────────────────────────
def run_streaming_request(
    url: str,
    model: str,
    prompt: str,
    schema: str,
    name: str,
    max_tokens: int = 256,
) -> RequestResult:
    """
    Sends a single streaming request and measures TTFT and ITL precisely.
    """
    system = "You are an expert SQL engineer. Generate accurate SQL queries."
    if schema:
        system += f" Schema: {schema}"

    payload = {
        "model"      : model,
        "messages"   : [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens" : max_tokens,
        "stream"     : True,
    }

    token_times   = []
    output_chunks = []
    ttft_ms       = None

    try:
        request_start = time.perf_counter()

        with requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk   = json.loads(line)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        now = time.perf_counter()
                        if ttft_ms is None:
                            ttft_ms = (now - request_start) * 1000
                        token_times.append(now)
                        output_chunks.append(content)
                except json.JSONDecodeError:
                    continue

        total_time_ms  = (time.perf_counter() - request_start) * 1000
        output_tokens  = len(token_times)

        # ITL = average gap between consecutive tokens
        if len(token_times) >= 2:
            gaps   = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))]
            itl_ms = statistics.mean(gaps)
        else:
            itl_ms = 0.0

        tokens_per_sec = (output_tokens / total_time_ms * 1000) if total_time_ms > 0 else 0

        return RequestResult(
            name           = name,
            prompt         = prompt,
            ttft_ms        = ttft_ms or 0.0,
            total_time_ms  = total_time_ms,
            output_tokens  = output_tokens,
            itl_ms         = itl_ms,
            tokens_per_sec = tokens_per_sec,
            success        = True,
            output         = "".join(output_chunks),
        )

    except Exception as e:
        return RequestResult(
            name=name, prompt=prompt, ttft_ms=0, total_time_ms=0,
            output_tokens=0, itl_ms=0, tokens_per_sec=0,
            success=False, error=str(e)
        )


# ── Benchmark Runner ──────────────────────────────────────────────────────────
def run_benchmark(
    url        : str,
    model      : str,
    n_requests : int  = 20,
    concurrency: int  = 1,
    verbose    : bool = False,
) -> list[RequestResult]:
    """
    Runs the full benchmark suite.

    Args:
        n_requests  : Total number of requests to send
        concurrency : Number of parallel requests (1 = sequential)
    """
    # Repeat prompts to hit n_requests
    prompts = (BENCHMARK_PROMPTS * ((n_requests // len(BENCHMARK_PROMPTS)) + 1))[:n_requests]
    results = []

    print(f"Running {n_requests} requests (concurrency={concurrency})...\n")

    if concurrency == 1:
        for i, p in enumerate(prompts, 1):
            result = run_streaming_request(url, model, p["prompt"], p.get("schema",""), p["name"])
            results.append(result)
            status = "✅" if result.success else "❌"
            if verbose or not result.success:
                print(
                    f"  [{i:>3}/{n_requests}] {status} {result.name:<20} "
                    f"TTFT={result.ttft_ms:>6.1f}ms  "
                    f"ITL={result.itl_ms:>5.1f}ms  "
                    f"{result.tokens_per_sec:>6.1f} tok/s"
                )
            else:
                print(f"  [{i:>3}/{n_requests}] {status} {result.name}", end="\r")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    run_streaming_request, url, model,
                    p["prompt"], p.get("schema",""), p["name"]
                ): i for i, p in enumerate(prompts, 1)
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "✅" if result.success else "❌"
                print(f"  {status} {result.name:<20} TTFT={result.ttft_ms:.1f}ms  ITL={result.itl_ms:.1f}ms")

    print()
    return results


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(results: list[RequestResult]):
    successful = [r for r in results if r.success]
    failed     = [r for r in results if not r.success]

    if not successful:
        print("❌ All requests failed. Check server logs.")
        return

    ttfts  = [r.ttft_ms        for r in successful]
    itls   = [r.itl_ms         for r in successful if r.itl_ms > 0]
    tps    = [r.tokens_per_sec for r in successful]
    totals = [r.total_time_ms  for r in successful]

    def p(label, values, unit, target=None, lower_is_better=True):
        if not values:
            return
        avg = statistics.mean(values)
        p50 = statistics.median(values)
        p99 = sorted(values)[int(len(values) * 0.99)]
        hit = ""
        if target is not None:
            passed = avg <= target if lower_is_better else avg >= target
            hit = f"  {'✅ TARGET MET' if passed else '❌ ABOVE TARGET'} (target: {target}{unit})"
        print(f"  {label:<30} avg={avg:>8.2f}{unit}  p50={p50:>8.2f}{unit}  p99={p99:>8.2f}{unit}{hit}")

    print("=" * 75)
    print("BENCHMARK RESULTS — SQL-Genie Phase 2")
    print("=" * 75)
    print(f"  Requests   : {len(results)} total, {len(successful)} successful, {len(failed)} failed")
    print(f"  Server     : {DEFAULT_URL}")
    print()

    print("── Latency ──────────────────────────────────────────────────────────")
    p("TTFT (Time to First Token)", ttfts,  "ms", target=TTFT_TARGET_MS)
    p("ITL  (Inter-Token Latency)", itls,   "ms", target=ITL_TARGET_MS)
    p("Total generation time",      totals, "ms")
    print()

    print("── Throughput ───────────────────────────────────────────────────────")
    p("Tokens/sec (per request)", tps, " tok/s", lower_is_better=False)
    total_tokens    = sum(r.output_tokens for r in successful)
    total_time_secs = sum(r.total_time_ms for r in successful) / 1000
    if total_time_secs > 0:
        print(f"  {'Overall throughput':<30} {total_tokens / total_time_secs:.1f} tok/s (across all requests)")
    print()

    print("── Target Summary ───────────────────────────────────────────────────")
    avg_ttft = statistics.mean(ttfts)
    avg_itl  = statistics.mean(itls) if itls else float("inf")
    targets = [
        ("TTFT < 15ms",              avg_ttft <= TTFT_TARGET_MS),
        ("ITL < 8ms",                avg_itl  <= ITL_TARGET_MS),
        ("< 0.5W/1k tokens",         None),   # Requires external power meter
    ]
    for label, passed in targets:
        if passed is None:
            print(f"  ⚠️  {label:<30} requires external power meter")
        else:
            icon = "✅" if passed else "❌"
            print(f"  {icon}  {label:<30} avg={avg_ttft:.1f}ms" if "TTFT" in label else f"  {icon}  {label}")

    if failed:
        print(f"\n── Failed Requests ({len(failed)}) ──────────────────────────────────────")
        for r in failed:
            print(f"  ❌ {r.name}: {r.error}")

    print("=" * 75)


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SQL-Genie Phase 2 Benchmark")
    parser.add_argument("--url",         default=DEFAULT_URL,    help="vLLM server URL")
    parser.add_argument("--model",       default=DEFAULT_MODEL,  help="Model name")
    parser.add_argument("--requests",    default=20,  type=int,  help="Number of requests")
    parser.add_argument("--concurrency", default=1,   type=int,  help="Parallel requests")
    parser.add_argument("--verbose",     action="store_true",    help="Print every request")
    args = parser.parse_args()

    # Health check
    try:
        r = requests.get(f"{args.url}/v1/models", timeout=5)
        r.raise_for_status()
        print(f"✅ Server reachable at {args.url}\n")
    except Exception as e:
        print(f"❌ Cannot reach server at {args.url}: {e}")
        print(f"   Start it with: python serve_vllm.py")
        sys.exit(1)

    results = run_benchmark(
        url         = args.url,
        model       = args.model,
        n_requests  = args.requests,
        concurrency = args.concurrency,
        verbose     = args.verbose,
    )
    print_report(results)


if __name__ == "__main__":
    main()
