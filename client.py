"""
client.py — SQL-Genie inference client

Two ways to use:
  1. Python client (import and call directly)
  2. CLI (python client.py "How many orders were placed in Q3?")

Usage:
    python client.py "How many orders were placed in Q3?"
    python client.py "List all customers with more than 5 orders" --schema "orders(id, customer_id, amount, created_at), customers(id, name, email)"
    python client.py --interactive
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Iterator


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL    = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
DEFAULT_MODEL       = "sql-genie"
DEFAULT_TEMPERATURE = 0.1     # Low temp for deterministic SQL
DEFAULT_MAX_TOKENS  = 512
DEFAULT_SYSTEM      = (
    "You are an expert SQL engineer. "
    "Generate accurate, efficient SQL queries based on the user's request. "
    "Return only the SQL query, no explanation unless asked."
)


# ── OpenAI-compatible REST client ─────────────────────────────────────────────
class SQLGenieClient:
    """
    Thin wrapper around vLLM's OpenAI-compatible /v1/chat/completions endpoint.
    Works with any OpenAI-compatible server (vLLM, llama.cpp, Ollama, etc.)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str    = DEFAULT_MODEL,
        timeout: int  = 60,
    ):
        self.base_url         = base_url.rstrip("/")
        self.model            = model
        self.timeout          = timeout
        self.completions_url  = f"{self.base_url}/v1/chat/completions"
        self.models_url       = f"{self.base_url}/v1/models"

    # ── Health Check ──────────────────────────────────────────────────────────
    def is_healthy(self) -> bool:
        """Returns True if the vLLM server is reachable."""
        try:
            r = requests.get(self.models_url, timeout=5)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def wait_until_ready(self, timeout_seconds: int = 120):
        """Blocks until the server is healthy or timeout is reached."""
        print(f"⏳ Waiting for server at {self.base_url}...")
        start = time.time()
        while time.time() - start < timeout_seconds:
            if self.is_healthy():
                print("✅ Server is ready\n")
                return True
            time.sleep(3)
        raise TimeoutError(f"Server not ready after {timeout_seconds}s")

    # ── Core Generation ───────────────────────────────────────────────────────
    def generate(
        self,
        user_prompt: str,
        schema: str        = "",
        system_prompt: str = DEFAULT_SYSTEM,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int    = DEFAULT_MAX_TOKENS,
        stream: bool       = False,
    ) -> str | Iterator[str]:
        """
        Generate a SQL query from a natural language prompt.

        Args:
            user_prompt  : Natural language question e.g. "How many sales in Q3?"
            schema       : Optional table schema context
            system_prompt: Override the default SQL system prompt
            temperature  : Sampling temperature (0.1 = deterministic)
            max_tokens   : Max tokens to generate
            stream       : If True, yields token chunks as they arrive

        Returns:
            Full response string, or a generator of chunks if stream=True
        """
        # Build system content with optional schema
        system_content = system_prompt
        if schema:
            system_content += f"\n\nDatabase schema:\n{schema}"

        messages = [
            {"role": "system",  "content": system_content},
            {"role": "user",    "content": user_prompt},
        ]

        payload = {
            "model"      : self.model,
            "messages"   : messages,
            "temperature": temperature,
            "max_tokens" : max_tokens,
            "stream"     : stream,
        }

        if stream:
            return self._stream(payload)
        else:
            return self._complete(payload)

    def _complete(self, payload: dict) -> str:
        """Non-streaming completion."""
        response = requests.post(
            self.completions_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _stream(self, payload: dict) -> Iterator[str]:
        """Streaming completion — yields text chunks as they arrive."""
        with requests.post(
            self.completions_url,
            json=payload,
            stream=True,
            timeout=self.timeout,
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
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue

    # ── Convenience: Multi-turn SQL session ───────────────────────────────────
    def sql_session(self, schema: str = ""):
        """
        Interactive multi-turn SQL session.
        Maintains conversation history so follow-up questions work.
        """
        system_content = DEFAULT_SYSTEM
        if schema:
            system_content += f"\n\nDatabase schema:\n{schema}"

        history = [{"role": "system", "content": system_content}]

        print("SQL-Genie interactive session. Type 'exit' to quit.\n")
        if schema:
            print(f"Schema loaded: {schema[:100]}{'...' if len(schema) > 100 else ''}\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Exiting.")
                break

            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 Exiting.")
                break
            if not user_input:
                continue

            history.append({"role": "user", "content": user_input})

            payload = {
                "model"      : self.model,
                "messages"   : history,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens" : DEFAULT_MAX_TOKENS,
                "stream"     : True,
            }

            print("SQL-Genie: ", end="", flush=True)
            full_response = ""
            for chunk in self._stream(payload):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

            history.append({"role": "assistant", "content": full_response})


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SQL-Genie inference client")
    parser.add_argument("prompt",   nargs="?",  help="Natural language SQL prompt")
    parser.add_argument("--schema", default="", help="Table schema context")
    parser.add_argument("--url",    default=DEFAULT_BASE_URL, help="vLLM server URL")
    parser.add_argument("--stream", action="store_true",      help="Stream output tokens")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")
    parser.add_argument("--wait",   action="store_true",      help="Wait for server to be ready")
    args = parser.parse_args()

    client = SQLGenieClient(base_url=args.url)

    # Health check
    if args.wait:
        client.wait_until_ready()
    elif not client.is_healthy():
        print(f"❌ Server not reachable at {args.url}")
        print(f"   Start it with: python serve_vllm.py")
        print(f"   Or set VLLM_BASE_URL env var to the correct address.")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        client.sql_session(schema=args.schema)
        return

    # Single prompt mode
    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    print(f"Prompt : {args.prompt}")
    if args.schema:
        print(f"Schema : {args.schema}")
    print(f"Output : ", end="", flush=True)

    if args.stream:
        for chunk in client.generate(args.prompt, schema=args.schema, stream=True):
            print(chunk, end="", flush=True)
        print()
    else:
        result = client.generate(args.prompt, schema=args.schema)
        print(result)


if __name__ == "__main__":
    main()
