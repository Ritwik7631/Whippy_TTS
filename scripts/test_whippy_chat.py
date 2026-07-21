from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.whippy_client import WhippyApiError, WhippyClient, WhippyConfig


def main() -> int:
    load_dotenv(ROOT_DIR / ".env")

    try:
        config = WhippyConfig.from_env()
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1

    client = WhippyClient(config)
    messages = [{"role": "user", "content": "Hello"}]

    print(f"POST /api/v2/agents/{config.agent_id}/chat")
    print(f"Base URL: {config.base_url}")
    print(f"Organization ID: {config.organization_id}")
    print("Request body:")
    print(json.dumps({"messages": messages}, indent=2))
    print()

    try:
        data = client.chat(messages)
    except WhippyApiError as error:
        print("Whippy API request failed.", file=sys.stderr)
        print(f"Error: {error}", file=sys.stderr)

        if error.url:
            print(f"URL: {error.url}", file=sys.stderr)
        if error.status is not None:
            print(f"HTTP status: {error.status}", file=sys.stderr)
        if error.response_json is not None:
            print("Response JSON:", file=sys.stderr)
            print(json.dumps(error.response_json, indent=2), file=sys.stderr)
        elif error.response_body:
            print("Response body:", file=sys.stderr)
            print(error.response_body, file=sys.stderr)

        return 1

    assistant_response = data.get("response")
    print("Assistant response:")
    print(assistant_response)

    if data.get("messages"):
        print()
        print("Full messages returned:")
        print(json.dumps(data["messages"], indent=2))

    if data.get("usage"):
        print()
        print("Token usage:")
        print(json.dumps(data["usage"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
