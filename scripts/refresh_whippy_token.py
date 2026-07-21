from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
BASE_URL = "http://localhost:4000"


def http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30,
) -> tuple[int, dict | str]:
    body = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8")
        status = error.code
    except urllib.error.URLError as error:
        raise RuntimeError(f"Request failed: {error}") from error

    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def wait_for_health(timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            status, _ = http_json("GET", f"{BASE_URL}/api/health", timeout=5)
            if status == 200:
                print("API health: OK")
                return
        except RuntimeError:
            pass
        time.sleep(2)
    raise RuntimeError("API did not become healthy in time")


def login() -> str:
    status, data = http_json(
        "POST",
        f"{BASE_URL}/api/session",
        {
            "user": {
                "email": "testuser@whippy.co",
                "password": "Test1234!",
            }
        },
    )
    if status != 200 or not isinstance(data, dict):
        raise RuntimeError(f"Login failed ({status}): {data}")

    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"Login response missing access_token: {data}")
    print("Login: OK")
    return token


def update_env_token(token: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    updated = False
    for index, line in enumerate(lines):
        if line.startswith("WHIPPY_API_KEY="):
            lines[index] = f"WHIPPY_API_KEY={token}"
            updated = True
            break
    if not updated:
        lines.append(f"WHIPPY_API_KEY={token}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Updated .env WHIPPY_API_KEY")


def main() -> int:
    wait_for_health()
    token = login()
    update_env_token(token)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
