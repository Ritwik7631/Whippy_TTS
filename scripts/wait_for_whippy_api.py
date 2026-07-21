from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://localhost:4000"
HEALTH_PATH = "/api/health"
POLL_INTERVAL_SECONDS = 2
TIMEOUT_SECONDS = 180
REQUEST_TIMEOUT_SECONDS = 90
WSL_API_LOG = "/tmp/whippy-api.log"


def poll_health(base_url: str) -> int | None:
    url = f"{base_url.rstrip('/')}{HEALTH_PATH}"
    request = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except urllib.error.URLError:
        return None
    except TimeoutError:
        return None


def print_wsl_api_logs() -> None:
    import subprocess

    result = subprocess.run(
        [
            "wsl",
            "-d",
            "Ubuntu",
            "--",
            "bash",
            "-lc",
            f"tail -80 {WSL_API_LOG} 2>/dev/null || echo '(no log at {WSL_API_LOG})'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout.strip(), file=sys.stderr)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    deadline = time.time() + TIMEOUT_SECONDS
    last_status: int | str = "none"

    print(f"Polling {base_url}{HEALTH_PATH} for up to {TIMEOUT_SECONDS}s")
    print("Status meanings: timeout/connection failure=not ready, 422=booting, 200=ready")
    print()

    while time.time() < deadline:
        status = poll_health(base_url)
        if status is None:
            last_status = "timeout"
            print(f"[{time.strftime('%H:%M:%S')}] not ready (connection failure or timeout)")
        else:
            last_status = status
            print(f"[{time.strftime('%H:%M:%S')}] HTTP {status}")
            if status == 200:
                print()
                print("API ready.")
                return 0
            if status == 422:
                print("  -> listening but still booting")

        time.sleep(POLL_INTERVAL_SECONDS)

    print()
    print(f"API did not reach HTTP 200 within {TIMEOUT_SECONDS}s (last status: {last_status})")
    print()
    print("Latest API logs:", file=sys.stderr)
    print_wsl_api_logs()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
