from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


REQUIRED_ENV_VARS = (
    "WHIPPY_BASE_URL",
    "WHIPPY_API_KEY",
    "WHIPPY_AGENT_ID",
    "WHIPPY_ORGANIZATION_ID",
)

OPTIONAL_ENV_VARS = (
    "WHIPPY_USER_ID",
    "WHIPPY_CHANNEL_ID",
)


@dataclass(frozen=True)
class WhippyConfig:
    base_url: str
    api_key: str
    agent_id: str
    organization_id: str
    user_id: str | None = None
    channel_id: str | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> WhippyConfig:
        env = os.environ if environ is None else environ
        missing = [name for name in REQUIRED_ENV_VARS if not env.get(name, "").strip()]

        if missing:
            raise ValueError(
                "Missing required environment variables: "
                + ", ".join(missing)
            )

        return cls(
            base_url=env["WHIPPY_BASE_URL"].strip().rstrip("/"),
            api_key=env["WHIPPY_API_KEY"].strip(),
            agent_id=env["WHIPPY_AGENT_ID"].strip(),
            organization_id=env["WHIPPY_ORGANIZATION_ID"].strip(),
            user_id=_optional_env(env, "WHIPPY_USER_ID"),
            channel_id=_optional_env(env, "WHIPPY_CHANNEL_ID"),
        )


def _optional_env(env: dict[str, str], key: str) -> str | None:
    value = env.get(key, "").strip()
    return value or None


class WhippyApiError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        url: str | None = None,
        response_body: str | None = None,
        response_json: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.url = url
        self.response_body = response_body
        self.response_json = response_json


class WhippyClient:
    def __init__(self, config: WhippyConfig) -> None:
        self.config = config

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        url = (
            f"{self.config.base_url}/api/v2/agents/"
            f"{self.config.agent_id}/chat"
        )
        payload = json.dumps({"messages": messages}).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=payload,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "authorization": self.config.api_key,
                "organization_id": self.config.organization_id,
            },
        )

        try:
            with urllib.request.urlopen(request) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            raw_body = error.read().decode("utf-8", errors="replace")
            parsed_body = _parse_json(raw_body)
            message = _extract_error_message(parsed_body, raw_body)
            raise WhippyApiError(
                message,
                status=error.code,
                url=url,
                response_body=raw_body,
                response_json=parsed_body,
            ) from error
        except urllib.error.URLError as error:
            raise WhippyApiError(
                f"Network error calling Whippy API: {error.reason}",
                url=url,
            ) from error

        parsed = _parse_json(raw_body)
        if parsed is None:
            raise WhippyApiError(
                "Whippy API returned a non-JSON response.",
                url=url,
                response_body=raw_body,
            )

        data = parsed.get("data")
        if not isinstance(data, dict):
            raise WhippyApiError(
                "Whippy API response is missing a data object.",
                url=url,
                response_body=raw_body,
                response_json=parsed,
            )

        return data


def _parse_json(raw_body: str) -> Any | None:
    if not raw_body.strip():
        return None

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return None


def _extract_error_message(parsed_body: Any | None, raw_body: str) -> str:
    if isinstance(parsed_body, dict):
        error = parsed_body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])

        errors = parsed_body.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict) and first.get("description"):
                return str(first["description"])

    return raw_body.strip() or "Unknown Whippy API error"
