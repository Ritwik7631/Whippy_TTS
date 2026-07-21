from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import load_voice_config, resolve_repo_path
from app.whippy_client import WhippyApiError, WhippyClient, WhippyConfig

TEST_MESSAGE = "Hello"
OUTPUT_AUDIO = resolve_repo_path("outputs/output.wav")


def _require_reference_audio(reference_audio: Path) -> None:
    if reference_audio.exists():
        return

    print("Reference audio is required for Chatterbox voice cloning.", file=sys.stderr)
    print(f"Missing file: {reference_audio}", file=sys.stderr)
    print(
        "Place an 8–15 second clean WAV clip at "
        f"{reference_audio.relative_to(ROOT_DIR)} "
        "(see voices/README.md).",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _require_cuda() -> None:
    if torch.cuda.is_available():
        return

    print("CUDA GPU is required for Chatterbox generation.", file=sys.stderr)
    print(
        "Run this script on a CUDA machine such as Google Colab with a GPU runtime.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> int:
    load_dotenv(ROOT_DIR / ".env")

    voice_config = load_voice_config()
    reference_audio = resolve_repo_path(voice_config["reference_audio_path"])
    generation_config = voice_config["generation"]

    _require_reference_audio(reference_audio)
    _require_cuda()

    try:
        whippy_config = WhippyConfig.from_env()
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        print(
            "Copy .env.example to .env and fill in values from your Whippy session.",
            file=sys.stderr,
        )
        return 1

    client = WhippyClient(whippy_config)
    messages = [{"role": "user", "content": TEST_MESSAGE}]

    print(f"Whippy base URL: {whippy_config.base_url}")
    print(f"Sending test message to Whippy agent: {TEST_MESSAGE!r}")

    try:
        data = client.chat(messages)
    except WhippyApiError as error:
        print(f"Whippy API request failed: {error}", file=sys.stderr)
        if "localhost" in whippy_config.base_url:
            print(
                "Colab cannot reach localhost on your PC. "
                "Expose the local API with a temporary Cloudflare Tunnel and "
                "set WHIPPY_BASE_URL to the tunnel URL.",
                file=sys.stderr,
            )
        return 1

    response_text = data.get("response")
    if not isinstance(response_text, str) or not response_text.strip():
        print("Whippy response is missing data.response text.", file=sys.stderr)
        return 1

    OUTPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Chatterbox model...")
    model = ChatterboxTTS.from_pretrained(device="cuda")

    print("Generating speech from Whippy response...")
    started_at = time.perf_counter()
    waveform = model.generate(
        response_text,
        audio_prompt_path=str(reference_audio),
        exaggeration=generation_config["exaggeration"],
        cfg_weight=generation_config["cfg_weight"],
    )
    generation_seconds = time.perf_counter() - started_at

    ta.save(
        str(OUTPUT_AUDIO),
        waveform.detach().cpu(),
        model.sr,
    )

    print()
    print("Whippy agent response:")
    print(response_text)
    print()
    print(f"Generation time: {generation_seconds:.2f}s")
    print(f"Output path: {OUTPUT_AUDIO.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
