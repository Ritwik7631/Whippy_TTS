from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import load_voice_config, resolve_repo_path
from app.whippy_client import WhippyApiError, WhippyClient, WhippyConfig

DEFAULT_PROMPT = (
    "Hi, I recently applied for a position. "
    "Can you tell me what role this interview is for?"
)
RESULTS_PATH = resolve_repo_path("benchmark/results/latest.json")
OUTPUT_ORIGINAL = resolve_repo_path("outputs/benchmark_original.wav")
OUTPUT_TURBO = resolve_repo_path("outputs/benchmark_turbo.wav")

FEASIBILITY_DISCLAIMER = (
    "Provisional engineering thresholds only. "
    "This benchmark does not automatically determine suitability for live calls."
)


def classify_real_time_factor(real_time_factor: float) -> str:
    if real_time_factor < 0.5:
        return "promising"
    if real_time_factor <= 1.0:
        return "potentially usable but needs optimization"
    return "not ready for live calls without chunking, streaming, or another model"


def _require_cuda() -> tuple[int, str]:
    if not torch.cuda.is_available():
        print("CUDA GPU is required for Chatterbox generation.", file=sys.stderr)
        print(
            "Run this script on a CUDA machine such as Google Colab with a GPU runtime.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    device_index = torch.cuda.current_device()
    return device_index, torch.cuda.get_device_name(device_index)


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


def _audio_duration_seconds(waveform: torch.Tensor, sample_rate: int) -> float:
    num_samples = waveform.shape[-1]
    return num_samples / sample_rate


def _word_count(text: str) -> int:
    return len(text.split())


def _ensure_float32_reference_path(reference_audio: Path) -> str:
    """Return a float32 reference clip path for Chatterbox Turbo on Colab."""
    prepared_path = resolve_repo_path("outputs/_reference_float32.wav")
    prepared_path.parent.mkdir(parents=True, exist_ok=True)

    waveform, sample_rate = ta.load(str(reference_audio))
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)

    ta.save(str(prepared_path), waveform, sample_rate)
    return str(prepared_path)


def _call_whippy(
    client: WhippyClient,
    prompt: str,
) -> tuple[str, float]:
    started_at = time.perf_counter()
    try:
        data = client.chat([{"role": "user", "content": prompt}])
    except WhippyApiError as error:
        print(f"Whippy API request failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    latency_seconds = time.perf_counter() - started_at
    response_text = data.get("response")
    if not isinstance(response_text, str) or not response_text.strip():
        print("Whippy response is missing data.response text.", file=sys.stderr)
        raise SystemExit(1)

    return response_text, latency_seconds


def _benchmark_model(
    *,
    model_name: str,
    model_type: str,
    response_text: str,
    reference_audio_path: str,
    generation_config: dict[str, Any],
    output_path: Path,
    whippy_latency_seconds: float,
) -> dict[str, Any]:
    if model_type == "original":
        load_started_at = time.perf_counter()
        model = ChatterboxTTS.from_pretrained(device="cuda")
        model_loading_seconds = time.perf_counter() - load_started_at

        generation_started_at = time.perf_counter()
        waveform = model.generate(
            response_text,
            audio_prompt_path=reference_audio_path,
            exaggeration=generation_config["exaggeration"],
            cfg_weight=generation_config["cfg_weight"],
        )
    elif model_type == "turbo":
        load_started_at = time.perf_counter()
        model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        model_loading_seconds = time.perf_counter() - load_started_at

        generation_started_at = time.perf_counter()
        waveform = model.generate(
            response_text,
            audio_prompt_path=reference_audio_path,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    tts_generation_seconds = time.perf_counter() - generation_started_at

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_started_at = time.perf_counter()
    ta.save(
        str(output_path),
        waveform.detach().cpu(),
        model.sr,
    )
    save_seconds = time.perf_counter() - save_started_at

    audio_duration_seconds = _audio_duration_seconds(waveform, model.sr)
    real_time_factor = tts_generation_seconds / audio_duration_seconds
    total_end_to_end_seconds = (
        whippy_latency_seconds
        + model_loading_seconds
        + tts_generation_seconds
        + save_seconds
    )

    return {
        "name": model_name,
        "model_type": model_type,
        "model_loading_seconds": round(model_loading_seconds, 3),
        "tts_generation_seconds": round(tts_generation_seconds, 3),
        "save_seconds": round(save_seconds, 3),
        "audio_duration_seconds": round(audio_duration_seconds, 3),
        "real_time_factor": round(real_time_factor, 3),
        "total_end_to_end_seconds": round(total_end_to_end_seconds, 3),
        "output_path": str(output_path.relative_to(ROOT_DIR)),
        "sample_rate": model.sr,
        "feasibility": classify_real_time_factor(real_time_factor),
    }


def run_benchmark(
    *,
    prompt: str = DEFAULT_PROMPT,
    results_path: Path = RESULTS_PATH,
) -> dict[str, Any]:
    load_dotenv(ROOT_DIR / ".env")

    voice_config = load_voice_config()
    reference_audio = resolve_repo_path(voice_config["reference_audio_path"])
    generation_config = voice_config["generation"]

    _require_reference_audio(reference_audio)
    cuda_device, gpu_name = _require_cuda()

    try:
        whippy_config = WhippyConfig.from_env()
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        print(
            "Copy .env.example to .env and fill in values from your Whippy session.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error

    client = WhippyClient(whippy_config)
    response_text, whippy_latency_seconds = _call_whippy(client, prompt)
    reference_audio_path = _ensure_float32_reference_path(reference_audio)

    original_result = _benchmark_model(
        model_name="original",
        model_type="original",
        response_text=response_text,
        reference_audio_path=reference_audio_path,
        generation_config=generation_config,
        output_path=OUTPUT_ORIGINAL,
        whippy_latency_seconds=whippy_latency_seconds,
    )

    torch.cuda.empty_cache()

    turbo_result = _benchmark_model(
        model_name="turbo",
        model_type="turbo",
        response_text=response_text,
        reference_audio_path=reference_audio_path,
        generation_config=generation_config,
        output_path=OUTPUT_TURBO,
        whippy_latency_seconds=whippy_latency_seconds,
    )

    results = {
        "benchmark_version": "1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "prompt": prompt,
        "disclaimer": FEASIBILITY_DISCLAIMER,
        "whippy": {
            "base_url": whippy_config.base_url,
            "api_latency_seconds": round(whippy_latency_seconds, 3),
            "response_text": response_text,
            "response_character_count": len(response_text),
            "response_word_count": _word_count(response_text),
        },
        "environment": {
            "cuda_available": True,
            "cuda_device": cuda_device,
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
        },
        "voice_config": {
            "reference_audio_path": voice_config["reference_audio_path"],
            "exaggeration": generation_config["exaggeration"],
            "cfg_weight": generation_config["cfg_weight"],
        },
        "models": {
            "original": original_result,
            "turbo": turbo_result,
        },
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    return results


def print_results(results: dict[str, Any]) -> None:
    whippy = results["whippy"]
    environment = results["environment"]

    print()
    print("Latency Feasibility Benchmark")
    print("=" * 72)
    print(f"Prompt: {results['prompt']}")
    print(f"GPU: {environment['gpu_name']} (device {environment['cuda_device']})")
    print(f"torch: {environment['torch_version']}")
    print()
    print("Whippy Chat API")
    print(f"  API latency: {whippy['api_latency_seconds']:.3f}s")
    print(f"  Response characters: {whippy['response_character_count']}")
    print(f"  Response words: {whippy['response_word_count']}")
    print(f"  Response text: {whippy['response_text']}")
    print()

    for model_key in ("original", "turbo"):
        model = results["models"][model_key]
        print(model["name"].upper())
        print(f"  Model loading: {model['model_loading_seconds']:.3f}s")
        print(f"  TTS generation: {model['tts_generation_seconds']:.3f}s")
        print(f"  Audio duration: {model['audio_duration_seconds']:.3f}s")
        print(f"  Real-time factor: {model['real_time_factor']:.3f}")
        print(
            "  Time from user message to audio file: "
            f"{model['total_end_to_end_seconds']:.3f}s"
        )
        print(f"  Output: {model['output_path']}")
        print(f"  Feasibility: {model['feasibility']}")
        print()

    print(FEASIBILITY_DISCLAIMER)
    print(f"Saved results to: {RESULTS_PATH.relative_to(ROOT_DIR)}")


def main() -> int:
    results = run_benchmark()
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
