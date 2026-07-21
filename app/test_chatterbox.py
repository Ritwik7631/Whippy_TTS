from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

from app.config import load_voice_config, resolve_repo_path


OUTPUT_AUDIO = resolve_repo_path("outputs/chatterbox-test.wav")


def main() -> None:
    config = load_voice_config()

    reference_audio = resolve_repo_path(config["reference_audio_path"])
    generation_config = config["generation"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    if not reference_audio.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {reference_audio}"
        )

    OUTPUT_AUDIO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Chatterbox...", flush=True)

    model = ChatterboxTTS.from_pretrained(
        device="cuda"
    )

    text = (
        "Hi, this is the Whippy AI Recruiter. "
        "Is now a good time to complete your interview?"
    )

    print("Generating audio...", flush=True)

    waveform = model.generate(
        text,
        audio_prompt_path=str(reference_audio),
        exaggeration=generation_config["exaggeration"],
        cfg_weight=generation_config["cfg_weight"],
    )

    ta.save(
        str(OUTPUT_AUDIO),
        waveform.detach().cpu(),
        model.sr,
    )

    print(f"Saved audio to {OUTPUT_AUDIO.resolve()}", flush=True)


if __name__ == "__main__":
    main()
