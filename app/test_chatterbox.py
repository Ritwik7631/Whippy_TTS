from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS


REFERENCE_AUDIO = Path("voices/reference.wav")
OUTPUT_AUDIO = Path("outputs/chatterbox-test.wav")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Run this script in Google Colab "
            "with a GPU runtime enabled."
        )

    if not REFERENCE_AUDIO.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {REFERENCE_AUDIO}"
        )

    OUTPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Chatterbox Turbo...")
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    text = (
        "Hi, this is the Whippy AI Recruiter. "
        "Is now a good time to complete your interview?"
    )

    print("Generating audio...")
    waveform = model.generate(
        text,
        audio_prompt_path=str(REFERENCE_AUDIO),
    )

    ta.save(
        str(OUTPUT_AUDIO),
        waveform.detach().cpu(),
        model.sr,
    )

    print(f"Saved audio to {OUTPUT_AUDIO}")
    print(f"Sample rate: {model.sr}")


if __name__ == "__main__":
    main()
