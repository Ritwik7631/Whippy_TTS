# Whippy_TTS

Text-to-speech pipeline that combines a **Whippy agent chat response** with **Chatterbox** voice cloning.

Proven flow:

```
user text → Whippy POST /api/v2/agents/:id/chat → agent response text → Chatterbox → output.wav
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `app/whippy_client.py` | Whippy chat HTTP client |
| `app/config.py` | Repo-root path helpers + voice config loader |
| `config/voice_config.json` | Chatterbox generation settings |
| `scripts/test_whippy_chat.py` | Text-only Whippy integration test |
| `scripts/test_whippy_to_speech.py` | End-to-end Whippy → Chatterbox test |
| `voices/reference.wav` | Reference clip for voice cloning (not in Git) |
| `outputs/output.wav` | Generated speech output (not in Git) |

All scripts resolve paths from the repository root, so they work the same on Windows, Linux, and Colab.

## Local setup (Windows)

```powershell
cd C:\Dev\chatterbox-poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Fill in `.env` from your local Whippy session. See `.env.example` for required variables.

Place your reference clip at `voices/reference.wav` (see `voices/README.md`).

### Text-only Whippy test

```powershell
.\.venv\Scripts\python.exe scripts\test_whippy_chat.py
```

### Full speech test (requires CUDA GPU + reference.wav)

```powershell
.\.venv\Scripts\python.exe scripts\test_whippy_to_speech.py
```

---

## Google Colab end-to-end test

Colab provides the CUDA GPU. Your Whippy API stays on your local machine, so Colab cannot use `http://localhost:4000` directly. Use a **temporary Cloudflare Tunnel** to expose only port `4000` for the test.

### Step A — Start local Whippy API

On your machine (WSL recommended for the Whippy API):

```bash
cd "/mnt/c/Users/ritwi/OneDrive/Documents/Whippy Dashboard/whippy-app/api"
./start_dev_server.sh
```

Confirm the API responds locally:

```powershell
cd C:\Dev\chatterbox-poc
.\.venv\Scripts\python.exe scripts\wait_for_whippy_api.py
```

### Step B — Expose local API with Cloudflare Tunnel

Install `cloudflared` if needed:

- Windows: `winget install Cloudflare.cloudflared`
- macOS: `brew install cloudflared`
- Linux/WSL: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

Run the tunnel **on the same machine where the API listens on port 4000**:

```bash
cloudflared tunnel --url http://localhost:4000
```

Cloudflare prints a temporary public URL like:

```text
https://random-words-here.trycloudflare.com
```

Set in Colab:

```env
WHIPPY_BASE_URL=https://random-words-here.trycloudflare.com
```

Rules:

- No trailing slash
- No `/api` suffix
- Use the `https://` URL exactly as printed

#### Security precautions

- The tunnel URL is **public to anyone who has it** while the tunnel is running.
- Use it only for a short test, then **Ctrl+C** the tunnel.
- The URL changes every time you restart `cloudflared`.
- Put your Pow session token only in Colab secrets / `.env`, never in the notebook as plain text.
- Rotate the session token after testing if it was exposed.
- Keep the tunnel scoped to `http://localhost:4000` only — do not expose other local services.
- Use a dev org/agent, not production credentials.

### Step C — Get Whippy credentials locally

Do **not** invent values. Obtain them from your running local Whippy stack:

| Variable | How to obtain |
| --- | --- |
| `WHIPPY_BASE_URL` | Tunnel URL from Step B (Colab) or `http://localhost:4000` (local) |
| `WHIPPY_API_KEY` | Pow session `access_token` from `POST /api/session` |
| `WHIPPY_AGENT_ID` | Agent UUID from your local database or dashboard |
| `WHIPPY_ORGANIZATION_ID` | Organization UUID for the authenticated user |

Session login body shape:

```json
{
  "user": {
    "email": "your-dev-user@example.com",
    "password": "your-dev-password"
  }
}
```

`WHIPPY_USER_ID` and `WHIPPY_CHANNEL_ID` are optional for this test.

---

## Colab notebook cells

Open [Google Colab](https://colab.research.google.com/), choose **Runtime → Change runtime type → T4 GPU** (or better), then run:

### Cell 1 — Clone the repository

Push your branch to GitHub before cloning if you have local changes Colab should see.

```python
!git clone https://github.com/Ritwik7631/Whippy_TTS.git
%cd Whippy_TTS
!git checkout feature/whippy-integration
```

### Cell 2 — Verify Python and GPU

```python
import sys
import torch

print("Python:", sys.version)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Colab ships with Python 3.10+ and a CUDA-enabled PyTorch build. You normally do **not** need to create a separate Python 3.11 environment or reinstall PyTorch.

### Cell 3 — Install Python dependencies

```python
!pip install -q -r requirements.txt
```

### Cell 4 — Upload reference audio

```python
from pathlib import Path
from google.colab import files

Path("voices").mkdir(parents=True, exist_ok=True)
uploaded = files.upload()

if "reference.wav" not in uploaded:
    raise RuntimeError("Upload a file named reference.wav")

Path("reference.wav").rename("voices/reference.wav")
print("Saved:", Path("voices/reference.wav").resolve())
```

Alternatively, copy the file from Google Drive:

```python
from pathlib import Path

Path("voices").mkdir(parents=True, exist_ok=True)
# !cp "/content/drive/MyDrive/path/to/reference.wav" "voices/reference.wav"
```

### Cell 5 — Create `.env` securely

Prefer Colab secrets (**🔑 Secrets** in the left sidebar). Add:

- `WHIPPY_BASE_URL`
- `WHIPPY_API_KEY`
- `WHIPPY_AGENT_ID`
- `WHIPPY_ORGANIZATION_ID`

Then run:

```python
from pathlib import Path
from google.colab import userdata

def require_secret(name: str) -> str:
    try:
        value = userdata.get(name).strip()
    except userdata.SecretNotFoundError as error:
        raise RuntimeError(f"Missing Colab secret: {name}") from error
    if not value:
        raise RuntimeError(f"Colab secret is empty: {name}")
    return value

env_lines = [
    f"WHIPPY_BASE_URL={require_secret('WHIPPY_BASE_URL')}",
    f"WHIPPY_API_KEY={require_secret('WHIPPY_API_KEY')}",
    f"WHIPPY_AGENT_ID={require_secret('WHIPPY_AGENT_ID')}",
    f"WHIPPY_ORGANIZATION_ID={require_secret('WHIPPY_ORGANIZATION_ID')}",
]

Path(".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
print("Wrote .env with", len(env_lines), "variables.")
```

Set these Colab secrets before running the cell:

| Secret | Example |
| --- | --- |
| `WHIPPY_BASE_URL` | `https://random-words-here.trycloudflare.com` |
| `WHIPPY_API_KEY` | Your Pow session access token |
| `WHIPPY_AGENT_ID` | Your agent UUID |
| `WHIPPY_ORGANIZATION_ID` | Your organization UUID |

Manual fallback (avoid in shared notebooks):

```python
from pathlib import Path

Path(".env").write_text("""\
WHIPPY_BASE_URL=
WHIPPY_API_KEY=
WHIPPY_AGENT_ID=
WHIPPY_ORGANIZATION_ID=
""", encoding="utf-8")
```

### Cell 6 — Run the full end-to-end test

One command:

```python
!python scripts/test_whippy_to_speech.py
```

Expected output:

- Whippy base URL
- Agent response text
- Chatterbox generation time
- Absolute path to `outputs/output.wav`

### Cell 7 — Listen to the generated audio

```python
from IPython.display import Audio, display
from pathlib import Path

output_path = Path("outputs/output.wav")
if not output_path.exists():
    raise FileNotFoundError(output_path)

display(Audio(filename=str(output_path), autoplay=False))
```

Optional download:

```python
from google.colab import files

files.download("outputs/output.wav")
```

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `Reference audio not found` | Upload `voices/reference.wav` before running the script |
| `CUDA GPU is required` | Colab runtime is CPU-only — switch to a GPU runtime |
| `Network error calling Whippy API` | Tunnel not running, wrong `WHIPPY_BASE_URL`, or local API down |
| `missing authentication token` | Local API missing `BRAINTRUST_API_KEY` in `api/.env` |
| HTTP 401 from Whippy | Expired session token — log in again and update `WHIPPY_API_KEY` |
| Colab cannot reach localhost | Expected — use Cloudflare Tunnel URL in `WHIPPY_BASE_URL` |

## Voice configuration

Generation settings live in `config/voice_config.json`:

```json
{
  "reference_audio_path": "voices/reference.wav",
  "generation": {
    "exaggeration": 0.45,
    "cfg_weight": 0.3
  }
}
```

## What is intentionally out of scope

- Twilio
- Speech-to-text
- WebSockets / streaming
- Changes to the Whippy dashboard repository
