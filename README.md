# 🎙 The Empathy Engine — Giving AI a Human Voice

A service that dynamically modulates synthesized speech based on detected text emotion, bridging the gap between text sentiment and expressive, human-like audio output.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

### Core (Must-Haves)
- **Text Input** — Accept any text via web UI, API endpoint, or CLI
- **6-Emotion Detection** — Joy, Sadness, Anger, Fear, Surprise, Neutral (+ Disgust with transformer backend)
- **Vocal Parameter Modulation** — Dynamically adjusts Rate, Pitch, Volume, Emphasis, and Pause timing
- **Emotion-to-Voice Mapping** — Clear, intensity-scaled interpolation between neutral baseline and emotion profiles
- **Audio Output** — Generates playable `.mp3` / `.wav` files

### Bonus & Innovations
- **Granular Emotions** — 7-class emotion detection via DistilRoBERTa (j-hartmann model), far beyond simple pos/neg/neutral
- **Intensity Scaling** — Continuous modulation: "This is good" gets a mild pitch lift, while "This is THE BEST NEWS EVER!" gets dramatically different parameters. The system linearly interpolates between neutral and peak emotion profiles using the confidence score
- **SSML Generation** — Full Speech Synthesis Markup Language output with `<prosody>`, `<emphasis>`, and `<break>` tags for every utterance
- **Web Interface** — Beautiful, responsive FastAPI + Jinja2 UI with real-time emotion visualization, animated confidence bars, waveform audio player, and SSML inspector
- **Pluggable Architecture** — Swappable emotion backends (Transformer / VADER+TextBlob hybrid) and TTS engines (Edge-TTS / pyttsx3 / gTTS) via environment variables
- **Emotion-Aware Voice Selection** — When using Edge-TTS, the system selects different neural voices per emotion for added expressiveness (e.g., a warmer voice for joy, a deeper voice for anger)
- **Fallback Cascade** — Gracefully degrades: if the transformer model can't load, falls back to VADER; if Edge-TTS fails (no network), falls back to pyttsx3 (offline)

---

## 🏗 Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Text Input  │────▶│ Emotion Detector  │────▶│  Vocal Modulator   │
│  (UI / API)  │     │ (Transformer or   │     │  (Intensity-scaled │
│              │     │  VADER hybrid)    │     │   interpolation)   │
└─────────────┘     └──────────────────┘     └────────┬──────────┘
                                                       │
                          ┌────────────────────────────┘
                          ▼
                    ┌────────────┐     ┌──────────────┐
                    │ SSML Gen   │────▶│  TTS Engine   │──▶ 🔊 Audio File
                    │ (Prosody,  │     │ (Edge / pyttsx│
                    │  Emphasis) │     │   / gTTS)     │
                    └────────────┘     └──────────────┘
```

### Emotion → Voice Mapping Logic

| Emotion   | Rate (wpm) | Pitch | Volume | Emphasis | Pause (ms) |
|-----------|-----------|-------|--------|----------|------------|
| Joy       | 210 ↑     | 1.25× ↑ | 100% ↑ | strong   | 200 ↓ |
| Sadness   | 140 ↓     | 0.80× ↓ | 60% ↓  | reduced  | 500 ↑ |
| Anger     | 195 ↑     | 1.15× ↑ | 100% ↑ | strong   | 150 ↓ |
| Fear      | 220 ↑     | 1.30× ↑ | 70% ↓  | moderate | 250   |
| Surprise  | 205 ↑     | 1.35× ↑ | 95%    | strong   | 350 ↑ |
| Neutral   | 175       | 1.00×   | 85%    | moderate | 300   |

All values are **peak profiles at intensity=1.0**. The actual config is linearly interpolated: `actual = neutral + (peak - neutral) × intensity`.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/empathy-engine.git
cd empathy-engine
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Web Server

```bash
python app.py
# → Server starts at http://localhost:8000
```

Open your browser to [http://localhost:8000](http://localhost:8000) and start typing!

### 3. Run in CLI Mode

```bash
python app.py --cli
```

### 4. Use the API Directly

```bash
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so excited about this project!"}'
```

Response:
```json
{
  "emotion": {
    "primary_emotion": "joy",
    "confidence": 0.92,
    "intensity": 1.0,
    "scores": { "joy": 0.92, "surprise": 0.04, ... },
    "backend": "transformer"
  },
  "vocal_config": {
    "rate": 210,
    "pitch": 1.25,
    "volume": 1.0,
    "emphasis": "strong",
    "pause_ms": 200
  },
  "ssml": "<speak><prosody rate=\"120%\" pitch=\"+25%\" ...>...</prosody></speak>",
  "audio_url": "/static/audio/abc123.mp3"
}
```

---

## ⚙️ Configuration

| Env Variable      | Options                          | Default  |
|-------------------|----------------------------------|----------|
| `EMOTION_BACKEND` | `transformer`, `vader`, `auto`   | `auto`   |
| `TTS_ENGINE`      | `edge`, `pyttsx3`, `gtts`, `auto`| `auto`   |
| `PORT`            | Any integer                      | `8000`   |

```bash
# Example: Use VADER (fast, no model download) + offline TTS
EMOTION_BACKEND=vader TTS_ENGINE=pyttsx3 python app.py
```

---

## 📁 Project Structure

```
empathy-engine/
├── app.py                 # Main application (FastAPI + emotion + TTS)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web UI template
└── static/
    └── audio/             # Generated audio files (auto-created)
```

---

## 🧪 Design Decisions

1. **Intensity interpolation over discrete profiles**: Rather than mapping each emotion to a fixed set of parameters, we interpolate between a neutral baseline and peak profile using continuous intensity scores. This means "I'm a bit happy" and "I'M ECSTATIC!!!" produce meaningfully different voice outputs, not the same "happy voice."

2. **VADER hybrid as fallback**: The transformer model (DistilRoBERTa) needs ~250MB downloaded on first run. For environments where this isn't feasible, the VADER + keyword heuristic backend provides reasonable 6-class emotion classification without any model downloads.

3. **Edge-TTS as primary engine**: Microsoft's Edge TTS offers free, high-quality neural voices with prosody control — the best balance of quality and accessibility. The system cascades to pyttsx3 (fully offline, espeak-based) and gTTS (requires internet, limited parameter control) as fallbacks.

4. **SSML as intermediate representation**: Even when the TTS engine doesn't consume SSML directly, generating it serves as documentation of the modulation decisions and enables easy porting to SSML-native systems (Google Cloud TTS, Amazon Polly, Azure Speech).

---

## 📜 License

MIT
