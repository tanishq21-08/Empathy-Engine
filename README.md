# 🎙 The Empathy Engine — Giving AI a Human Voice

A service that dynamically modulates synthesized speech based on detected text emotion. Type any text, and the engine detects its emotional content, computes intensity-scaled vocal parameters (pitch, rate, volume, emphasis, pauses), generates SSML markup, and synthesizes expressive audio — with different voices and speaking styles per emotion.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Project Description & Capabilities

The Empathy Engine bridges the gap between text sentiment and expressive speech. It processes input text through a four-stage pipeline:

1. **Emotion Detection** — Classifies text into 7 emotions (joy, sadness, anger, fear, surprise, disgust, neutral) using either a DistilRoBERTa transformer model or a VADER + keyword hybrid backend.
2. **Vocal Parameter Computation** — Maps the detected emotion and its intensity to five vocal parameters (rate, pitch, volume, emphasis, pause duration) using continuous interpolation between a neutral baseline and peak emotion profiles.
3. **SSML Generation** — Produces Speech Synthesis Markup Language with `<prosody>`, `<emphasis>`, and `<break>` tags encoding the computed vocal parameters.
4. **Audio Synthesis** — Generates expressive speech audio using ElevenLabs (most realistic, with emotion-specific voice selection and style controls), Edge-TTS (free, with prosody modulation), or fallback engines.

### Key Features

- **7-Class Emotion Detection** — Joy, sadness, anger, fear, surprise, disgust, neutral — far beyond simple positive/negative/neutral.
- **Intensity-Scaled Modulation** — "I'm okay" and "I'M ECSTATIC!!!" produce genuinely different vocal outputs. The system linearly interpolates between neutral and peak emotion profiles using the confidence score.
- **5 Vocal Parameters** — Rate (words per minute), pitch (multiplier), volume (0–1), emphasis level (SSML), and inter-sentence pause duration (ms).
- **ElevenLabs Integration** — Emotion-specific voice actors (warm female for joy, calm male for neutral, assertive male for anger, soft female for fear), with per-emotion stability, style exaggeration, and speed controls.
- **Emotion-Aware Voice Selection** — Edge-TTS selects different Microsoft Neural voices per emotion (JennyNeural for joy, AriaNeural for sadness, GuyNeural for anger).
- **SSML Output** — Full Speech Synthesis Markup Language generation for portability to any SSML-compatible TTS system (Google Cloud TTS, Amazon Polly, Azure Speech).
- **Fallback Cascade** — ElevenLabs → Edge-TTS → pyttsx3 → gTTS. Always produces audio regardless of available engines.
- **Polished Web UI** — Dark-themed interface with emotion visualization bars, vocal parameter display, waveform audio player, and SSML inspector.

---

## 🚀 Setup & Execution

### Prerequisites

- Python 3.10+
- pip

### Step 1: Clone & Install

```bash
git clone https://github.com/tanishq21-08/Empathy-Engine.git
cd Empathy-Engine
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2: API Key Management

Create a `.env` file in the project root:

```bash
# Linux/Mac:
cp .env.example .env

# Windows:
copy .env.example .env
```

Open `.env` and configure:

```env
# Emotion detection backend: "transformer" | "vader" | "auto"
EMOTION_BACKEND=auto

# TTS engine: "elevenlabs" | "edge" | "pyttsx3" | "gtts" | "auto"
# auto = tries ElevenLabs (if key set) → Edge-TTS → pyttsx3 → gTTS
TTS_ENGINE=auto

# ElevenLabs API key (free at https://elevenlabs.io — 10,000 chars/month)
# Sign up → Profile → API Key → Copy
ELEVENLABS_API_KEY=

# Server port
PORT=8000
```

**Without any API key:** The app works fully out of the box using Edge-TTS (free Microsoft Neural voices with pitch/rate/volume controls) and VADER emotion detection. No sign-up required.

**With ElevenLabs (recommended for best quality):** Sign up free at [elevenlabs.io](https://elevenlabs.io) → Profile icon → "Profile + API key" → copy key. Free tier gives 10,000 characters/month. The difference in expressiveness is dramatic — ElevenLabs voices have genuine emotional range with stability, style, and speed controls per emotion.

**Security:** The `.env` file contains secrets and is excluded from Git via `.gitignore`. Never commit it. The `.env.example` file (without real keys) is safe to commit.

### Step 3: Run the Application

```bash
python app.py
# → Server starts at http://localhost:8000
```

Open your browser to [http://localhost:8000](http://localhost:8000), type or paste text, and click "Synthesize Voice."

### Alternative: CLI Mode

```bash
python app.py --cli
```

### Alternative: Direct API Usage

```bash
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so excited about this project!"}'
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/synthesize` | POST | Detect emotion, modulate voice, return audio |
| `/api/health` | GET | Backend status (emotion backend, TTS engine) |

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
                    │ SSML Gen   │────▶│  TTS Engine   │──▶ 🔊 Audio
                    │ (Prosody,  │     │ (ElevenLabs / │
                    │  Emphasis) │     │  Edge / gTTS) │
                    └────────────┘     └──────────────┘
```

---

## 🎨 Design Choices: Emotion-to-Voice Mapping Logic

### The Core Idea: Intensity-Scaled Interpolation

The central design choice is **continuous interpolation** rather than discrete emotion profiles. Every vocal parameter is computed as:

```
actual_value = neutral_baseline + (peak_profile - neutral_baseline) × intensity
```

Where `intensity` is derived from the emotion detector's confidence score (0–1). This means "I'm a bit happy" produces a mild pitch lift and slight rate increase, while "I'M ECSTATIC!!!" produces dramatically faster, higher-pitched, louder speech. The same emotion at different intensities sounds genuinely different.

### Emotion → Vocal Parameter Mapping

The neutral baseline is: **175 wpm, 1.0× pitch, 0.9 volume, moderate emphasis, 300ms pause.**

Each emotion has a peak profile that represents maximum emotional expression:

| Emotion | Rate (wpm) | Pitch | Volume | Emphasis | Pause (ms) | Rationale |
|---------|-----------|-------|--------|----------|------------|-----------|
| **Joy** | 230 ↑ | 1.40× ↑ | 100% ↑ | strong | 150 ↓ | Excitement speeds up speech, raises pitch, reduces pauses |
| **Sadness** | 120 ↓ | 0.65× ↓ | 50% ↓ | reduced | 700 ↑ | Heaviness slows everything down, drops pitch and volume |
| **Anger** | 210 ↑ | 1.20× ↑ | 100% ↑ | strong | 100 ↓ | Intensity increases rate and volume, short forceful pauses |
| **Fear** | 240 ↑ | 1.45× ↑ | 65% ↓ | moderate | 200 | Anxiety speeds up speech, raises pitch, but quieter (trembling) |
| **Surprise** | 225 ↑ | 1.50× ↑ | 100% | strong | 400 ↑ | Shock raises pitch dramatically, longer pauses (processing) |
| **Disgust** | 145 ↓ | 0.80× ↓ | 70% ↓ | moderate | 500 ↑ | Controlled displeasure, slower, lower, deliberate pauses |
| **Neutral** | 175 | 1.00× | 85% | moderate | 300 | Calm, balanced baseline |

### ElevenLabs Voice Settings per Emotion

When ElevenLabs is available, the system maps emotions to specific voice settings that leverage ElevenLabs' expressiveness controls:

| Emotion | Voice | Stability | Style | Speed | Effect |
|---------|-------|-----------|-------|-------|--------|
| Joy | Gigi (warm female) | 0.30 (expressive) | 0.85 | Faster | Dynamic, warm, excited delivery |
| Sadness | Daniel (calm male) | 0.75 (steady) | 0.70 | Slower | Subdued, heavy, measured delivery |
| Anger | Liam (assertive male) | 0.25 (intense) | 0.90 | Fast | Forceful, variable, powerful delivery |
| Fear | Sarah (soft female) | 0.35 (unstable) | 0.75 | Fast | Trembling, uncertain, breathy delivery |
| Surprise | Gigi (warm female) | 0.20 (very dynamic) | 0.95 | Fast | Maximally reactive, energetic delivery |
| Neutral | Daniel (calm male) | 0.70 (stable) | 0.20 | Normal | Clear, balanced, professional delivery |

**Stability** controls how variable the voice is — lower = more expressive. Anger and surprise get the lowest stability for maximum emotional impact, while sadness and neutral get high stability for steady delivery.

**Style** controls exaggeration — higher = more dramatic. Surprise at 0.95 is nearly maximally expressive, while neutral at 0.20 is restrained and professional.

### Edge-TTS Voice Selection

When using Edge-TTS (free, no API key), different Microsoft Neural voices are selected per emotion:
- **Joy/Surprise** → en-US-JennyNeural (warm, bright)
- **Sadness/Fear** → en-US-AriaNeural (soft, gentle)
- **Anger/Disgust** → en-US-GuyNeural (deep, strong)
- **Neutral** → en-US-JennyNeural (balanced)

### Emotion Detection: Dual Backend

- **Transformer backend** (default if available): Uses `j-hartmann/emotion-english-distilroberta-base`, a 7-class emotion model fine-tuned on multiple emotion datasets. Downloads ~250MB on first run, then cached. Most accurate.
- **VADER hybrid backend** (fallback): Combines NLTK's VADER sentiment analyzer with keyword-based emotion boosters for 6-class emotion classification. Zero downloads, instant startup, always available.

The app preloads the chosen backend at server startup so the first request is fast.

### SSML as Intermediate Representation

Even when the TTS engine doesn't consume SSML directly, the system generates it as documentation of modulation decisions. The SSML includes `<prosody>` (rate, pitch, volume), `<emphasis>` (per-sentence), and `<break>` (inter-sentence pauses). This enables easy porting to any SSML-native TTS system (Google Cloud TTS, Amazon Polly, Azure Speech).

---

## 📁 Project Structure

```
empathy-engine/
├── app.py                 # Main application (FastAPI + emotion detection + TTS)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .env.example           # Environment variable template (safe to commit)
├── .env                   # Your actual API keys (git-ignored, never committed)
├── .gitignore             # Git ignore rules
├── templates/
│   └── index.html         # Web UI (Jinja2 template)
└── static/
    └── audio/             # Generated audio files (auto-created at runtime)
```

---

## ⚙️ Configuration Reference

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `EMOTION_BACKEND` | `transformer`, `vader`, `auto` | `auto` | Emotion detection model |
| `TTS_ENGINE` | `elevenlabs`, `edge`, `pyttsx3`, `gtts`, `auto` | `auto` | Speech synthesis engine |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | — | Free at elevenlabs.io (10K chars/month) |
| `PORT` | Any integer | `8000` | Server port |

---

## 📜 License

MIT
