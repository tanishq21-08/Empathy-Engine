"""
The Empathy Engine — Giving AI a Human Voice 🎙
================================================
A service that dynamically modulates synthesized speech based on detected
text emotion. Features granular 6-emotion detection, intensity-scaled vocal
modulation, SSML integration, and a polished web interface.

Author: Tanishq
"""

import os
import re
import json
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Emotion Detection — pluggable backend
# ---------------------------------------------------------------------------
# We support two backends:
#   1. "transformer" — Hugging Face pipeline (accurate, needs model download)
#   2. "vader"       — VADER + TextBlob hybrid (fast, no GPU, always available)
# Set EMOTION_BACKEND env var to choose. Default: tries transformer, falls back.
# ---------------------------------------------------------------------------

EMOTION_BACKEND = os.getenv("EMOTION_BACKEND", "auto")

# Lazy-loaded singletons
_hf_classifier = None
_vader_analyzer = None


@dataclass
class EmotionResult:
    """Structured emotion analysis output."""
    primary_emotion: str          # e.g. "joy", "anger", "sadness", ...
    confidence: float             # 0-1 confidence in primary label
    intensity: float              # 0-1 intensity scalar for modulation
    scores: dict = field(default_factory=dict)  # all emotion scores
    backend: str = "unknown"


# ---- Transformer backend ---------------------------------------------------

def _get_hf_classifier():
    global _hf_classifier
    if _hf_classifier is None:
        from transformers import pipeline
        # ek-emotion is a 6-class emotion model: joy, sadness, anger, fear, surprise, disgust
        # Fallback: j-hartmann/emotion-english-distilroberta-base (7-class with neutral)
        try:
            _hf_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=-1,  # CPU
            )
        except Exception:
            _hf_classifier = pipeline(
                "text-classification",
                model="bhadresh-savani/distilbert-base-uncased-emotion",
                top_k=None,
                device=-1,
            )
    return _hf_classifier


def detect_emotion_transformer(text: str) -> EmotionResult:
    clf = _get_hf_classifier()
    results = clf(text[:512])[0]  # list of {label, score}
    scores = {r["label"].lower(): round(r["score"], 4) for r in results}
    top = max(results, key=lambda r: r["score"])
    primary = top["label"].lower()
    confidence = round(top["score"], 4)
    # Intensity: use the raw confidence as a proxy, scaled so >0.6 → strong
    intensity = min(1.0, round(confidence * 1.3, 4))
    return EmotionResult(
        primary_emotion=primary,
        confidence=confidence,
        intensity=intensity,
        scores=scores,
        backend="transformer",
    )


# ---- VADER + TextBlob hybrid backend --------------------------------------

def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer
    return _vader_analyzer()


def detect_emotion_vader(text: str) -> EmotionResult:
    analyzer = _get_vader()
    vs = analyzer.polarity_scores(text)
    compound = vs["compound"]  # -1 to 1

    # Map compound + keyword heuristics to 6 emotions
    text_lower = text.lower()

    # Keyword boosters
    anger_words = {"angry", "furious", "hate", "annoyed", "irritated", "outraged", "mad", "frustrated"}
    fear_words = {"afraid", "scared", "anxious", "worried", "terrified", "nervous", "panic"}
    surprise_words = {"wow", "amazing", "shocked", "unexpected", "unbelievable", "incredible", "surprise"}
    sadness_words = {"sad", "depressed", "heartbroken", "miserable", "disappointed", "grief", "crying"}
    joy_words = {"happy", "delighted", "thrilled", "excited", "wonderful", "fantastic", "love", "great"}

    tokens = set(re.findall(r"\w+", text_lower))

    anger_score = len(tokens & anger_words) * 0.25 + max(0, -compound) * 0.3
    fear_score = len(tokens & fear_words) * 0.25 + max(0, -compound) * 0.2
    sadness_score = len(tokens & sadness_words) * 0.25 + max(0, -compound) * 0.25
    surprise_score = len(tokens & surprise_words) * 0.3
    joy_score = len(tokens & joy_words) * 0.25 + max(0, compound) * 0.4
    neutral_score = 0.3 if abs(compound) < 0.2 else 0.05

    raw = {
        "joy": joy_score,
        "sadness": sadness_score,
        "anger": anger_score,
        "fear": fear_score,
        "surprise": surprise_score,
        "neutral": neutral_score,
    }
    total = sum(raw.values()) or 1.0
    scores = {k: round(v / total, 4) for k, v in raw.items()}
    primary = max(scores, key=scores.get)
    confidence = scores[primary]
    intensity = min(1.0, round(abs(compound) * 1.2 + 0.1, 4))

    return EmotionResult(
        primary_emotion=primary,
        confidence=confidence,
        intensity=intensity,
        scores=scores,
        backend="vader",
    )


# ---- Unified detect function -----------------------------------------------

def detect_emotion(text: str) -> EmotionResult:
    if EMOTION_BACKEND == "transformer":
        return detect_emotion_transformer(text)
    elif EMOTION_BACKEND == "vader":
        return detect_emotion_vader(text)
    else:  # auto
        try:
            return detect_emotion_transformer(text)
        except Exception:
            return detect_emotion_vader(text)


# ---------------------------------------------------------------------------
# Vocal Parameter Modulation
# ---------------------------------------------------------------------------
# Maps each emotion to a set of vocal parameters.  Intensity scaling adjusts
# the deviation from the neutral baseline.
# ---------------------------------------------------------------------------

@dataclass
class VocalConfig:
    """TTS vocal parameters."""
    rate: int = 175         # words per minute (pyttsx3 default ~200)
    pitch: float = 1.0      # multiplier (1.0 = default)
    volume: float = 0.9     # 0.0 – 1.0
    emphasis: str = "moderate"  # SSML emphasis level
    pause_ms: int = 300     # inter-sentence pause


# Baseline configs per emotion (at max intensity)
EMOTION_PROFILES = {
    "joy": VocalConfig(rate=230, pitch=1.40, volume=1.0, emphasis="strong", pause_ms=150),
    "sadness": VocalConfig(rate=120, pitch=0.65, volume=0.5, emphasis="reduced", pause_ms=700),
    "anger": VocalConfig(rate=210, pitch=1.20, volume=1.0, emphasis="strong", pause_ms=100),
    "fear": VocalConfig(rate=240, pitch=1.45, volume=0.65, emphasis="moderate", pause_ms=200),
    "surprise": VocalConfig(rate=225, pitch=1.50, volume=1.0, emphasis="strong", pause_ms=400),
    "disgust": VocalConfig(rate=145, pitch=0.80, volume=0.7, emphasis="moderate", pause_ms=500),
    "neutral": VocalConfig(rate=175, pitch=1.0, volume=0.85, emphasis="moderate", pause_ms=300),
}

NEUTRAL = VocalConfig()


def compute_vocal_config(emotion: EmotionResult) -> VocalConfig:
    """Interpolate between neutral baseline and emotion profile based on intensity."""
    profile = EMOTION_PROFILES.get(emotion.primary_emotion, NEUTRAL)
    t = emotion.intensity  # interpolation factor
    return VocalConfig(
        rate=int(NEUTRAL.rate + (profile.rate - NEUTRAL.rate) * t),
        pitch=round(NEUTRAL.pitch + (profile.pitch - NEUTRAL.pitch) * t, 3),
        volume=round(NEUTRAL.volume + (profile.volume - NEUTRAL.volume) * t, 3),
        emphasis=profile.emphasis if t > 0.5 else NEUTRAL.emphasis,
        pause_ms=int(NEUTRAL.pause_ms + (profile.pause_ms - NEUTRAL.pause_ms) * t),
    )


# ---------------------------------------------------------------------------
# SSML Generation
# ---------------------------------------------------------------------------

def text_to_ssml(text: str, config: VocalConfig, emotion: str) -> str:
    """Wrap text in SSML with prosody and emphasis tags."""
    # Pitch as percentage offset from default
    pitch_pct = int((config.pitch - 1.0) * 100)
    pitch_str = f"{pitch_pct:+d}%" if pitch_pct != 0 else "+0%"

    # Rate as percentage
    rate_pct = int((config.rate / 175) * 100)
    rate_str = f"{rate_pct}%"

    # Volume
    vol_map = {
        (0.0, 0.4): "x-soft",
        (0.4, 0.6): "soft",
        (0.6, 0.8): "medium",
        (0.8, 0.95): "loud",
        (0.95, 1.01): "x-loud",
    }
    vol_str = "medium"
    for (lo, hi), label in vol_map.items():
        if lo <= config.volume < hi:
            vol_str = label
            break

    # Split sentences and add pauses
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    body_parts = []
    for sent in sentences:
        body_parts.append(
            f'<emphasis level="{config.emphasis}">{sent}</emphasis>'
        )
        body_parts.append(f'<break time="{config.pause_ms}ms"/>')

    body = "\n    ".join(body_parts)

    ssml = f"""<speak>
  <prosody rate="{rate_str}" pitch="{pitch_str}" volume="{vol_str}">
    {body}
  </prosody>
</speak>"""
    return ssml


# ---------------------------------------------------------------------------
# TTS Synthesis — pluggable engines
# ---------------------------------------------------------------------------
# Priority: "elevenlabs" → "edge" → "pyttsx3" → "gtts"
# Set TTS_ENGINE env var to force one. Default: auto (tries in order).
# For ElevenLabs, set ELEVENLABS_API_KEY in .env
# ---------------------------------------------------------------------------

TTS_ENGINE = os.getenv("TTS_ENGINE", "auto")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
AUDIO_DIR = Path("static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ---- ElevenLabs (most realistic) ------------------------------------------

async def synthesize_elevenlabs(text: str, config: VocalConfig, emotion: str) -> Path:
    """
    ElevenLabs TTS with emotion-mapped voice settings.
    
    Key parameters mapped from our emotion profiles:
    - stability: Lower = more expressive/variable. Sadness/neutral → high stability,
                 Joy/anger/surprise → low stability for more dynamic delivery.
    - similarity_boost: How closely to match the base voice. We keep this moderate.
    - style: 0-1, higher = more expressive. Maps directly to our intensity.
    - speed: Maps from our rate (wpm). ElevenLabs range ~0.5-2.0.
    """
    import httpx

    # Voice selection — ElevenLabs has many voices; these are good defaults
    # You can replace these with your own voice IDs from your ElevenLabs dashboard
    voice_map = {
        "joy":      "jBpfuIE2acCO8z3wKNLl",  # Gigi — warm, bright female
        "sadness":  "onwK4e9ZLuTAKqWW03F9",  # Daniel — calm, gentle male
        "anger":    "TX3LPaxmHKxFdv7VOQHJ",  # Liam — strong, assertive male
        "fear":     "EXAVITQu4vr4xnSDxMaL",  # Sarah — soft, breathy female
        "surprise": "jBpfuIE2acCO8z3wKNLl",  # Gigi — energetic
        "disgust":  "TX3LPaxmHKxFdv7VOQHJ",  # Liam — deep, disapproving
        "neutral":  "onwK4e9ZLuTAKqWW03F9",  # Daniel — balanced, clear
    }
    voice_id = voice_map.get(emotion, "onwK4e9ZLuTAKqWW03F9")

    # Map emotion to ElevenLabs voice_settings
    # stability: 0 = very expressive/variable, 1 = very stable/monotone
    stability_map = {
        "joy": 0.30,       # Expressive, dynamic
        "sadness": 0.75,   # Slow, steady, subdued
        "anger": 0.25,     # Intense, variable
        "fear": 0.35,      # Trembling, unstable
        "surprise": 0.20,  # Very dynamic
        "disgust": 0.60,   # Controlled displeasure
        "neutral": 0.70,   # Calm, stable
    }

    # style: 0 = neutral delivery, 1 = maximally expressive
    style_map = {
        "joy": 0.85,
        "sadness": 0.70,
        "anger": 0.90,
        "fear": 0.75,
        "surprise": 0.95,
        "disgust": 0.60,
        "neutral": 0.20,
    }

    # speed: map our wpm (120-240) to ElevenLabs range (0.7-1.4)
    speed = round(0.7 + (config.rate - 120) / (240 - 120) * 0.7, 2)
    speed = max(0.5, min(2.0, speed))

    stability = stability_map.get(emotion, 0.5)
    style = style_map.get(emotion, 0.3)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": 0.75,
                    "style": style,
                    "use_speaker_boost": True,
                    "speed": speed,
                },
            },
            timeout=30,
        )
        response.raise_for_status()

        filename = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
        filename.write_bytes(response.content)
        return filename


# ---- pyttsx3 (offline) ----------------------------------------------------

async def synthesize_pyttsx3(text: str, config: VocalConfig) -> Path:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", config.rate)
    engine.setProperty("volume", config.volume)
    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[0].id)
    try:
        pitch_val = int(50 * config.pitch)
        pitch_val = max(0, min(99, pitch_val))
        engine._driver._engine.setParameter(
            engine._driver._engine.Parameter.Pitch, pitch_val, 0
        )
    except Exception:
        pass

    filename = AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
    engine.save_to_file(text, str(filename))
    engine.runAndWait()
    return filename


# ---- gTTS (Google, online, flat) -------------------------------------------

async def synthesize_gtts(text: str, config: VocalConfig) -> Path:
    from gtts import gTTS
    slow = config.rate < 160
    tts = gTTS(text=text, lang="en", slow=slow)
    filename = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
    tts.save(str(filename))
    return filename


# ---- Edge TTS (free, decent prosody) --------------------------------------

async def synthesize_edge_tts(text: str, config: VocalConfig, emotion: str) -> Path:
    """Use Microsoft Edge TTS with native prosody controls."""
    import edge_tts

    voice_map = {
        "joy": "en-US-JennyNeural",
        "sadness": "en-US-AriaNeural",
        "anger": "en-US-GuyNeural",
        "fear": "en-US-AriaNeural",
        "surprise": "en-US-JennyNeural",
        "disgust": "en-US-GuyNeural",
        "neutral": "en-US-JennyNeural",
    }
    voice = voice_map.get(emotion, "en-US-JennyNeural")

    rate_offset = int(((config.rate / 175) - 1.0) * 100)
    rate_str = f"{rate_offset:+d}%"
    pitch_offset = int((config.pitch - 1.0) * 100)
    pitch_str = f"{pitch_offset:+d}Hz"
    vol_offset = int((config.volume - 0.9) * 100)
    vol_str = f"{vol_offset:+d}%"

    communicate = edge_tts.Communicate(
        text, voice=voice, rate=rate_str, pitch=pitch_str, volume=vol_str,
    )
    filename = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
    await communicate.save(str(filename))
    return filename


# ---- Dispatcher ------------------------------------------------------------

async def synthesize(text: str, config: VocalConfig, emotion: str) -> tuple[Path, str]:
    """Try engines in priority order. Returns (filepath, engine_name)."""
    engines = []
    if TTS_ENGINE == "auto":
        # ElevenLabs first if API key is set, then cascade
        if ELEVENLABS_API_KEY:
            engines.append("elevenlabs")
        engines.extend(["edge", "pyttsx3", "gtts"])
    else:
        engines = [TTS_ENGINE]

    last_err = None
    for eng in engines:
        try:
            if eng == "elevenlabs":
                path = await synthesize_elevenlabs(text, config, emotion)
                return path, "ElevenLabs (neural, expressive)"
            elif eng == "edge":
                path = await synthesize_edge_tts(text, config, emotion)
                return path, "Edge-TTS (prosody controls)"
            elif eng == "pyttsx3":
                path = await synthesize_pyttsx3(text, config)
                return path, "pyttsx3 (offline)"
            elif eng == "gtts":
                path = await synthesize_gtts(text, config)
                return path, "gTTS (basic)"
        except Exception as e:
            last_err = e
            logging.warning(f"TTS engine '{eng}' failed: {e}")
            continue

    raise RuntimeError(f"All TTS engines failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="The Empathy Engine", version="1.0.0")

# Serve static files (audio, CSS, JS)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/synthesize")
async def api_synthesize(request: Request):
    """
    Main API endpoint.
    
    Request JSON:
        { "text": "Hello, I'm so excited to share this news!" }
    
    Response JSON:
        {
            "emotion": { ... EmotionResult ... },
            "vocal_config": { ... VocalConfig ... },
            "ssml": "<speak>...</speak>",
            "audio_url": "/static/audio/abc123.mp3"
        }
    """
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text must be under 5000 characters.")

    # Step 1: Detect emotion
    emotion = detect_emotion(text)

    # Step 2: Compute vocal parameters (intensity-scaled)
    vocal = compute_vocal_config(emotion)

    # Step 3: Generate SSML
    ssml = text_to_ssml(text, vocal, emotion.primary_emotion)

    # Step 4: Synthesize audio
    try:
        audio_path, engine_used = await synthesize(text, vocal, emotion.primary_emotion)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "emotion": asdict(emotion),
        "vocal_config": asdict(vocal),
        "ssml": ssml,
        "audio_url": f"/static/audio/{audio_path.name}",
        "tts_engine": engine_used,
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "backend": EMOTION_BACKEND, "tts_engine": TTS_ENGINE}


# ---------------------------------------------------------------------------
# CLI Mode
# ---------------------------------------------------------------------------

def cli_mode():
    """Run in command-line mode for quick testing."""
    import asyncio

    print("=" * 60)
    print("  🎙  The Empathy Engine — CLI Mode")
    print("=" * 60)

    while True:
        text = input("\nEnter text (or 'quit' to exit): ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break

        emotion = detect_emotion(text)
        vocal = compute_vocal_config(emotion)
        ssml = text_to_ssml(text, vocal, emotion.primary_emotion)

        print(f"\n  Emotion  : {emotion.primary_emotion} "
              f"(confidence: {emotion.confidence:.2f}, intensity: {emotion.intensity:.2f})")
        print(f"  Backend  : {emotion.backend}")
        print(f"  Scores   : {json.dumps(emotion.scores, indent=2)}")
        print(f"  Vocal    : rate={vocal.rate} wpm, pitch={vocal.pitch}x, "
              f"volume={vocal.volume}, emphasis={vocal.emphasis}")
        print(f"\n  SSML:\n{ssml}")

        try:
            audio_path = asyncio.run(synthesize(text, vocal, emotion.primary_emotion))
            print(f"\n  ✅ Audio saved: {audio_path}")
        except Exception as e:
            print(f"\n  ⚠️  TTS failed: {e}")
            print("     (Install edge-tts, pyttsx3, or gtts for audio output)")


if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        cli_mode()
    else:
        import uvicorn
        port = int(os.getenv("PORT", 8000))
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)