"""Reusable voice helpers. Copy this file into InterviewMentor AI unchanged."""
import asyncio
import io
import os
import re
import wave

import edge_tts
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

STT_MODEL = "whisper-large-v3-turbo"   # check Groq docs for current model names
TTS_VOICE = "en-IN-NeerjaNeural"       # try en-US-JennyNeural, en-GB-RyanNeural, etc.

FILLER_PATTERN = re.compile(
    r"\b(um+|uh+|er+|erm|like|basically|actually|literally|you know|kind of|sort of)\b"
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


# ---------- Speech to text ----------
def transcribe(audio_bytes: bytes, language: str = "en") -> str:
    """Send recorded audio to Groq Whisper and return the transcript."""
    result = _get_client().audio.transcriptions.create(
        file=("answer.wav", audio_bytes),
        model=STT_MODEL,
        language=language,
    )
    return result.text.strip()


# ---------- Text to speech ----------
async def _tts_async(text: str, voice: str) -> bytes:
    chunks = []
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def text_to_speech(text: str, voice: str = TTS_VOICE) -> bytes:
    """Return MP3 bytes for the given text."""
    return asyncio.run(_tts_async(text, voice))


# ---------- Audio + fluency metrics ----------
def audio_seconds(audio_bytes: bytes) -> float:
    """Duration of a WAV recording in seconds."""
    with wave.open(io.BytesIO(audio_bytes)) as w:
        return w.getnframes() / float(w.getframerate())


def analyze_speech(transcript: str, seconds: float) -> dict:
    """Simple fluency metrics computed from transcript + duration."""
    words = transcript.split()
    word_count = len(words)
    wpm = round(word_count / (seconds / 60)) if seconds > 0 else 0
    fillers = FILLER_PATTERN.findall(transcript.lower())
    return {
        "duration_sec": round(seconds, 1),
        "word_count": word_count,
        "wpm": wpm,
        "filler_count": len(fillers),
        "filler_words": sorted(set(fillers)),
        "filler_rate_pct": round(100 * len(fillers) / word_count, 1) if word_count else 0.0,
    }