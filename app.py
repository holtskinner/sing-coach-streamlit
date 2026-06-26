import io
import logging
import urllib.parse
import wave

import matplotlib

matplotlib.use("Agg")  # headless backend; no GUI needed for rendering to PNG
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.chats import Chat
from google.genai.types import (
    GenerateContentConfig,
    Part,
    PartMediaResolution,
    PartMediaResolutionLevel,
    PrebuiltVoiceConfig,
    SpeechConfig,
    ThinkingConfig,
    ThinkingLevel,
    VoiceConfig,
)

# Display name -> model id.
MODELS = {
    "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite",
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemini 3.1 Pro": "gemini-3.1-pro-preview",
}
# Gemini-TTS prebuilt voices, see
# https://docs.cloud.google.com/text-to-speech/docs/gemini-tts#voice_options
VOICES = [
    "Achernar",
    "Achird",
    "Algenib",
    "Algieba",
    "Alnilam",
    "Aoede",
    "Autonoe",
    "Callirrhoe",
    "Charon",
    "Despina",
    "Enceladus",
    "Erinome",
    "Fenrir",
    "Gacrux",
    "Iapetus",
    "Kore",
    "Laomedeia",
    "Leda",
    "Orus",
    "Pulcherrima",
    "Puck",
    "Rasalgethi",
    "Sadachbia",
    "Sadaltager",
    "Schedar",
    "Sulafat",
    "Umbriel",
    "Vindemiatrix",
    "Zephyr",
    "Zubenelgenubi",
]
VOICE_MODEL_ID = "gemini-3.1-flash-tts-preview"
LANGUAGE_CODE = "en-us"
MAX_TURNS = 20  # UI history cap; trims oldest audio to bound session_state growth

SYSTEM_INSTRUCTION = """\
You are AriaCoach, an expert vocal coach giving spoken feedback on a short \
singing clip the user just recorded. Your goal is to help them sing better, \
fast.

You may also receive a mel spectrogram image of the same clip. Use it to \
corroborate what you hear - vertical smearing or shifting bands reveal pitch \
drift and vibrato, bright high-frequency energy hints at strain or breathiness, \
and gaps show phrasing and timing. Let the audio lead; treat the image as \
supporting evidence and never mention the spectrogram itself to the user.

Listen to the audio and assess these dimensions, in priority order:
1. Pitch accuracy and intonation - are notes on key? Note any consistent \
flat/sharp tendency.
2. Rhythm and timing - do they stay on the beat? Is phrasing rushed or dragging?
3. Tone and vocal quality - resonance, vibrato, vowel shaping, audible strain.

How to respond:
- Lead with one genuine, specific positive, then give the most important fix.
- Be warm and encouraging. You may add a light, witty Simon Cowell-style jab \
ONLY when a performance is genuinely poor - never mean-spirited or personal.
- Be concrete: reference what you actually heard ("the chorus went flat on the \
high notes"), not generic advice.
- ALWAYS end with one specific practice exercise they can try right now.
- If the audio is silent, too noisy, or not singing, say so kindly and ask them \
to re-record.

Output is read aloud by text-to-speech, so:
- Keep it to roughly 3-5 short sentences. No lists, headings, markdown, emojis, \
or stage directions - just natural spoken prose.
- Write numbers and symbols as words a narrator would say.
- Use the prior turns to track progress and avoid repeating the same notes."""


@st.cache_resource
def load_client() -> genai.Client:
    """Load the Google Gen AI client."""
    load_dotenv(override=True)
    return genai.Client()


def get_chat(client: genai.Client, model: str) -> Chat:
    """Per-session chat so history persists across reruns but not across users.

    Recreates the chat when the model changes (history is dropped, since the
    new model has no record of the old turns anyway).
    """
    if "chat" not in st.session_state or st.session_state.get("model") != model:
        st.session_state.model = model

        thinking_level = ThinkingLevel.LOW if "pro" in model else ThinkingLevel.MINIMAL
        st.session_state.chat = client.chats.create(
            model=model,
            config=GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                thinking_config=ThinkingConfig(thinking_level=thinking_level),
            ),
        )
    return st.session_state.chat


client = load_client()


def generate_audio(text: str, voice: str) -> bytes:
    """Generates WAV audio from text using Gemini TTS (returns 24kHz PCM as WAV)."""
    response = client.models.generate_content(
        model=VOICE_MODEL_ID,
        contents=text,
        config=GenerateContentConfig(
            speech_config=SpeechConfig(
                language_code=LANGUAGE_CODE,
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice)
                ),
            ),
        ),
    )
    parts = response.parts
    if not parts or not parts[0].inline_data or not parts[0].inline_data.data:
        raise RuntimeError("Gemini TTS returned no audio")
    pcm = parts[0].inline_data.data

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm)
    return buffer.getvalue()


def make_spectrogram(wav_bytes: bytes) -> bytes:
    """Render a spectrogram PNG from WAV bytes (st.audio_input gives 16-bit mono WAV)."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    fig, ax = plt.subplots(figsize=(8, 3))
    with np.errstate(divide="ignore"):
        ax.specgram(samples, Fs=sr)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def clean_youtube_url(url: str) -> str:
    if not url.strip().startswith(("http://", "https://")):
        url = "https://" + url.strip()
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    clean_qs = urllib.parse.urlencode({k: qs[k][0] for k in ("v", "t") if k in qs})
    return parsed._replace(query=clean_qs).geturl()


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title("AriaCoach - Singing Teacher")

    with st.sidebar:
        model = MODELS[st.selectbox("Model", MODELS)]
        voice = st.selectbox("Voice", VOICES, index=VOICES.index("Achird"))

    chat = get_chat(client, model)

    if st.button("Start over"):
        # Drop chat + UI history; next get_chat() makes a fresh session.
        st.session_state.pop("chat", None)
        st.session_state.pop("turns", None)
        st.rerun()

    # Replay prior turns; chat.send_message keeps model-side history, this keeps the UI in sync.
    for turn in st.session_state.get("turns", []):
        with st.chat_message("user"):
            if turn.get("input_mime") == "youtube":
                st.video(turn["input"])
            else:
                st.audio(turn["input"], format=turn.get("input_mime", "audio/wav"))
                if turn.get("spectrogram"):
                    st.image(turn["spectrogram"], caption="Spectrogram")
        with st.chat_message("assistant"):
            st.markdown(turn["text"])
            st.audio(turn["output"], format="audio/wav")

    audio_input = st.audio_input("Record your singing")
    uploaded = st.file_uploader(
        "...or upload an audio/video file",
        type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "mp4", "mov", "webm"],
    )
    youtube_url = st.text_input("...or paste a YouTube URL")
    if youtube_url:
        youtube_url = clean_youtube_url(youtube_url)

    source = audio_input or uploaded

    if source or youtube_url:
        spectrogram = None
        if source:
            data = source.getvalue()
            mime_type = source.type or "audio/wav"
            message = [
                Part.from_bytes(
                    data=data,
                    mime_type=mime_type,
                    media_resolution=PartMediaResolution(
                        level=PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
                    ),
                )
            ]
            try:
                spectrogram = make_spectrogram(data)
                message.append(
                    Part.from_bytes(
                        data=spectrogram,
                        mime_type="image/png",
                        media_resolution=PartMediaResolution(
                            level=PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
                        ),
                    )
                )
            except Exception as e:
                logging.warning("Spectrogram generation failed: %s", e)
            with st.chat_message("user"):
                st.audio(data, format=mime_type)
                if spectrogram:
                    st.image(spectrogram, caption="Spectrogram")
        else:
            data = youtube_url
            mime_type = "youtube"
            message = Part.from_uri(
                file_uri=youtube_url,
                mime_type="video/*",
                media_resolution=PartMediaResolution(
                    level=PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
                ),
            )
            with st.chat_message("user"):
                st.video(youtube_url)

        try:
            with st.spinner("Analyzing your singing..."):
                response = chat.send_message(message=message)
                text = response.text or "Sorry, I couldn't analyze that."

            with st.spinner("Generating voice feedback..."):
                output_audio_bytes = generate_audio(text, voice)
        except Exception as e:
            logging.exception(e)
            st.error("Something went wrong talking to Gemini. Please try again.")
            st.stop()

        with st.chat_message("assistant"):
            st.markdown(text)
            st.audio(output_audio_bytes, format="audio/wav", autoplay=True)

        turns = st.session_state.setdefault("turns", [])
        turns.append(
            {
                "input": data,
                "input_mime": mime_type,
                "spectrogram": spectrogram,
                "text": text,
                "output": output_audio_bytes,
            }
        )
        # ponytail: UI memory cap only; model-side chat history is untouched.
        st.session_state.turns = turns[-MAX_TURNS:]


if __name__ == "__main__":
    main()
