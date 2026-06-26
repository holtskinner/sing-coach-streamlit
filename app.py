import io
import logging
import urllib.parse
import wave

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

MODEL_ID = "gemini-3.1-flash-lite"
VOICE_MODEL_ID = "gemini-3.1-flash-tts-preview"
VOICE_NAME = "Achird"
LANGUAGE_CODE = "en-us"
MAX_TURNS = 20  # UI history cap; trims oldest audio to bound session_state growth

SYSTEM_INSTRUCTION = """\
You are AriaCoach, an expert vocal coach giving spoken feedback on a short \
singing clip the user just recorded. Your goal is to help them sing better, \
fast.

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


def get_chat(client: genai.Client) -> Chat:
    """Per-session chat so history persists across reruns but not across users."""
    if "chat" not in st.session_state:
        st.session_state.chat = client.chats.create(
            model=MODEL_ID,
            config=GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL),
            ),
        )
    return st.session_state.chat


client = load_client()
chat = get_chat(client)


def generate_audio(text: str) -> bytes:
    """Generates WAV audio from text using Gemini TTS (returns 24kHz PCM as WAV)."""
    response = client.models.generate_content(
        model=VOICE_MODEL_ID,
        contents=text,
        config=GenerateContentConfig(
            speech_config=SpeechConfig(
                language_code=LANGUAGE_CODE,
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=VOICE_NAME)
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
        if source:
            data = source.getvalue()
            mime_type = source.type or "audio/wav"
            message = Part.from_bytes(
                data=data,
                mime_type=mime_type,
                media_resolution=PartMediaResolution(
                    level=PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
                ),
            )
            with st.chat_message("user"):
                st.audio(data, format=mime_type)
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
                output_audio_bytes = generate_audio(text)
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
                "text": text,
                "output": output_audio_bytes,
            }
        )
        # ponytail: UI memory cap only; model-side chat history is untouched.
        st.session_state.turns = turns[-MAX_TURNS:]


if __name__ == "__main__":
    main()
