import io
import wave

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.chats import Chat
from google.genai.types import (
    GenerateContentConfig,
    Part,
    PrebuiltVoiceConfig,
    SpeechConfig,
    ThinkingConfig,
    ThinkingLevel,
    VoiceConfig,
)

MODEL_ID = "gemini-3.5-flash"
VOICE_MODEL_ID = "gemini-3.1-flash-tts-preview"
VOICE_NAME = "Aoede"
LANGUAGE_CODE = "en-us"


@st.cache_resource
def load_clients() -> tuple[Chat, genai.Client]:
    """Load Google Gen AI chat session and client."""
    load_dotenv(override=True)
    client = genai.Client()

    chat = client.chats.create(
        model=MODEL_ID,
        config=GenerateContentConfig(
            system_instruction="Be as brief as possible and respond for speech. You are an expert singing coach. Help improve the singing performance of singer in the audio.",
            thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL),
        ),
    )
    return chat, client


chat, client = load_clients()


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
    assert parts and parts[0].inline_data, "Gemini TTS returned no audio"
    pcm = parts[0].inline_data.data
    assert pcm is not None, "Gemini TTS returned no audio"

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm)
    return buffer.getvalue()


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title("AriaCoach - Singing Teacher")

    audio_input = st.audio_input("Record your singing")

    if audio_input:
        with st.spinner("Analyzing your singing..."):
            response = chat.send_message(
                message=Part.from_bytes(
                    data=audio_input.getvalue(), mime_type="audio/wav"
                )
            )
            text = response.text or "Sorry, I couldn't analyze that."

        with st.chat_message("assistant"):
            st.markdown(text)

        with st.spinner("Generating voice feedback..."):
            output_audio_bytes = generate_audio(text)

        st.audio(output_audio_bytes, format="audio/wav", autoplay=True)


if __name__ == "__main__":
    main()
