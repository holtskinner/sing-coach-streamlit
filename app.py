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

MODEL_ID = "gemini-3.1-flash-lite"
VOICE_MODEL_ID = "gemini-3.1-flash-tts-preview"
VOICE_NAME = "Achird"
LANGUAGE_CODE = "en-us"
MAX_TURNS = 20  # UI history cap; trims oldest audio to bound session_state growth


@st.cache_resource
def load_client() -> genai.Client:
    """Load the Google Gen AI client."""
    load_dotenv(override=True)
    return genai.Client(location="global")


def get_chat(client: genai.Client) -> Chat:
    """Per-session chat so history persists across reruns but not across users."""
    if "chat" not in st.session_state:
        st.session_state.chat = client.chats.create(
            model=MODEL_ID,
            config=GenerateContentConfig(
                system_instruction="Be brief and respond for speech. You are an expert singing coach. Help improve the singing performance of singer in the audio. If the performance is especially bad, feel free to be a little snarky (think Simon Cowell) if needed.",
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

    if st.button("Start over") and "turns" in st.session_state:
        del st.session_state["chat"]  # next get_chat() makes a fresh session
        del st.session_state["turns"]
        st.rerun()

    # Replay prior turns; chat.send_message keeps model-side history, this keeps the UI in sync.
    for turn in st.session_state.get("turns", []):
        with st.chat_message("user"):
            st.audio(turn["input"], format="audio/wav")
        with st.chat_message("assistant"):
            st.markdown(turn["text"])
            st.audio(turn["output"], format="audio/wav")

    audio_input = st.audio_input("Record your singing")

    if audio_input:
        data = audio_input.getvalue()
        with st.chat_message("user"):
            st.audio(data, format="audio/wav")

        with st.spinner("Analyzing your singing..."):
            response = chat.send_message(
                message=Part.from_bytes(data=data, mime_type="audio/wav")
            )
            text = response.text or "Sorry, I couldn't analyze that."

        with st.spinner("Generating voice feedback..."):
            output_audio_bytes = generate_audio(text)

        with st.chat_message("assistant"):
            st.markdown(text)
            st.audio(output_audio_bytes, format="audio/wav", autoplay=True)

        turns = st.session_state.setdefault("turns", [])
        turns.append({"input": data, "text": text, "output": output_audio_bytes})
        # ponytail: UI memory cap only; model-side chat history is untouched.
        st.session_state.turns = turns[-MAX_TURNS:]


if __name__ == "__main__":
    main()
