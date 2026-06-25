import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech
from google.genai.chats import Chat
from google.genai.types import (
    GenerateContentConfig,
    Part,
    ThinkingConfig,
    ThinkingLevel,
)

MODEL_ID = "gemini-3.5-flash"
VOICE_MODEL_ID = "gemini-3.1-flash-tts-preview"
VOICE_NAME = "Aoede"
LANGUAGE_CODE = "en-us"


@st.cache_resource
def load_chat() -> Chat:
    """Load Google Gen AI Client."""
    load_dotenv(override=True)
    client = genai.Client()

    return client.chats.create(
        model=MODEL_ID,
        config=GenerateContentConfig(
            system_instruction="Be as brief as possible and respond for speech. You are an expert singing coach. Help improve the singing performance of singer in the audio.",
            thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL),
        ),
    )


@st.cache_resource
def load_tts_client() -> texttospeech.TextToSpeechClient:
    """Load Text-to-Speech Client."""
    return texttospeech.TextToSpeechClient(
        client_options=ClientOptions(api_endpoint="us-texttospeech.googleapis.com")
    )


chat = load_chat()
tts_client = load_tts_client()


def generate_audio(
    text: str, voice_name: str = VOICE_NAME, language_code: str = LANGUAGE_CODE
) -> bytes:
    """Generates audio from text using Google Cloud Text-to-Speech."""
    response = tts_client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            model_name=VOICE_MODEL_ID,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        ),
    )
    return response.audio_content


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

        if output_audio_bytes:
            st.audio(output_audio_bytes, format="audio/mp3", autoplay=True)


if __name__ == "__main__":
    main()
