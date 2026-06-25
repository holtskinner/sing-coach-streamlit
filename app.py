import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech
from google.genai.chats import Chat
from google.genai.types import GenerateContentConfig, Part

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

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
            system_instruction="Be as brief as possible. You are an expert singing coach. Help improve the singing performance of the audio. Be conversational, and straight to the point.",
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


def play_audio(audio_bytes: bytes) -> None:
    """Plays the audio from a byte stream."""
    if audio_bytes is not None:
        try:
            st.audio(audio_bytes, format="audio/wav", autoplay=True)
        except Exception as e:  # pylint: disable=broad-except
            st.error(f"Error playing audio: {e}")


def generate_audio(text: str, voice_name: str = VOICE_NAME, language_code: str = LANGUAGE_CODE) -> bytes:
    """Generates audio from text using Google Cloud Text-to-Speech."""
    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
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

    audio_input = st.audio_input("Record a voice message")

    if audio_input:
        user_input = Part.from_bytes(data=audio_input.getvalue(), mime_type="audio/wav")

        instruction = "Give feedback on the following audio of singing."
        response = chat.send_message(message=[instruction, user_input])

        with st.chat_message("assistant"):
            st.markdown(response.text)

        output_audio_bytes = generate_audio(
            response.text,
        )

        if output_audio_bytes:
            play_audio(output_audio_bytes)

        audio_input = None


if __name__ == "__main__":
    main()
