import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.chats import Chat
from google.genai.types import GenerateContentConfig, Part

load_dotenv(override=True)
# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


MODEL_ID = "gemini-3.5-flash"
TTS_MODEL_ID = "gemini-3.1-flash-tts-preview"

@st.cache_resource
def load_client() -> genai.Client:
    """Load Client."""
    return genai.Client()

@st.cache_resource
def load_chat() -> Chat:
    """Load Google Gen AI Client."""
    return client.chats.create(
        model=MODEL_ID,
        config=GenerateContentConfig(
            system_instruction="You are an expert singing coach. Help improve the singing performance of the audio. Be conversational, and striaght to the point."
        ),
    )

client = load_client()
chat = load_chat()


def play_audio(audio_bytes: bytes) -> None:
    """Plays the audio from a byte stream."""
    if audio_bytes is not None:
        try:
            st.audio(audio_bytes, format="audio/wav", autoplay=True)
        except Exception as e:  # pylint: disable=broad-except
            st.error(f"Error playing audio: {e}")

def generate_audio(text: str) -> bytes:
    """Generates audio from text using Gemini Text-to-Speech."""
    response = client.models.generate_content(
        model=TTS_MODEL_ID,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Fenrir")
                )
            ),
        ),
    )
    return response.candidates[0].content.parts[0].inline_data.data


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title("AriaCoach - Singing Teacher")

    audio_input = st.audio_input("Record a voice message")

    if audio_input:
        user_input = Part.from_bytes(data=audio_input.getvalue(), mime_type="audio/wav")

        instruction = "Give feedback on the following audio of singing."

        assistant_response = chat.send_message(message=[instruction, user_input]).text

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        output_audio_bytes = generate_audio(
            assistant_response,
        )

        if output_audio_bytes:
            play_audio(output_audio_bytes)

        audio_input = None


if __name__ == "__main__":
    main()
