import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import texttospeech

# ponytail: Simple configuration constants. No complex config parsers or abstractions.
MODEL = "gemini-3.5-flash"
TTS_MODEL = "gemini-3.1-flash-tts-preview"
TTS_VOICE = "Aoede"
TTS_LOCALE = "en-US"


def main():
    load_dotenv(override=True)

    # Initialize both clients
    llm_client = genai.Client()
    tts_client = texttospeech.TextToSpeechClient()

    # Initialize chat session
    chat = llm_client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL
            ),
            audio_timestamp=True,
            system_instruction="You are a helpful and concise assistant. Keep your responses short and natural for speech.",
        ),
    )

    print("=== Gemini Text & Audio Streaming CLI Tool ===")
    print("Type your prompt and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input("User: ")
            if not prompt.strip():
                continue

            print("Gemini: ", end="", flush=True)

            # The TTS request generator pulls LLM chunks directly off the wire and
            # forwards them to TTS as soon as they arrive, so synthesis starts before
            # the LLM has finished generating the full response.
            config_request = texttospeech.StreamingSynthesizeRequest(
                streaming_config=texttospeech.StreamingSynthesizeConfig(
                    voice=texttospeech.VoiceSelectionParams(
                        name=TTS_VOICE, language_code=TTS_LOCALE, model_name=TTS_MODEL
                    )
                )
            )

            produced_text = False

            def request_generator():
                nonlocal produced_text
                yield config_request
                for chunk in chat.send_message_stream(prompt):
                    if chunk.text:
                        produced_text = True
                        print(chunk.text, end="", flush=True)
                        yield texttospeech.StreamingSynthesizeRequest(
                            input=texttospeech.StreamingSynthesisInput(text=chunk.text)
                        )
                print()  # Newline after the LLM finishes streaming text

            # Start requesting the TTS stream. The first request half-closes only
            # once the LLM stream above is exhausted.
            streaming_responses = tts_client.streaming_synthesize(request_generator())

            # ponytail: OutputStream's own buffer decouples synthesis from playback,
            # so write chunks straight to the device as they arrive. No queue/thread
            # needed unless a profiler shows the network read stalling on the device.
            with sd.OutputStream(
                samplerate=24000, channels=1, dtype="int16"
            ) as audio_stream:
                for response in streaming_responses:
                    if response.audio_content:
                        audio_stream.write(
                            np.frombuffer(response.audio_content, dtype=np.int16)
                        )

            if not produced_text:
                continue

            print("\n[Playback Finished]\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")


if __name__ == "__main__":
    main()
