# file: example_usage.py
from srb_tts_edge import TTSHandler
from pydub import AudioSegment
from pydub.playback import play

# Instantiate the TTS handler which loads all components and models
tts = TTSHandler()

# Text to be synthesized
text_to_synthesize = "Moj lanac je decentraliziran i voli da ide u cirkus."

# Generate and play the audio five times
for i in range(5):
    print(f"Generating voice output {i+1}...")
    audio_path = tts.generate_voice(text_to_synthesize)
    audio_output = AudioSegment.from_wav(audio_path)
    play(audio_output)
