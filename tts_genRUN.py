
# # Example usage in another script or an interactive session
# from tts_gen import TTSHandler
# from pydub import AudioSegment
# from pydub.playback import play

# tts = TTSHandler()

# # Whenever you need to generate speech
# text_to_synthesize = "Dobar dan, kako ste vi danas."
# audio_path = tts.generate_voice(text_to_synthesize)
# audio_output = AudioSegment.from_wav(audio_path)
# play(audio_output)


# file: example_usage.py
from tts_gen import TTSHandler  # Assuming TTSHandler is the class defined in tts_gen.py
from pydub import AudioSegment
from pydub.playback import play

# Instantiate the TTS handler which loads all components and models
tts = TTSHandler()

# Text to be synthesized
text_to_synthesize = "Dobar dan, kako ste vi danas."

# Generate and play the audio five times
for i in range(5):
    print(f"Generating voice output {i+1}...")
    audio_path = tts.generate_voice(text_to_synthesize)
    audio_output = AudioSegment.from_wav(audio_path)
    play(audio_output)
