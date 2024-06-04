import os
import torch
from dotenv import load_dotenv
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from openai import OpenAI

class TTSHandler:
    def __init__(self):
        # Initialization
        ckpt_converter = '/home/ubuntucminya/Desktop/pyapp/OpenVoice/checkpoints/converter'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dir = '/home/ubuntucminya/Desktop/pyapp/OpenVoice/outputs'
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        # Load the OpenAI API key
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key)

        # Obtain Tone Color Embedding
        reference_speaker = '/home/ubuntucminya/Desktop/pyapp/OpenVoice/ser.mp3'  # Your target/reference speaker audio file
        self.source_se, _ = se_extractor.get_se(reference_speaker, self.tone_color_converter, vad=True)
        self.target_se, _ = se_extractor.get_se(reference_speaker, self.tone_color_converter, vad=True)

    def generate_voice(self, text):
        # Generate TTS audio
        response = self.openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        temp_audio_path = f"{self.output_dir}/temp_audio.mp3"
        with open(temp_audio_path, 'wb') as f:
            f.write(response.content)

        save_path = f"{self.output_dir}/output_crosslingual.wav"
        self.tone_color_converter.convert(
            audio_src_path=temp_audio_path,
            src_se=self.source_se,
            tgt_se=self.target_se,
            output_path=save_path,
            message="@MyShell"
        )

        return save_path
