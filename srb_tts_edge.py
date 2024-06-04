import os
import torch
import subprocess
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from modules.latin_to_cyrillic_converter import convert_to_cyrillic

class TTSHandler:
    def __init__(self):
        # Initialization
        ckpt_converter = '/home/ubuntucminya/Desktop/pyapp/OpenVoice/checkpoints/converter'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dir = '/home/ubuntucminya/Desktop/pyapp/OpenVoice/outputs'
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        # Obtain Tone Color Embedding
        reference_speaker = '/home/ubuntucminya/Desktop/pyapp/OpenVoice/ser.mp3'  # Your target/reference speaker audio file
        self.source_se, _ = se_extractor.get_se(reference_speaker, self.tone_color_converter, vad=True)
        self.target_se, _ = se_extractor.get_se(reference_speaker, self.tone_color_converter, vad=True)

    def generate_voice(self, text):
        converted_text = convert_to_cyrillic(text)
        tts_filename = f'redgesponse.wav'
        tts_cmd = [
            "edge-tts",
            "--voice", "sr-RS-SophieNeural",
            # "--voice", "hr-HR-GabrijelaNeural", # SreckoNeural
            "--text", converted_text,
            # "--rate=+33%",
            # "--volume=-95%",
            # "--pitch=-23Hz",
            "--write-media", os.path.join(self.output_dir, tts_filename)
        ]
        subprocess.run(tts_cmd, check=True)

        temp_audio_path = os.path.join(self.output_dir, tts_filename)
        save_path = os.path.join(self.output_dir, f"output_crosslingualdhr.wav")
        self.tone_color_converter.convert(
            audio_src_path=temp_audio_path,
            src_se=self.source_se,
            tgt_se=self.target_se,
            output_path=save_path,
            message="@MyShell"
        )

        return save_path
