# file: tts_system.py
import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

class TTSHandler:
    def __init__(self):
        # Setup paths
        self.ckpt_base = '/home/ubuntucminya/Desktop/pyapp/OpenVoice/checkpoints/base_speakers/EN'
        self.ckpt_converter = '/home/ubuntucminya/Desktop/pyapp/OpenVoice/checkpoints/converter'
        self.output_dir = 'outputs'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.output_dir, exist_ok=True)

        # Load models
        self.base_speaker_tts = BaseSpeakerTTS(f'{self.ckpt_base}/config.json', device=self.device)
        self.base_speaker_tts.load_ckpt(f'{self.ckpt_base}/checkpoint.pth')
        self.tone_color_converter = ToneColorConverter(f'{self.ckpt_converter}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{self.ckpt_converter}/checkpoint.pth')

        # Load target speaker embedding
        reference_speaker = '/home/ubuntucminya/Desktop/pyapp/OpenVoice/luna.mp3'
        self.target_se, _ = se_extractor.get_se(reference_speaker, self.tone_color_converter, target_dir='processed', vad=True)

        # Load source embedding for tone matching (optional, only if different from target)
        self.source_se = torch.load(f'{self.ckpt_base}/en_default_se.pth').to(self.device)

    def generate_voice(self, text):
        src_path = f'{self.output_dir}/tmp.wav'
        self.base_speaker_tts.tts(text, src_path, speaker='default', speed=1.5)

        save_path = f'{self.output_dir}/output_cloned_voice.wav'
        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=self.source_se,
            tgt_se=self.target_se,
            output_path=save_path,
            message="@MyShell"
        )
        return save_path

