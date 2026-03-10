import torch
import librosa
import numpy as np
from fadtk import VGGishModel, MERTModel, CLAPLaionModel
from tqdm import tqdm

class Audio_to_Embedding_Tensor:
    def __init__(self, duration=5, embedding_type="clap"):
        self.duration = duration
        self.embedding_type = embedding_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        match embedding_type:
            case "clap":
                # Initialize the CLAP model
                self.model = CLAPLaionModel(type="audio")
                self.sample_rate = 48000

            case "clap-music":
                # Initialize the CLAP model
                self.model = CLAPLaionModel(type="music")
                self.sample_rate = 48000

            case "vggish":
                # Initialize the VGG model
                self.model = VGGishModel()
                self.sample_rate = 16000

            case "mert":
                # Initialize the MERT model
                self.model = MERTModel()
                self.sample_rate = 24000

            case _:
                raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        self.model.load_model()

    def load_audio(self, file_path, crop_to_duration=None, pad_to_duration=None):
        audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        audio = librosa.util.normalize(audio)
        if crop_to_duration:
            target_length = int(self.sample_rate * crop_to_duration)
            if audio.shape[0] > target_length:
                audio = audio[:target_length]
        if pad_to_duration:
            target_length = int(self.sample_rate * pad_to_duration)
            if audio.shape[0] < target_length:
                audio = np.pad(audio, (0, target_length - audio.shape[0]), mode='constant')
        return audio
    
    def load_all_audios(self, file_paths, crop_to_duration=None, pad_to_duration=None):
        audios = []
        for file_path in tqdm(file_paths, total=len(file_paths), desc="Loading audios"):
            audios.append(self.load_audio(file_path, crop_to_duration=crop_to_duration, pad_to_duration=pad_to_duration))
        return audios

    def get_embedding(self, audio):

        embedding = self.model._get_embedding(audio)
        audio_embedding = torch.mean(embedding, dim=0)

        return audio_embedding
    