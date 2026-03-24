import torch
import librosa
import numpy as np
from fadtk import VGGishModel, MERTModel, CLAPLaionModel  # Import pre-trained models for audio embeddings
from tqdm import tqdm  # Progress bar for iterative tasks

class Audio_to_Embedding_Tensor:
    """
    A class to convert audio files into embedding tensors using various pre-trained models.
    Supports multiple embedding types: CLAP, CLAP-Music, VGGish, and MERT.
    """

    def __init__(self, duration=5, embedding_type="clap"):
        """
        Initialize the Audio_to_Embedding_Tensor class.

        Args:
            duration (int): Default duration (in seconds) for audio loading. Defaults to 5.
            embedding_type (str): Type of embedding model to use. Options: "clap", "clap-music", "vggish", "mert".
        """
        self.duration = duration  # Duration of audio to process (in seconds)
        self.embedding_type = embedding_type  # Type of embedding model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU

        # Initialize the appropriate model based on the embedding type
        match embedding_type:
            case "clap":
                # CLAP model for general audio embeddings
                self.model = CLAPLaionModel(type="audio")
                self.sample_rate = 48000  # Sample rate for CLAP model

            case "clap-music":
                # CLAP model optimized for music audio embeddings
                self.model = CLAPLaionModel(type="music")
                self.sample_rate = 48000  # Sample rate for CLAP-Music model

            case "vggish":
                # VGGish model for audio embeddings
                self.model = VGGishModel()
                self.sample_rate = 16000  # Sample rate for VGGish model

            case "mert":
                # MERT model for audio embeddings
                self.model = MERTModel()
                self.sample_rate = 24000  # Sample rate for MERT model

            case _:
                # Raise an error for unsupported embedding types
                raise ValueError(f"Unknown embedding type: {self.embedding_type}")

        # Load the model weights
        self.model.load_model()

    def load_audio(self, file_path, crop_to_duration=None, pad_to_duration=None):
        """
        Load and preprocess an audio file.

        Args:
            file_path (str): Path to the audio file.
            crop_to_duration (float, optional): Crop audio to this duration (in seconds). Defaults to None.
            pad_to_duration (float, optional): Pad audio to this duration (in seconds). Defaults to None.

        Returns:
            np.ndarray: Processed audio as a numpy array.
        """
        # Load audio file with the specified sample rate and duration
        audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        # Normalize audio to ensure consistent amplitude
        audio = librosa.util.normalize(audio)

        # Crop audio if required
        if crop_to_duration:
            target_length = int(self.sample_rate * crop_to_duration)
            if audio.shape[0] > target_length:
                audio = audio[:target_length]

        # Pad audio if required
        if pad_to_duration:
            target_length = int(self.sample_rate * pad_to_duration)
            if audio.shape[0] < target_length:
                audio = np.pad(audio, (0, target_length - audio.shape[0]), mode='constant')

        return audio

    def load_all_audios(self, file_paths, crop_to_duration=None, pad_to_duration=None):
        """
        Load and preprocess multiple audio files.

        Args:
            file_paths (list): List of paths to audio files.
            crop_to_duration (float, optional): Crop each audio to this duration (in seconds). Defaults to None.
            pad_to_duration (float, optional): Pad each audio to this duration (in seconds). Defaults to None.

        Returns:
            list: List of processed audio numpy arrays.
        """
        audios = []
        # Iterate over file paths and load each audio file
        for file_path in tqdm(file_paths, total=len(file_paths), desc="Loading audios"):
            audios.append(self.load_audio(file_path, crop_to_duration=crop_to_duration, pad_to_duration=pad_to_duration))
        return audios

    def get_embedding(self, audio):
        """
        Generate an embedding tensor for the provided audio.

        Args:
            audio (np.ndarray): Audio as a numpy array.

        Returns:
            torch.Tensor: Embedding tensor for the audio.
        """
        # Generate embedding using the loaded model
        embedding = self.model._get_embedding(audio)
        # Average the embedding across the time dimension to get a single vector
        audio_embedding = torch.mean(embedding, dim=0)

        return audio_embedding
