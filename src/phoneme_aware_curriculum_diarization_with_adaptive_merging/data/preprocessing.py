"""Audio preprocessing utilities for speaker diarization."""

import logging
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from scipy.signal import butter, lfilter

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Preprocess audio data for speaker diarization with phoneme awareness."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        duration: float = 3.0,
        augment: bool = True,
    ):
        """Initialize audio preprocessor.

        Args:
            sample_rate: Target sample rate in Hz.
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            n_mels: Number of mel filterbanks.
            duration: Duration of audio segments in seconds.
            augment: Whether to apply data augmentation.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.duration = duration
        self.augment = augment

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and resample to target sample rate.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (audio waveform, sample rate).
        """
        try:
            # Try torchaudio first
            try:
                waveform, sr = torchaudio.load(audio_path)
            except (ImportError, RuntimeError) as e:
                # Fallback to scipy for wav files
                from scipy.io import wavfile
                sr, audio_data = wavfile.read(audio_path)
                # Convert to float32 and normalize
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                waveform = torch.from_numpy(audio_data).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.t()

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            return waveform, self.sample_rate
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram features from audio waveform.

        Args:
            waveform: Audio waveform tensor [1, num_samples].

        Returns:
            Mel spectrogram tensor [n_mels, time_frames].
        """
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec.squeeze(0)

    def segment_audio(
        self, waveform: torch.Tensor, overlap: float = 0.5
    ) -> List[torch.Tensor]:
        """Segment audio into fixed-duration chunks with overlap.

        Args:
            waveform: Audio waveform tensor [1, num_samples].
            overlap: Overlap ratio between segments (0-1).

        Returns:
            List of audio segments.
        """
        num_samples = waveform.shape[1]
        segment_samples = int(self.duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))

        segments = []
        start = 0
        while start + segment_samples <= num_samples:
            segment = waveform[:, start:start + segment_samples]
            segments.append(segment)
            start += hop_samples

        # Handle last segment if needed
        if start < num_samples and len(segments) > 0:
            last_segment = waveform[:, -segment_samples:]
            if last_segment.shape[1] == segment_samples:
                segments.append(last_segment)

        return segments

    def add_noise(self, waveform: torch.Tensor, snr_db: float = 15.0) -> torch.Tensor:
        """Add Gaussian noise to waveform.

        Args:
            waveform: Audio waveform tensor.
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Noisy waveform.
        """
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def time_stretch(self, waveform: torch.Tensor, rate: float = 1.1) -> torch.Tensor:
        """Apply time stretching augmentation.

        Args:
            waveform: Audio waveform tensor.
            rate: Stretch rate (>1 speeds up, <1 slows down).

        Returns:
            Time-stretched waveform.
        """
        waveform_np = waveform.squeeze().numpy()
        stretched = librosa.effects.time_stretch(waveform_np, rate=rate)
        return torch.from_numpy(stretched).unsqueeze(0)

    def normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform to zero mean and unit variance.

        Args:
            waveform: Audio waveform tensor.

        Returns:
            Normalized waveform.
        """
        mean = torch.mean(waveform)
        std = torch.std(waveform)
        return (waveform - mean) / (std + 1e-9)


def create_speaker_segments(
    audio_path: str,
    speaker_timestamps: List[Tuple[float, float, int]],
    sample_rate: int = 16000,
) -> List[Dict[str, any]]:
    """Create speaker segments from audio with timestamps.

    Args:
        audio_path: Path to audio file.
        speaker_timestamps: List of (start_time, end_time, speaker_id) tuples.
        sample_rate: Audio sample rate.

    Returns:
        List of segment dictionaries with waveform and speaker label.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)

        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        segments = []
        for start_time, end_time, speaker_id in speaker_timestamps:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            segment_waveform = waveform[:, start_sample:end_sample]

            segments.append({
                'waveform': segment_waveform,
                'speaker_id': speaker_id,
                'start_time': start_time,
                'end_time': end_time,
            })

        return segments
    except Exception as e:
        logger.error(f"Error creating speaker segments: {e}")
        return []
