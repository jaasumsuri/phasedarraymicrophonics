#######################################
#
#      Phased Array Microphonics
# AI Microphone Utility Functions
# Helper functions for AI phantom microphone generation
#
#   Authors : Joe Do
#           Date : 11/2025
#
#######################################

import sys
import json
import numpy as np
import torch
import torchaudio


def get_ai_config():
    """Get AI microphone configuration from command-line argument.
    Expects sys.argv[7] to be a JSON string with the configuration.
    Returns None if not provided or disabled."""
    if len(sys.argv) > 7:
        try:
            # Parse JSON string from 7th argument
            config = json.loads(sys.argv[7])
            return config
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse AI config JSON: {e}")
            return None
        except Exception as e:
            print(f"Warning: Error reading AI config: {e}")
            return None
    return None  # Disabled by default


def spectrogram_to_waveform(phantom_spec, n_fft=1024, hop_length=None):
    """
    Convert spectrogram tensor back to waveform numpy array.
    
    Args:
        phantom_spec: Torch tensor of shape [1, 1, freq_bins, time_frames] or [freq_bins, time_frames]
        n_fft: FFT window size used for spectrogram
        hop_length: Hop length (defaults to n_fft // 4)
    
    Returns:
        numpy array of waveform data
    """
    # Handle different input shapes
    if len(phantom_spec.shape) == 4:
        # [batch, channels, freq, time] -> [freq, time]
        spec = phantom_spec[0, 0].detach().cpu()
    elif len(phantom_spec.shape) == 3:
        # [channels, freq, time] -> [freq, time]
        spec = phantom_spec[0].detach().cpu()
    else:
        # [freq, time]
        spec = phantom_spec.detach().cpu()
    
    if hop_length is None:
        hop_length = n_fft // 4
    
    # Convert magnitude spectrogram back to waveform using Griffin-Lim or ISTFT
    # For now, use ISTFT (Inverse Short-Time Fourier Transform)
    # Note: This assumes the spectrogram is a complex spectrogram or magnitude
    # You may need to adjust based on how torchaudio.transforms.Spectrogram works
    try:
        # If spectrogram is magnitude-only, we need to use Griffin-Lim
        # For simplicity, let's use ISTFT (requires complex spectrogram)
        # Create a dummy phase or use Griffin-Lim algorithm
        waveform = torchaudio.functional.griffinlim(
            spec, 
            n_fft=n_fft, 
            hop_length=hop_length,
            length=None,
            window=torch.hann_window(n_fft),
            n_iter=32
        )
    except:
        # Fallback: simple reconstruction (may not be perfect)
        waveform = torch.istft(
            spec.unsqueeze(0), 
            n_fft=n_fft, 
            hop_length=hop_length,
            length=None,
            window=torch.hann_window(n_fft)
        ).squeeze(0)
    
    return waveform.numpy()


def calculate_phantom_distance(phantom_pos, SOD, SAD):
    """
    Calculate distance from phantom microphone position to target sound.
    
    Args:
        phantom_pos: Tuple (x, y, z) or list [x, y, z] - phantom mic position
        SOD: Opposite distance to target from reference point
        SAD: Adjacent distance to target from reference point
    
    Returns:
        Distance in meters (hypotenuse)
    """
    x, y, z = phantom_pos
    
    # Calculate distance components
    # Assuming phantom mic is at (x, y, z) and target is at (SOD, SAD, 0)
    # You may need to adjust this based on your coordinate system
    dx = SOD - x
    dy = SAD - y
    dz = 0 - z  # Assuming target is at z=0
    
    # Calculate hypotenuse distance
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    return distance
