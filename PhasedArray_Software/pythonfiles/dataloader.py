#######################################
#
#      Phased Array Microphonics
# 
# This file is meant to load current session's
# microphone data and convert it to spectrograms for AI model
#
#       Author : Joe Do
#       Date : 11/4/2025
#
#######################################

import os
import scipy.io.wavfile
import torch
import torchaudio
import numpy as np

###########################################
#      session_code (str or int): Unique identifier for the current recording session.
#      base_dir (str, optional): Base directory where session folders are stored. Defaults to "../MICRECORD".
#      n_fft (int, optional): Number of FFT components for spectrogram computation. Defaults to 1024.
###########################################

def curr_session_mics(session_code, base_dir="../MICRECORD", n_fft=1024):
    """
    Dynamically loads all real mic wav files for a session and converts them to spectrograms.
    Returns:
        torch.Tensor: [1, n_mics, freq_bins, time_frames],
        int: Sample rate,
        list: List of Mic filenames
    """
    # Find all wav files for this session's INDIV directory
    indiv_dir = os.path.join(base_dir, str(session_code), "INDIV")
    mic_files = sorted([
        os.path.join(indiv_dir, f)
        for f in os.listdir(indiv_dir)
        if f.endswith(".wav") and f.startswith("Mic")
    ])
    if len(mic_files) == 0:
        raise FileNotFoundError(f"No mic .wav files found in: {indiv_dir}")

    signals = []
    sample_rate = None
    for path in mic_files:
        sr, data = scipy.io.wavfile.read(path)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise ValueError(f"Sample rate mismatch in {path}")
        signals.append(data)

    # Make all signals the same length (truncate or pad with zeros)
    maxlen = max(len(s) for s in signals)
    signals = [np.pad(s, (0, maxlen - len(s))) for s in signals]

    # Convert to spectrograms
    specs = []
    for s in signals:
        s_tensor = torch.tensor(s).float()
        spec = torchaudio.transforms.Spectrogram(n_fft=n_fft)(s_tensor)
        specs.append(spec)
    specs = torch.stack(specs, dim=0).unsqueeze(0)  # [1, n_mics, freq_bins, time_frames]
    return specs, sample_rate, mic_files
