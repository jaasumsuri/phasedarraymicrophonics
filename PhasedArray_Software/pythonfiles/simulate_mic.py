#######################################
#
#      Phased Array Microphonics
# 
# This file is meant for the the model 
# that will simulate the microphone array system
#
#       Author : Neil Nainani
#       Date : 09/29/2025
#
#######################################

#Imports
import torch
import torchaudio
import torch.nn as nn

class MicSimNet(nn.Module):
    def __init__(self, num_input_mics):
        super().__init__()
        # This part processes the sound.
        self.audio_tower = nn.Sequential(
            nn.Conv2d(in_channels=num_input_mics, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # This part processes the 3D (x, y, z) position.
        self.pos_tower = nn.Sequential(
            nn.Linear(in_features=3, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16)
        )
        # This part takes the combined info and reconstructs the predicted sound.
        self.decoder = nn.Sequential(
            # Input channels = 16 (from audio) + 16 (from position) = 32
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        )
    def forward(self, input_spectrograms, target_position):
        """
        Simulate the microphone array signals given a source signal and its position.
        
        Args:
            input_spectrograms (Tensor): The audio signal from the source (1D tensor).
            target_position (Tensor): The position of the sound source (3D tensor).
        """
        #Process audio and position information separately
        audio_features = self.audio_tower(input_spectrograms)
        pos_features = self.pos_tower(target_position)

        #Combine both streams of information
        b, _, freq_bins, time_frames = audio_features.shape
        pos_features = pos_features.view(b, 16, 1, 1).expand(b, 16, freq_bins, time_frames)

        #Concatenate along the channel dimension
        combined_features = torch.cat((audio_features, pos_features), dim=1)

        #Decode to get the predicted microphone signals
        predicted_spec = self.decoder(combined_features) 
        return predicted_spec 
    
def run_sanity_check():
    """
    A simple sanity check to ensure the model runs without errors.
    """
    # --- Configuration ---
    num_input_mics = 3  # Assuming 4 total mics, so 3 are used as input.
    batch_size = 4      # A small batch of 4 examples.
    freq_bins = 513     # A typical number of frequency bins for a spectrogram.
    time_frames = 100   # A typical number of time frames.

    # --- Step 1: Create an instance of the model ---
    model = MicSimNet(num_input_mics=num_input_mics)
    print("Model created successfully.")

    # --- Step 2: Create FAKE input data with the correct shape ---
    print("\nCreating fake data for testing...")
    fake_specs = torch.randn(batch_size, num_input_mics, freq_bins, time_frames)
    fake_pos = torch.randn(batch_size, 3)
    
    print(f"Shape of fake spectrograms: {fake_specs.shape}")
    print(f"Shape of fake positions:   {fake_pos.shape}")

    # --- Step 3: Pass the fake data through the model ---
    try:
        output = model(fake_specs, fake_pos)
        print("\nModel forward pass successful!")
        print(f"Output shape: {output.shape}")
        
        # Check if the output shape is what we expect
        expected_shape = (batch_size, 1, freq_bins, time_frames)
        assert output.shape == expected_shape, "Output shape is incorrect!"
        print("\n✅ Success! The model works and the output shape is correct.")
        
    except Exception as e:
        print(f"\n❌ Error during model forward pass: {e}")

# --- Run the test when the script is executed ---
if __name__ == '__main__':
    run_sanity_check()
