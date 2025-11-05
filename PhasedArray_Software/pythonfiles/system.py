#######################################
#
#      Phased Array Microphonics
# Main Phased Array Algorithm Script 
# Combines all other Scripts to output summation 
# file for Phased Alteration and Graphs
#
#   Authors : Kyle Gruen, Ibrahim Shabini, Joe Do
#           Date : 02/27/2024
#
#######################################

# # Import External Scripts # 
import conversion as con
import distanceCalc as dc
import figureCreate as fc
import wavCollection as wc
import generateCode as gc
from dataloader import curr_session_mics
from simulate_mic import MicSimNet

# Import Python Libraries #
import numpy as np
import pandas as pd
import sys
import torch
import torchaudio

# Import AI microphone utility functions
from ai_mic_util import get_ai_config, spectrogram_to_waveform, calculate_phantom_distance

def main():
    
    data = {
        'SAD': [float(sys.argv[1])], 
        'SOD': [float(sys.argv[2])],
        'SAMRATE': 44000,
        'MIC_COUNT': [int(sys.argv[3])], 
        'TOTALSPREAD': [float(sys.argv[4])],
        'TYPE': [str(sys.argv[5])],
        'NOISE': [str(sys.argv[6])]
    }
    
    recordingInformation = pd.DataFrame(data)
    
    CODE = gc.generate(recordingInformation['MIC_COUNT'].iloc[0], 
                       recordingInformation['TYPE'].iloc[0], 
                       recordingInformation['NOISE'].iloc[0])
    
    recordingInformation['CODE'] = CODE

    fc.generateDiagram(recordingInformation)
    
    CODE = recordingInformation['CODE'].iloc[0]
    
    FS = recordingInformation['SAMRATE'].iloc[0] # AVG number of samples obtained per second (Sample Rate)
    TOTAL_SPREAD = recordingInformation['TOTALSPREAD'].iloc[0] # Total Spread of Phased Array from Mic 1 -> Mic N
    MIC_COUNT = recordingInformation['MIC_COUNT'].iloc[0] # Total Number of Microphones 
    
    SCALAR = TOTAL_SPREAD / (MIC_COUNT - 1) # Scalar Multiple For Equidistant Microphones
    
    SYS_ADJ_DIST = recordingInformation['SAD'].iloc[0] # Adjacent Distance to Target from Left Most Microphone
    SYS_OPP_DIST = recordingInformation['SOD'].iloc[0] # Opposite Distance to Target from Left Most Microphone

    FORLOOPARR = np.arange(MIC_COUNT) # Iteration Array For Number of Microphones

    # Record Audio 
    wc.recordAudio(CODE, FS, MIC_COUNT)
    
    # Calculate Mic Distances to Target Values # 
    Mic_Distance_Target = np.zeros(MIC_COUNT, dtype=float)
    Mic_Distance_Target = dc.calcDistance(TOTAL_SPREAD, MIC_COUNT, SYS_OPP_DIST, SYS_ADJ_DIST)
    print("Mic Distances to Target : " + str(Mic_Distance_Target) + "\n")

    # Find ArgMin/Max and Min/Max of Distance Values # 
    minDis, minIndex = np.min(Mic_Distance_Target), np.argmin(Mic_Distance_Target)
    maxDis, maxIndex = np.max(Mic_Distance_Target), np.argmax(Mic_Distance_Target)
    print("Max Distance / Mic : " + str(maxDis) + "/ " + str(maxIndex) + "\n")

    # Calculate How Many Samples Need to be Appended to Each Microphone Array #
    Mic_Sample_Delay = np.zeros(MIC_COUNT, dtype=float)
    Mic_Sample_Delay = dc.calcSample(Mic_Distance_Target, FS, maxIndex)
    print("Mic Sample Delays : " + str(Mic_Sample_Delay) + "\n")

    # Create NumPy Array of Arrays to store microphone data #
    mic_Signal_Cells = [np.array([]) for _ in range(MIC_COUNT)]

    # Convert Wav Files into CSV Data and Store in mic_Signal_Cells #
    for mic in FORLOOPARR: 
        file_path = "../MICRECORD/" + CODE + "/INDIV/Mic" + str(mic + 1) + "_" + CODE + ".wav" 
        mic_Signal_Cells[mic] = con.convertWavCSV(file_path, FS)
    
    # Load current session's microphone data and convert to spectrograms
    specs, _, _ = curr_session_mics(CODE)
    
    # ============ AI PHANTOM MIC GENERATION ============
    ai_config = get_ai_config()
    
    if ai_config and ai_config.get('enabled', False):
        print("AI Phantom Mics: ENABLED")
        print(f"Generating {ai_config.get('num_phantom_mics', 0)} phantom microphone(s)...")
        
        # Initialize AI model
        model = MicSimNet(num_input_mics=MIC_COUNT)
        model.eval()  # Set to evaluation mode
        
        # Generate phantom mic signals for each position
        with torch.no_grad():  # No gradients needed for inference
            for i, position in enumerate(ai_config.get('positions', [])):
                print(f"Generating Phantom Mic #{i+1} at position {position}...")
                
                # Convert position to tensor [1, 3]
                pos_tensor = torch.tensor([position], dtype=torch.float32)
                
                # Run AI model to get phantom spectrogram
                phantom_spec = model(specs, pos_tensor)  # Output: [1, 1, freq_bins, time_frames]
                
                # Convert spectrogram back to waveform
                phantom_wave = spectrogram_to_waveform(phantom_spec, n_fft=1024)
                
                # Ensure same length as real mic signals (take minimum length)
                min_length = min(len(sig) for sig in mic_Signal_Cells)
                if len(phantom_wave) > min_length:
                    phantom_wave = phantom_wave[:min_length]
                elif len(phantom_wave) < min_length:
                    phantom_wave = np.pad(phantom_wave, (0, min_length - len(phantom_wave)), mode='constant')
                
                # Add to mic_Signal_Cells
                mic_Signal_Cells.append(phantom_wave)
                
                # Update MIC_COUNT and add to distance calculations
                MIC_COUNT += 1
                
                # Calculate distance for this phantom mic
                phantom_distance = calculate_phantom_distance(position, SYS_OPP_DIST, SYS_ADJ_DIST)
                Mic_Distance_Target = np.append(Mic_Distance_Target, phantom_distance)
                print(f"  Phantom Mic #{i+1} distance to target: {phantom_distance:.3f} meters")
        
        # Update FORLOOPARR for new total mic count
        FORLOOPARR = np.arange(MIC_COUNT)
        
        # Recalculate max/min distances including phantom mics
        minDis, minIndex = np.min(Mic_Distance_Target), np.argmin(Mic_Distance_Target)
        maxDis, maxIndex = np.max(Mic_Distance_Target), np.argmax(Mic_Distance_Target)
        print(f"Updated Max Distance / Mic: {maxDis:.3f} / {maxIndex}\n")
        
        # Recalculate sample delays including phantom mics
        Mic_Sample_Delay = np.zeros(MIC_COUNT, dtype=float)
        Mic_Sample_Delay = dc.calcSample(Mic_Distance_Target, FS, maxIndex)
        print("Updated Mic Sample Delays (including phantom mics): " + str(Mic_Sample_Delay) + "\n")
    else:
        print("AI Phantom Mics: DISABLED (using real mics only)\n")

    # Generate Figure Displaying all Microphone Waves on MatPlot #
    fc.multiFigure(mic_Signal_Cells, recordingInformation)

    # Create Storage Array for Maximum lengths of each audio file #
    maxSignalArrLengths = np.zeros(MIC_COUNT, dtype=int)

    # Loop over each microphone and append samples to start of array for system phasing #
    for mic in FORLOOPARR:
        if (mic != maxIndex):

            mic_Signal_Cells[mic] = np.append(np.zeros(int(Mic_Sample_Delay[mic])), mic_Signal_Cells[mic])
        
            maxSignalArrLengths[mic] = len(mic_Signal_Cells[mic])

    maxSize = np.max(maxSignalArrLengths)

    #  Reallign all arrays to have equivalent sizes #
    for mic in FORLOOPARR:
        #print("Pre-zero : " + str(len(mic_Signal_Cells[mic])) + " for Mic #" + mic)
        mic_Signal_Cells[mic] = np.append(mic_Signal_Cells[mic], np.zeros(int(maxSize - len(mic_Signal_Cells[mic]))))
        #print("Post-zero : " + str(len(mic_Signal_Cells[mic])) + " for Mic #" + mic)

    # Sum all audio signals into one large array #
    micSumSignal = np.zeros(int(maxSize))
    for finalSummationMic in FORLOOPARR:
        micSumSignal = micSumSignal + mic_Signal_Cells[int(finalSummationMic)]

    # Create Summation Figure and Overlapping Figure #
    fc.summationFigure(micSumSignal, recordingInformation)
    fc.overlappingFigure(micSumSignal, mic_Signal_Cells[minIndex], recordingInformation)

    # Create Mic Signal Folder and Store Converted Summation CSV to Wav # 
    folderPath = "../MICRECORD/" + CODE + "/SUM/FinalAudio_" + CODE + ".wav"
    con.convertCSVWav(folderPath, micSumSignal, FS)

if __name__=='__main__':
    main()