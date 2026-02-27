#!/bin/bash


#Start with "chmod +x capture.sh" to make the script executable
#Then run the script with "./capture.sh <SAD_X_COORD> <SOD_Y_COORD>"
#Example: "./capture.sh 1.5 2.0"
#This will create a session code "Grid_X1.5_Y2.0" and save the audio to the INDIV directory
#The audio will be saved as "Mic1_Grid_X1.5_Y2.0.wav", "Mic2_Grid_X1.5_Y2.0.wav", etc.
#The audio will be saved in the "../MICRECORD/Grid_X1.5_Y2.0/INDIV" directory
#The audio will be saved in the "../MICRECORD/Grid_X1.5_Y2.0/SUM" directory
#The audio will be saved in the "../MICRECORD/Grid_X1.5_Y2.0/FIGS" directory
# Exit immediately if a command exits with a non-zero status
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: ./capture.sh <SAD_X_COORD> <SOD_Y_COORD>"
    echo "Example: ./capture.sh 1.5 2.0"
    exit 1
fi

SAD_X=$1
SOD_Y=$2
DURATION=15

# Map the coordinates to a session code format your dataset can read
SESSION_CODE="Grid_X${SAD_X}_Y${SOD_Y}"
BASE_DIR="../MICRECORD/${SESSION_CODE}"
INDIV_DIR="${BASE_DIR}/INDIV"

# Create the standard Phased Array Microphonics directory structure
mkdir -p "$INDIV_DIR"
mkdir -p "${BASE_DIR}/SUM"
mkdir -p "${BASE_DIR}/FIGS"

echo "======================================"
echo "Session Code: $SESSION_CODE"
echo "Target Mic Position: SAD ($SAD_X), SOD ($SOD_Y)"
echo "Output Directory: $INDIV_DIR"
echo "======================================"

# 3-second countdown to eliminate handling noise
echo -n "Starting capture in 3... "
sleep 1
echo -n "2... "
sleep 1
echo -n "1... "
sleep 1
echo "RECORDING for $DURATION seconds!"

# -------------------------------------------------------------------------
# HARDWARE CAPTURE
# This is where your wavCollection module takes over. 
# We call a dedicated python collection script that bypasses the 
# system.py beamforming and just records the raw data to the INDIV folder.
# -------------------------------------------------------------------------

python3 collect_dataset_point.py --code "$SESSION_CODE" --duration $DURATION

echo "✅ Capture complete for SAD=$SAD_X, SOD=$SOD_Y."
echo "Move target mic stand to the next grid point!"
echo "======================================"