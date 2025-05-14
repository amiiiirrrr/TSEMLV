#!/bin/bash

# Define the folder containing the videos
FOLDER="output_video"

# Loop over all MP4 files in the folder
for FILE in "$FOLDER"/*.mp4; do
    # Get the base name of the file (without the extension)
    BASENAME=$(basename "$FILE" .mp4)
    
    # Define the output file name with '_converted' suffix
    OUTPUT="$FOLDER/${BASENAME}_converted.mp4"
    
    # Apply the ffmpeg command
    ffmpeg -i "$FILE" -vcodec libx264 -crf 28 -preset slow "$OUTPUT"
done