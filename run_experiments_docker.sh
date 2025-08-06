#!/bin/bash
if docker info >/dev/null 2>&1; then
    DIR=$(readlink -f Scripts)
    
    # Ensure scripts have proper permissions on host
    chmod +x $DIR/app_run.sh
    if command -v dos2unix >/dev/null; then
        dos2unix $DIR/app_run.sh
    fi
    
    if [ -z "$(docker images -q sf25/maldatagen:latest 2> /dev/null)" ]; then
        docker build -t sf25/maldatagen:latest .
    fi
    
    # Check if output file is provided as first argument
    if [ -z "$1" ]; then
        # No output file specified - use interactive mode
        docker run -it --name=MalDataGen-$RANDOM \
            -v $DIR:/MalDataGen/Scripts \
            -e DISPLAY=unix$DISPLAY \
            sf25/maldatagen:latest \
            python3 run_campaign_sbseg.py
    else
        # Output file specified - redirect output
        OUTPUT_FILE="$1"
        docker run --name=MalDataGen-$RANDOM \
            -v $DIR:/MalDataGen/Scripts \
            -e DISPLAY=unix$DISPLAY \
            sf25/maldatagen:latest \
            python3 run_campaign_sbseg.py > "$OUTPUT_FILE" 2>&1
    fi
    
else
    echo "Docker permission error. Run this command and restart your machine:"
    echo "sudo usermod -aG docker $USER"
fi
