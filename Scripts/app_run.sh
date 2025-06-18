!/bin/bash

echo "=============================================================="
echo "Running app with parameters: $*"
echo "=============================================================="
#USER_ID=$1
#shift
cd /MalDataGen/
python3 run_campaign_sbseg.py -c sf
#chown -R $USER_ID shared 
