#!/bin/bash
python3 run_campaign_sbseg.py 
mkdir saidas
cp -r ./outputs/out*/reduced_balanced_androcrawl/*/combination_1/EvaluationResults/  ./saidas
