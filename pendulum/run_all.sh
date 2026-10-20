#!/bin/bash
# Train all models: at most 6 parallel processes x 1 thread.
cd "$(dirname "$0")"
mkdir -p logs
cat ${1:-jobs.txt} | xargs -P ${2:-6} -L 1 bash -c 'python train.py "$0" "$1" $2 > "logs/${0}_${2:-x}_s${1}.log" 2>&1'
