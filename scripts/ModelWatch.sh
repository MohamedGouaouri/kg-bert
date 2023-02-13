#!/bin/bash


path="~/Documents/kg-bert/nohup.out"
while true
do
  # Read the last 100 characters from the file
  chunk=$(tail -c 100 /home/pfe-1301/Documents/kg-bert/nohup.out)
  chunk=$(echo $chunk | sed 's/\r//g');
  chunk=$(echo $chunk | sed 's/"/'"'"'/g');
  #chunk=$(echo $chunk | python -c "import json, sys; print(json.dumps(sys.stdin.read()))");

  #echo $chunk;

    curl "https://logs.logdna.com/logs/ingest?hostname=pfe1301&ip=10.0.4.240" -u 6ed1ff34a51b07ef784710499eac5c72: -H "Content-Type: text/plain; charset=UTF-8" -d "$chunk" > /dev/null 2>&1;

done
