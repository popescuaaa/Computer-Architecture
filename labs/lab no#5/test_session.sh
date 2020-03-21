#!/bin/bash


echo "Creating logile for the current session"

touch logfile

echo "Logfile created"
echo "========================================="

echo "Started testing and performance evaluation"
for script in basic loop_reorder cache_optimisation
do
   echo $script
done
