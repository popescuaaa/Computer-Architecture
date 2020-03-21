#!/bin/bash


echo "Creating logile for the current session"

touch logfile

echo "Logfile created"
echo "Current testing session" >> logfile
echo "Evaluating the performance for basic, loop_reorder and cache_optimisation" >> logfile

echo "========================================="

echo "Started testing and performance evaluation"
for script in basic loop_reorder cache_optimisation
do
   ./$script >> logfile
done
