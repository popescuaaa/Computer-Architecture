#!/bin/bash

# U DON`T KNOW WHO IS FROM CEA MAI INFERIOARA RASA? run the command below:
# qstat -q hp-sl.q -u '*' -f

# exemple de rulare
# ./.sh 1 2 1 -> run ./gpu_hashtable 2 1 on queue
# ./.sh 2 2 1 -> run cuda-memcheck ./gpu_hashtable 2 1 on queue
# ./.sh 3 X X -> run python bench.py checker on queue

# rularile sunt printate in consola dar sunt si puse in fisierul hp-sl-runs.log

# inainte de a rula ul stergeti toate fisierele .e si .o ale rularilor anterioare

if [ $# -ne 3 ]
then
	printf "Usage ./.sh <mode 1(run) or 2(run+cuda-memcheck) or 3(run checker)> <test_numKeys> <test_numChunks>"
	exit
fi

test_numKeys=$2
test_numChunks=$3
coada=hp-sl.q

if [ $1 -eq 1 ]
then
	#run only
	run="module load libraries/cuda && make clean && make && ./gpu_hashtable $test_numKeys $test_numChunks"
	qsub -cwd -q $coada -b y ${run}
fi


if [ $1 -eq 2 ]
then
	#run + cuda memcheck
	run="module load libraries/cuda && make clean && make && cuda-memcheck ./gpu_hashtable $test_numKeys $test_numChunks"
	qsub -cwd -q $coada -b y ${run}
fi

if [ $1 -eq 3 ]
then
	#run checker
	run="module load libraries/cuda && make clean && make && python bench.py"
	qsub -cwd -q $coada -b y ${run}
fi

while true
do
	if [ $(qstat | wc -l) -eq 0 ]
	then
		break
	fi
	sleep 3
done

cat *.o*
cat *.e*

cat *.o* >> hp-sl-runs.log
cat *.e* >> hp-sl-runs.log

printf "\n\n ------------------------------------- \n\n" >> hp-sl-runs.log

rm *.e* *.o*

