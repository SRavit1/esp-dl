#!/bin/bash
FILE="./main/data.h"
N=100

rm $FILE
touch $FILE

printf "#define N %d\n\n" $N >> $FILE

printf "#define ARR {" >> $FILE
for VAR in $(seq 1 $N) #{1..$N}
do
	printf "%d, " $[RANDOM % 256] >> $FILE
done
printf "}" >> $FILE
