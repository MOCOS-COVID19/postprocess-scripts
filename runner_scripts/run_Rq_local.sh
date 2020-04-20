#!/bin/bash
step_q=0.05
step_R=0.025
range_q=20
range_R=40
path_to_script=$1
path_to_json_template=$2
individuals_path=$3
households_path=$4
for (( i=0; i<=$range_q; i++ ))
do
	for (( j=0; j<=$range_R; j++ ))
	do
		q=`awk "BEGIN {print $step_q*$i}"`
		R=`awk "BEGIN {print $step_R*$j}"`
		echo "python $path_to_script --params-path $path_to_json_template --df-individuals-path $individuals_path --df-households-path $households_path run-simulation --detection-mild-proba $R --fear-factors-constant-limit-value $q"
	done
done
