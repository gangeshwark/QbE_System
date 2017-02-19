#!/bin/bash
bnf_query_dir=$1
bnf_database_dir=$2

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/home/tungpv/bnfExtr/query-by-example/'
rm -r $rootQuesst/dist_newquery_*.txt
mkdir -p $rootQuesst/log

./compute-feats-distance_withQueryVAD scp:$bnf_query_dir/raw_bnfea_fbank_pitch.1.scp scp:$bnf_database_dir/raw_bnfea_fbank_pitch.1.scp $rootQuesst/dist_newquery_${j}.JOB.txt ark:traceback_matt.${j}.ark ark,t:/home/tungpv/bnfExtr/vad.txt &


for (( i = 1 ; i <= $numJob ; i++ ))
do
	cat $rootQuesst/dist_newquery_*.${i}.txt > $rootQuesst/dist_newquery.${i}.txt
done

