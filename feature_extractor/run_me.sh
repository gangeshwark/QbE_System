#!/bin/bash
. path.sh || die "path.sh expected";
#. cmd.sh || die "cmd.sh expected";
nnet_dir=hierbn-nd-train80k #bnf extractor directory

########## Features extraction for test audio ##########
#test_audio="fgdrecording1.wav"
test_audio="my4Hellow.wav"
test_datadir=outdir/data/database
nj=1
mkdir -p $test_datadir
python data_preparation.py $test_audio $test_datadir
test_fbank_pitch_feat=outdir/fbank_pitch_database #physical location of fbank feats
test_bnf_feat=outdir/bnf_database #physical location of bnf feats
./feat_extr_test.sh $test_datadir $nj $test_fbank_pitch_feat $test_bnf_feat $nnet_dir false

########## Features extraction for query ##########

query_audio="queryHellow.wav"
query_datadir=outdir/data/query
nj=1

mkdir -p $query_datadir
python data_preparation.py $query_audio $query_datadir
query_fbank_pitch_feat=outdir/fbank_pitch_query #physical location of fbank feats
query_bnf_feat=outdir/bnf_query #physical location of bnf feats
./feat_extr_test.sh $query_datadir $nj $query_fbank_pitch_feat $query_bnf_feat $nnet_dir true

############# Search ##################
#search_dir=outdir/search_results
#mkdir -p $search_dir
#./compute-feats-distance_withQueryVAD scp:$query_bnf_feat/raw_bnfea_fbank_pitch.1.scp scp:$test_bnf_feat/raw_bnfea_fbank_pitch.1.scp $search_dir/dist.txt ark:$search_dir/traceback_matt.ark ark,t:vad.txt