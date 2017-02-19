#!/bin/bash
. path.sh || die "path.sh expected";
#. cmd.sh || die "cmd.sh expected";
nnet_dir=hierbn-nd-train80k #bnf extractor directory

########## Features extraction for audio1 ##########
#test_audio="fgdrecording1.wav"
test_audio="hello1.wav"
test_datadir=outdir/data/hello1
nj=1
mkdir -p $test_datadir
python data_preparation.py $test_audio $test_datadir
test_fbank_pitch_feat=outdir/fbank_pitch_hello1 #physical location of fbank feats
test_bnf_feat=outdir/bnf_hello1 #physical location of bnf feats
./feat_extr_test.sh $test_datadir $nj $test_fbank_pitch_feat $test_bnf_feat $nnet_dir false

########## Features extraction for audio2 ##########
#test_audio="fgdrecording1.wav"
test_audio="hello2.wav"
test_datadir=outdir/data/hello2
nj=1
mkdir -p $test_datadir
python data_preparation.py $test_audio $test_datadir
test_fbank_pitch_feat=outdir/fbank_pitch_hello2 #physical location of fbank feats
test_bnf_feat=outdir/bnf_hello2 #physical location of bnf feats
./feat_extr_test.sh $test_datadir $nj $test_fbank_pitch_feat $test_bnf_feat $nnet_dir false

########## Features extraction for audio3 ##########
#test_audio="fgdrecording1.wav"
test_audio="hello3.wav"
test_datadir=outdir/data/hello3
nj=1
mkdir -p $test_datadir
python data_preparation.py $test_audio $test_datadir
test_fbank_pitch_feat=outdir/fbank_pitch_hello3 #physical location of fbank feats
test_bnf_feat=outdir/bnf_hello3 #physical location of bnf feats
./feat_extr_test.sh $test_datadir $nj $test_fbank_pitch_feat $test_bnf_feat $nnet_dir false

########## Features extraction for audio4 ##########
#test_audio="fgdrecording1.wav"
test_audio="hello4.wav"
test_datadir=outdir/data/hello4
nj=1
mkdir -p $test_datadir
python data_preparation.py $test_audio $test_datadir
test_fbank_pitch_feat=outdir/fbank_pitch_hello4 #physical location of fbank feats
test_bnf_feat=outdir/bnf_hello4 #physical location of bnf feats
./feat_extr_test.sh $test_datadir $nj $test_fbank_pitch_feat $test_bnf_feat $nnet_dir false

########## Features extraction for audio5 ##########
#test_audio="fgdrecording1.wav"
test_audio="hello5.wav"
test_datadir=outdir/data/hello5
nj=1
mkdir -p $test_datadir
python data_preparation.py $test_audio $test_datadir
test_fbank_pitch_feat=outdir/fbank_pitch_hello5 #physical location of fbank feats
test_bnf_feat=outdir/bnf_hello5 #physical location of bnf feats
./feat_extr_test.sh $test_datadir $nj $test_fbank_pitch_feat $test_bnf_feat $nnet_dir false

########## Features extraction for audio6 ##########
#test_audio="fgdrecording1.wav"
test_audio="hello6.wav"
test_datadir=outdir/data/hello6
nj=1
mkdir -p $test_datadir
python data_preparation.py $test_audio $test_datadir
test_fbank_pitch_feat=outdir/fbank_pitch_hello6 #physical location of fbank feats
test_bnf_feat=outdir/bnf_hello6 #physical location of bnf feats
./feat_extr_test.sh $test_datadir $nj $test_fbank_pitch_feat $test_bnf_feat $nnet_dir false
