#!/bin/bash
. path.sh || die "path.sh expected";
#. cmd.sh || die "cmd.sh expected";
cmd=run.pl
 
# end user-defined variable

. parse_options.sh || die "parse_options.sh expected ";

data_database=$1 # Just contains general files:  wav.scp utt2spk spk2utt segments
numJob=$2
fbank_pitch_feat_dir=$3
bnf_feat_dir=$4
nnet_dir=$5
compute_vad=$6

data_database_fbank_pitch=$data_database/fbank_pitch # data_database plus some file 
data_database_bnf=$data_database/bnf

# Generate fbank features

  mkdir -p $data_database_fbank_pitch
  mkdir -p $fbank_pitch_feat_dir
  cp $data_database/* $data_database_fbank_pitch/
  steps/make_fbank_pitch.sh --fbank-config conf/fbank22.conf --pitch-config conf/pitch.conf --nj $numJob --cmd "$cmd" $data_database_fbank_pitch $fbank_pitch_feat_dir/log $fbank_pitch_feat_dir

  steps/compute_cmvn_stats.sh $data_database_fbank_pitch $fbank_pitch_feat_dir/log $fbank_pitch_feat_dir

if $compute_vad; then
    compute-vad --config=conf/vad.conf scp:$data_database_fbank_pitch/feats.scp ark,t:vad.txt
fi

##compute-vad scp:$data_database_fbank_pitch/feats.scp ark,t:/home/tungpv/bnfExtr/vad.txt
 
# Generate bnf features
  mkdir -p $bnf_feat_dir
  steps/nnet/make_bn_feats.sh --nj $numJob --cmd "$cmd" $data_database_bnf $data_database_fbank_pitch $nnet_dir $bnf_feat_dir/log $bnf_feat_dir




