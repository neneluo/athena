#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 wav.scp fbank.conf corpus_name"
  exit 1
fi

cmd=utils/run.pl
nj=20
scp=$1
fbank_config=$2
name=$3

fbankdir=fbank/$name
#fbankdir=`pwd`/fbank/$name
logdir=`pwd`/log/$name
mkdir -p $fbankdir $logdir
compress=false

if [ -f $fbankdir/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for n in $(seq $nj); do
      split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $fbankdir/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_fbank_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
    compute-fbank-feats --verbose=2 --config=$fbank_config ark:- ark:- \| \
    copy-feats --compress=$compress ark:- \
    ark,scp:$fbankdir/raw_fbank_$name.JOB.ark,$fbankdir/raw_fbank_$name.JOB.scp \
    || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;

  # add ,p to the input rspecifier so that we can just skip over
  # utterances that have bad wave data.

  $cmd JOB=1:$nj $logdir/make_fbank_${name}.JOB.log \
    compute-fbank-feats --verbose=2 --config=$fbank_config \
     scp,p:$logdir/wav_${name}.JOB.scp ark:- \| \
    apply-cmvn-sliding --cmn-window=300 ark:- \
    ark,scp:$fbankdir/fbank_cmvn_sliding_$name.JOB.ark,$fbankdir/fbank_cmvn_sliding_$name.JOB.scp \
      || exit 1;

fi
