#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-02-27 2019-12-22)

train_cmd=
extract_cmd=
stage=0
exp_name=$1
epoch="9"

nnet_dir=exp/$exp_name/far_epoch_${epoch}
# . ./cmd.sh
. ./subtools/path.sh
set -e
. ./subtools/parse_options.sh


if [[ ${stage} -le 0 ]]; then
    echo "$exp_name"

  # Compute the mean vector for centering the evaluation xvectors.
#  ${train_cmd} ${nnet_dir}/voxceleb1o2_train_aug/log/compute_mean.log \
    ivector-mean scp:${nnet_dir}/voxceleb2_train_aug/xvector.scp \
    ${nnet_dir}/voxceleb2_train_aug/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=128
#  ${train_cmd} ${nnet_dir}/voxceleb1o2_train_aug/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=${lda_dim} \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/voxceleb2_train_aug/xvector.scp ark:- |" \
    ark:data/mfcc_23_pitch/voxceleb2_train_aug/utt2spk ${nnet_dir}/voxceleb2_train_aug/transform.mat || exit 1;

  # Train the PLDA model.
#  ${train_cmd} ${nnet_dir}/voxceleb1o2_train_aug/log/plda.log \
    ivector-compute-plda ark:data/mfcc_23_pitch/voxceleb2_train_aug/spk2utt \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/voxceleb2_train_aug/xvector.scp ark:- | transform-vec ${nnet_dir}/voxceleb2_train_aug/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${nnet_dir}/voxceleb2_train_aug/plda || exit 1;

fi

if [[ ${stage} -le 1 ]]; then
  mkdir -p ${nnet_dir}/scores
  for name in sitw_dev sitw_eval; do
#    ${train_cmd} ${nnet_dir}/scores/log/${name}_scoring.log \
      ivector-mean ark:data/mfcc_23_pitch/${name}_enroll/spk2utt scp:${nnet_dir}/${name}_enroll/xvector.scp \
      ark,scp:${nnet_dir}/${name}_enroll/spk_xvector.ark,${nnet_dir}/${name}_enroll/spk_xvector.scp ark,t:${nnet_dir}/${name}_enroll/num_utts.ark || exit 1;

      
      ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:${nnet_dir}/${name}_enroll/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/voxceleb2_train_aug/plda - |" \
      "ark:ivector-mean ark:data/mfcc_23_pitch/${name}_enroll/spk2utt scp:${nnet_dir}/${name}_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/voxceleb2_train_aug/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/voxceleb2_train_aug/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-subtract-global-mean ${nnet_dir}/voxceleb2_train_aug/mean.vec scp:${nnet_dir}/${name}_test/xvector.scp ark:- | transform-vec ${nnet_dir}/voxceleb2_train_aug/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat 'data/mfcc_23_pitch/${name}_test/trials_set/core-core.lst' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/${name}_scores || exit 1;

      echo "SITW Core: $name"
      eer=$(paste data/mfcc_23_pitch/${name}_test/trials_set/core-core.lst ${nnet_dir}/scores/${name}_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
      mindcf1=`subtools/recipe/serialized-experiments/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores/${name}_scores data/mfcc_23_pitch/${name}_test/trials_set/core-core.lst 2> /dev/null`
      mindcf2=`subtools/recipe/serialized-experiments/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/${name}_scores data/mfcc_23_pitch/${name}_test/trials_set/core-core.lst 2> /dev/null`
      
      echo "EER: $eer%"
      echo "minDCF(p-target=0.01): $mindcf1"
      echo "minDCF(p-target=0.001): $mindcf2"
  done

fi


if [[ ${stage} -le 2 ]]; then
  mkdir -p ${nnet_dir}/scores
  for name in voxceleb1_E voxceleb1_H; do

    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/voxceleb2_train_aug/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/voxceleb2_train_aug/mean.vec scp:$nnet_dir/${name}/xvector.scp ark:- | transform-vec $nnet_dir/voxceleb2_train_aug/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/voxceleb2_train_aug/mean.vec scp:$nnet_dir/${name}/xvector.scp ark:- | transform-vec $nnet_dir/voxceleb2_train_aug/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat 'data/mfcc_23_pitch/${name}_clean/trials' | cut -d\  --fields=1,2 |" $nnet_dir/scores/${name}_scores || exit 1;

      echo "voxceleb: $name"
      eer=`compute-eer <(subtools/recipe/serialized-experiments/prepare_for_eer.py data/mfcc_23_pitch/${name}_clean/trials ${nnet_dir}/scores/${name}_scores) 2> /dev/null`
      mindcf1=`subtools/recipe/serialized-experiments/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores/${name}_scores data/mfcc_23_pitch/${name}_clean/trials 2> /dev/null`
      mindcf2=`subtools/recipe/serialized-experiments/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/${name}_scores data/mfcc_23_pitch/${name}_clean/trials 2> /dev/null`
      
      echo "EER: $eer%"
      echo "minDCF(p-target=0.01): $mindcf1"
      echo "minDCF(p-target=0.001): $mindcf2"
  done

fi
