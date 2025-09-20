#!/bin/bash
#set -euo pipefail

#MODEL_NAMES=(ASHyper1-128Net.pth  ASHyper8-96Net.pth   hyper64_EDLNet.pth  mamba_sNet.pth
#ASHyper1-256Net.pth  Classifier_hyper_EDLNet.pth       hyper64Net.pth
#ASHyper1-64Net.pth   Classifier_hyperNet.pth           mamba_tNet.pth
#ASHyper1-96Net.pth   Classifier_mambaNet.pth           mamba_xNet.pth
#ASHyper2-128Net.pth  ClassifierNet.pth
#ASHyper2-256Net.pth
#ASHyper2-64Net.pth
#ASHyper2-96Net.pth
#ASHyper4-128Net.pth      main1Net.pth
#ASHyper4-256Net.pth      main2Net.pth
#ASHyper4-64Net.pth       mamba_bNet.pth
#ASHyper4-96Net.pth           mambaDEP1Net.pth
#ASHyper8-128Net.pth          hyper128_EDLNet.pth                    mambaDIM64Net.pth
#ASHyper8-256Net.pth          hyper_1Net.pth                         mambaPS28Net.pth
#ASHyper8-64Net.pth           hyper_2Net.pth                         mambaPS8Net.pth
#)

EPOCHS=100
BS=64

MODEL_NAMES=(
ASHyper8-128Net.pth
ASHyper8-256Net.pth
ASHyper8-64Net.pth
ASHyper8-96Net.pth
)
rr=(8)
had=(64 96 128 256)

# 外层循环遍历模型名称
for NAME in "${MODEL_NAMES[@]}"; do
  # 中层循环遍历 reduction_ratio
  for r in "${rr[@]}"; do
    # 内层循环遍历 had_feat_dim
    for h in "${had[@]}"; do
      echo ">>> Running ${r} -- ${h} -- ${NAME}"
      CUDA_VISIBLE_DEVICES=0 python evaluate.py \
      --EDL 0 \
      --hyper_ad 1 \
      --reduction_ratio ${r} --had_feat_dim ${h} \
      --epochs ${EPOCHS} \
      --batch_size ${BS} \
      --save_name ${NAME} \
      --pretrained_weights ./models/${NAME}
    done
  done
done