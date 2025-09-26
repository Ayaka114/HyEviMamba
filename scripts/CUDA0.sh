MODEL=edl_adaptive_init5e-3_c0.8_min_maxNet
TSNE_P=70
CUDA_VISIBLE_DEVICES=1 python tsne_features.py --batch_size 1024 \
  --hyper_ad 0 \
  --EDL 1 --kl_coef 5e-3 \
  --l2norm \
  --pca_dim 50 \
  --tsne_metric cosine \
  --tsne_perplexity ${TSNE_P} \
  --tsne_iter 3000 \
  --tsne_ee 16 \
  --auto_lr \
  --feature proj_fused \
  --max_legend_classes 20 \
  --val_dir ../dataset/val \
  --pretrained_weights ./models/${MODEL}.pth \
  --save_fig tsne_out/tsne_${MODEL}_${TSNE_P}.png --save_csv tsne_out/tsne_${MODEL}_${TSNE_P}.csv

#CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch_size 168 --epochs 100 \
#--depths 2 2 12 2 --dims 128 256 512 1024 \
#--save_name mamba_b 2>&1 | tee ./logs/main/mamba_b.log
#
#CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch_size 256 --epochs 100 \
#--depths 2 2 8 2 --dims 96 192 384 768 \
#--save_name mamba_s 2>&1 | tee ./logs/main/mamba_s.log

#CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
#--hyper-ad 1 --reduction-ratio 1 --had-feat-dim 64 \
#--EDL 0 --kl-coef 5e-3 \
#--save-name ASHyper1-64 2>&1 | tee ./logs/AS/ASHyper1-64.log
#
#CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
#--hyper-ad 1 --reduction-ratio 2 --had-feat-dim 64 \
#--EDL 0 --kl-coef 5e-3 \
#--save-name ASHyper2-64 2>&1 | tee ./logs/AS/ASHyper2-64.log
#
#CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
#--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 64 \
#--EDL 0 --kl-coef 5e-3 \
#--save-name ASHyper4-64 2>&1 | tee ./logs/AS/ASHyper4-64.log
#
#CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
#--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 64 \
#--EDL 0 --kl-coef 5e-3 \
#--save-name ASHyper8-64 2>&1 | tee ./logs/AS/ASHyper8-64.log
