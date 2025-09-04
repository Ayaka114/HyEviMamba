CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 1 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper1-128 2>&1 | tee ./logs/AS/ASHyper1-128.log

CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 2 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper2-128 2>&1 | tee ./logs/AS/ASHyper2-128.log

CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper4-128 2>&1 | tee ./logs/AS/ASHyper4-128.log

CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper8-128 2>&1 | tee ./logs/AS/ASHyper8-128.log