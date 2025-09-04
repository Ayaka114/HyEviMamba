CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 100 --epochs 100 \
--hyper-ad 1 --reduction-ratio 1 --had-feat-dim 256 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper1-256 2>&1 | tee ./logs/AS/ASHyper1-256.log

CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 100 --epochs 100 \
--hyper-ad 1 --reduction-ratio 2 --had-feat-dim 256 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper2-256 2>&1 | tee ./logs/AS/ASHyper2-256.log

CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 100 --epochs 100 \
--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 256 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper4-256 2>&1 | tee ./logs/AS/ASHyper4-256.log

CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 100 --epochs 100 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 256 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper8-256 2>&1 | tee ./logs/AS/ASHyper8-256.log