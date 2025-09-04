CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
--save-name mambatest0 2>&1 | tee ./logs/mamba/mambatest0.log
#--hyper-ad 0 --reduction-ratio 1 --had-feat-dim 96 \
#--EDL 0 --kl-coef 5e-3 \

CUDA_VISIBLE_DEVICES=1 python MD_train.py --batch-size 128 --epochs 100 \
--patch-size 28 \
--save-name mambaPS28 2>&1 | tee ./logs/mamba/mambaPS28.log

CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--dims 64 128 256 512 \
--save-name mambaDIM64 2>&1 | tee ./logs/mamba/mambaDIM64.log

CUDA_VISIBLE_DEVICES=4 python MD_train.py --batch-size 128 --epochs 100 \
--depths 1 1 2 1 \
--save-name mambaDEP1 2>&1 | tee ./logs/mamba/mambaDEP1.log

CUDA_VISIBLE_DEVICES=1 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 2 --had-feat-dim 64 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper2-64 2>&1 | tee ./logs/AS/ASHyper2-64.log

CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 64 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper4-64 2>&1 | tee ./logs/AS/ASHyper4-64.log

CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 64 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper8-64 2>&1 | tee ./logs/AS/ASHyper8-64.log


CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 1 --had-feat-dim 96 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper1-96 2>&1 | tee ./logs/AS/ASHyper1-96.log

CUDA_VISIBLE_DEVICES=1 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 2 --had-feat-dim 96 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper2-96 2>&1 | tee ./logs/AS/ASHyper2-96.log

CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 96 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper4-96 2>&1 | tee ./logs/AS/ASHyper4-96.log

CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 96 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper8-96 2>&1 | tee ./logs/AS/ASHyper8-96.log



CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 1 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper1-128 2>&1 | tee ./logs/AS/ASHyper1-128.log

CUDA_VISIBLE_DEVICES=1 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 2 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper2-128 2>&1 | tee ./logs/AS/ASHyper2-128.log

CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper4-128 2>&1 | tee ./logs/AS/ASHyper4-128.log

CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper8-128 2>&1 | tee ./logs/AS/ASHyper8-128.log




CUDA_VISIBLE_DEVICES=0 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 1 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper1-128 2>&1 | tee ./logs/AS/ASHyper1-128.log

CUDA_VISIBLE_DEVICES=1 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 2 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper2-128 2>&1 | tee ./logs/AS/ASHyper2-128.log

CUDA_VISIBLE_DEVICES=2 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 4 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper4-128 2>&1 | tee ./logs/AS/ASHyper4-128.log

CUDA_VISIBLE_DEVICES=3 python MD_train.py --batch-size 128 --epochs 100 \
--hyper-ad 1 --reduction-ratio 8 --had-feat-dim 128 \
--EDL 0 --kl-coef 5e-3 \
--save-name ASHyper8-128 2>&1 | tee ./logs/AS/ASHyper8-128.log