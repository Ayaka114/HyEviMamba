#CUDA_VISIABLE_DEVICES=0 python evaluate.py --batch_size 128 \
#  --dims 64 128 256 512 --pretrained_weights /home/ywj/MedMamba/MedMamba_ywj/models/mambaDIM64Net.pth

CUDA_VISIBLE_DEVICES=0 python evaluate.py --batch_size 80 --epochs 100 \
--depths 2 2 12 2 --dims 128 256 512 1024 \
--pretrained_weights ./models/mamba_bNet.pth

CUDA_VISIBLE_DEVICES=0 python evaluate.py --batch_size 128 --epochs 100 \
--depths 2 2 8 2 --dims 96 192 384 768 \
--pretrained_weights ./models/mamba_sNet.pth

CUDA_VISIBLE_DEVICES=1 python evaluate.py --batch_size 128 --epochs 100 \
--depths 2 2 4 2 --dims 96 192 384 768 \
--pretrained_weights ./models/mamba_tNet.pth


CUDA_VISIBLE_DEVICES=1 python evaluate.py --batch_size 128 --epochs 100 \
--depths 2 2 4 2 --dims 96 192 384 768 --auto_augment True \
--pretrained_weights ./models/mamba_xNet.pth