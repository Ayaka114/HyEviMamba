import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('MedMamba training and evaluation script', add_help=False)
    
    # 基础配置
    parser.add_argument('--save-name', default="Classifier", type=str,help='Name of the model to save')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    # 模型配置
    parser.add_argument('--model', default='medmamba_t', type=str, metavar='MODEL',
                        help='Name of model to train: medmamba_t, medmamba_s, medmamba_b')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--d-state', type=int, default=16,
                        help='State dimension for selective scan (default: 16)')

    # MedMamba特定配置
    parser.add_argument('--patch-size', type=int, default=4,
                        help='Patch size for image embedding')
    parser.add_argument('--in-chans', type=int, default=3,
                        help='Input image channels')
    parser.add_argument('--num-classes', type=int, default=6,
                        help='Number of classes for classification head')
    
    # Tiny版本配置
    parser.add_argument('--depths-t', type=list, default=[2, 2, 4, 2],
                        help='Depth of each stage for tiny version')
    parser.add_argument('--dims-t', type=list, default=[96, 192, 384, 768],
                        help='Dimensions of each stage for tiny version')
    
    # Small版本配置
    parser.add_argument('--depths-s', type=list, default=[2, 2, 8, 2],
                        help='Depth of each stage for small version')
    parser.add_argument('--dims-s', type=list, default=[96, 192, 384, 768],
                        help='Dimensions of each stage for small version')
    
    # Base版本配置
    parser.add_argument('--depths-b', type=list, default=[2, 2, 12, 2],
                        help='Depth of each stage for base version')
    parser.add_argument('--dims-b', type=list, default=[128, 256, 512, 1024],
                        help='Dimensions of each stage for base version')

    # HyperAD配置
    parser.add_argument('--hyper-ad', default=0, type=int,
                        help='Whether to use HyperAD')
    parser.add_argument('--reduction-ratio', type=int, default=4,
                        help='Reduction ratio for HyperAD feature extraction')

    # EDL配置
    parser.add_argument('--EDL', default=0, type=int,
                        help='Whether to use EDL')
    parser.add_argument('--kl-coef', default=1e-2, type=float)
    parser.add_argument('--adaptive', default=True, type=bool)
    parser.add_argument('--c', default=1.2, type=float)

    # 优化器配置
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    
    # 其他训练配置
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default=None, type=str)

    return parser

def get_args():
    parser = argparse.ArgumentParser('MedMamba training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    return args 