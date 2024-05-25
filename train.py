import os
import os.path as osp
from argparse import ArgumentParser

from trainer import hr_encoder_trainer, au_encoder_trainer 


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--exp_name', required=True, help='Name of experiment')
    parser.add_argument('--type', '-t', required=True, help='Type of training model')
    
    
    # Resuming Flag
    parser.add_argument('--resume', default=None, help='Path of resuming checkpoint')
    parser.add_argument('--id', default=None, help='Resume id')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    exp_name = args.exp_name
    ckpt_root = '/home/sangwon/voice2face/checkpoint/'
    ckpt_dir = osp.join(ckpt_root, exp_name)
    log_dir = osp.join(ckpt_dir, 'logging')
    os.makedirs(log_dir, exist_ok=True if args.id is not None else False)
    
    if args.type == 'hr_encoder':
        hr_encoder_trainer.train(args, ckpt_dir, log_dir)
    elif args.type == 'au_encoder':
        au_encoder_trainer.train(args, ckpt_dir, log_dir)
    else:
        raise ValueError(f'{args.type} is not supported training type.')