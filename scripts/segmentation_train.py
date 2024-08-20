"""
Train a diffusion model on images.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from guided_diffusion import dist_util, logger
from guided_diffusion.dataset import load_data
from mpi4py import MPI
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.segment_train_util import TrainLoop
from guided_diffusion import pytorch_ssim
import torch
import random
import numpy as np

def create_argparser():
    defaults = dict(
        data_name='Pterygium',
        hyper = 'init',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=2500,
        save_interval=2500,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_ssim = False,
    )
    defaults.update(model_and_diffusion_defaults())  
    parser = argparse.ArgumentParser()            
    add_dict_to_argparser(parser, defaults)       
    return parser

def main():
    seed_torch(seed=816+MPI.COMM_WORLD.Get_rank())
    args = create_argparser().parse_args()           
    dist_util.setup_dist([3,4])
    
    logger.configure()
    logger.log("creating model and diffusion...")
    

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())  
    ssim = pytorch_ssim.SSIM().to(dist_util.dev()) if args.use_ssim else None

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)  

    logger.log("creating data loader...")
    data = load_data(                            
        data_name = args.data_name,                 
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_val_test = 'train'
    )
     
    logger.log("training...")
    TrainLoop(
        model=model,                     
        diffusion=diffusion,                
        data=data,                          
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        ssim=ssim,
        save_path = './Save/'+args.data_name + '/' + args.model_name + '/' + args.hyper  + '/model_save'
    ).run_loop()


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
