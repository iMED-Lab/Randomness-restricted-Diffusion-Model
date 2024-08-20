import argparse
from cmath import e
import os
from re import I, S
import sys, os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch as th
from guided_diffusion.dataset import data_loader
from guided_diffusion import dist_util, logger
import torchvision.transforms as transforms
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from PIL import Image

def create_argparser():
    defaults = dict(
        data_name='Pterygium',
        model_name = 'R2diff',
        model_time = 20000,
        hyper = 'init',
        use_pre_seg = False, #sampling of SCDM
        clip_denoised=True,
        num_samples=100,
        batch_size=8,
        use_ddim=False,   #loss of SCDM
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.3,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=False,
        diffusion_steps=100,
        noise_schedule="linear",
        timestep_respacing="50",
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

args = create_argparser().parse_args()
dist_util.setup_dist([2])

if os.path.exists('./Save/'+args.data_name + '/' + args.model_name + '/' + args.hyper + '/image_save' ) == False:
    os.mkdir('./Save/'+args.data_name + '/' + args.model_name + '/' + args.hyper + '/image_save')
root = './Save/'+args.data_name + '/' + args.model_name + '/' + args.hyper + '/image_save'

logger.configure()
logger.log("creating model and diffusion...")
model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

model_path = './Save/'+args.data_name + '/' + args.model_name + '/' + args.hyper  + '/model_save/model' +f'{args.model_time}'.zfill(6) + '.pt'

model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location=dist_util.dev())
    )
save_path = root + '/model' +f'{args.model_time}'.zfill(6)
if os.path.exists(save_path) == False:
        os.mkdir(save_path)
model.to(dist_util.dev())
if args.use_fp16:
    model.convert_to_fp16()
model.eval()
data = data_loader(batch_size = args.batch_size,image_size = 256,color_adjust=False,random_rotate=False,random_H_flip=False,random_V_flip=False
        ,train_val_test = 'test',data_name = args.data_name)
    
logger.log("sampling...")

for batch in data:

    image,mask,label,name,pre_seg = batch
    sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
    image = image.to(dist_util.dev())
    pre_seg = pre_seg.to(dist_util.dev()) if args.use_pre_seg else None
    step = int(args.timestep_respacing[4:])-1 if args.timestep_respacing.startswith("ddim") else int(args.timestep_respacing)-1
    step = 30 if args.use_pre_seg else step
       
    with th.no_grad():
            sample = sample_fn(
            model,
            (image.shape[0], 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            pre_seg= pre_seg,     #pre_segmentation for SCDM
            model_kwargs=image,
            step = step,
        )
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous().cpu()

    for i in range(sample.shape[0]):
        pred = sample[i].repeat(1,1,3).numpy()
        pred = Image.fromarray(pred)
        pred.save(save_path+'/'+name[i]+'.png')


       



