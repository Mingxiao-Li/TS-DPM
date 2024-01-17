import argparse
import yaml
import sys
import os
import numpy as np
import torch as th
import torchvision.utils as tvu
import time 
from  torch.nn.modules.upsampling import Upsample
from tqdm import tqdm 

def args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runner", type=str, default='sample',
                        help="Choose the mode of runner")
    parser.add_argument("--config", type=str, default='ddim_cifar10.yml',
                        help="Choose the config file")
    parser.add_argument("--model", type=str, default='DDIM',
                        help="Choose the model's structure (DDIM, iDDPM, PF)")
    parser.add_argument("--method", type=str, default='F-PNDM',
                        help="Choose the numerical methods (DDIM, FON, S-PNDM, F-PNDM, PF)")
    parser.add_argument("--sample_speed", type=int, default=50,
                        help="Control the total generation step")
    parser.add_argument("--image_path", type=str, default='temp/sample',
                        help="Choose the path to save images")
    parser.add_argument("--model_path", type=str, default='temp/models/ddim/ema_cifar10.ckpt',
                        help="Choose the path of model")
    parser.add_argument("--sampler", type=str, default="pnm_solver",
                        help="Choose sampler")
    parser.add_argument("--total_num_imgs", type=int, default=1,
                        help="total number of images to generate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch_size")
    parser.add_argument("--dataset", type=str,default="cifar")
    parser.add_argument("--time_shift", action="store_true", help="if running time-shift") # false if not given
    parser.add_argument("--window_size",type=int, default=None)
    parser.add_argument("--cut_off_value",type=int, default=None)
    
    parser.add_argument("--scale_method",action="store_true",help="if use scale")
    parser.add_argument("--step_size",type=float, default=None)
    parser.add_argument("--fix_scale",type=float, default=None)
    parser.add_argument("--nor_var",action="store_true",help="if normalize variance")
    parser.add_argument("--eta",type=float, default=0.0)
    
    args = parser.parse_args()

    work_dir = os.getcwd()
    with open(f'{work_dir}/config/{args.config}', 'r') as f:
        config = yaml.safe_load(f)

    return args, config


if __name__ == "__main__":
    args, config = args_and_config()
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

    if config['Model']['struc'] == 'DDIM':
        from model.ddim import Model
        model = Model(args, config['Model']).to(device)

    elif config['Model']['struc'] == 'iDDPM':
        from model.iDDPM.unet import UNetModel
        model = UNetModel(args, config['Model']).to(device)
    elif config['Model']['struc'] == 'PF':
        from model.scoresde.ddpm import DDPM
        model = DDPM(args, config['Model']).to(device)
    elif config['Model']['struc'] == 'PF_deep':
        from model.scoresde.ncsnpp import NCSNpp
        model = NCSNpp(args, config['Model']).to(device)
    else:
        model = None
    
    # load weights 
    ckpt = th.load(args.model_path,map_location="cpu")
    if isinstance(ckpt,list):
        model.load_state_dict(ckpt[0])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("Loading model from -->", args.model_path)
    pflow= True if "PF" in config["Model"]["struc"] else False

    # set sampler 
    if args.sampler == "pnm_solver":
        from sampler.pnm_solver import PNMSolver
       
        sampler = PNMSolver(
            model = model,
            device = device,
            diffusion_sampling_steps=config['Schedule']['diffusion_step'],
            beta_start=config['Schedule']['beta_start'],
            beta_end=config['Schedule']['beta_end'],
            schedule_type=config['Schedule']['type'],
            shift_time_step=args.time_shift,
            window_size=args.window_size,
            cut_off_value=args.cut_off_value,
            step_size=args.step_size,
            fix_scale=args.fix_scale,
            normalize_variance=args.nor_var,
            eta=args.eta,
            scale_method=args.scale_method
        )
    elif args.sampler == "dpm-solver":
        from sampler.dpm_solver import DPMSolverSampler
        sampler = DPMSolverSampler(
            model=model, 
            device=device,
            shift_time_step=args.time_shift,
            window_size=args.window_size,
            cut_off_value=args.cut_off_value,
            diffusion_sampling_steps=config["Schedule"]["diffusion_step"],
            beta_start = config["Schedule"]["beta_start"],
            beta_end = config["Schedule"]["beta_end"],
            schedule_type=config["Schedule"]["type"]
        )
    elif args.sampler == "pflow":
        from sampler.pflow import PFlow
        sampler = PFlow(
            model = model,
            device = device,
            diffusion_sampling_steps=config["Schedule"]["diffusion_step"],
            beta_start = config["Schedule"]["beta_start"],
            beta_end = config["Schedule"]["beta_end"],
            schedule_type = config["Schedule"]["type"],
        )
    elif args.sampler == "deis":
        from sampler.deis_sampler import DEIS_Sampler
        sampler = DEIS_Sampler(
            model = model,
            device=device,
            diffusion_sampling_steps=config["Schedule"]["diffusion_step"],
            beta_start = config["Schedule"]["beta_start"],
            beta_end=config["Schedule"]["beta_end"],
            schedule_type=config["Schedule"]["type"],
            num_steps = args.sample_speed,
            shift_time_step=args.time_shift,
            window_size = args.window_size,
            cut_off_value = args.cut_off_value
        )

    
    num_img = 0
    if args.time_shift:
        file_name = f"ts-ddim/{args.dataset}_{args.sampler}_{args.sample_speed}_w_{args.window_size}_c_{args.cut_off_value}_{args.method}"
    elif args.scale_method:
        if args.step_size is not None:
            file_name = f"scale-ddim/{args.dataset}_{args.sampler}_{args.sample_speed}_step_{args.step_size}_ets_{args.eta}"
        elif args.fix_scale is not None:
            file_name = f"scale-ddim/{args.dataset}_{args.sampler}_{args.sample_speed}_fix_scale_{args.fix_scale}_ets_{args.eta}"
    else:
        if args.nor_var:
            file_name = f"time-s-ddim/norm_var_{args.dataset}_{args.sampler}_{args.sample_speed}_{args.method}"
        else:
            file_name = f"dpm-time-ts-order-2/{args.dataset}_{args.sampler}_{args.sample_speed}_{args.method}"
    
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    num_file = len(os.listdir(file_name))
    num_img = num_file
    print("Saving to ",file_name)
    print("Cur num of images --> ",num_img)
    while num_img <= args.total_num_imgs:
        
        for j in range(args.batch_size):
            path = f"{file_name}/{num_img+j}.png"
            if os.path.exists(path):
                num_img += 1
                continue 
        with th.no_grad():
            img = sampler.sample(S=args.sample_speed,
                                batch_size=args.batch_size,
                                method=args.method,
                                # shape=(3,128,128),
                                shape=(config['Dataset']['channels'],
                                        config['Dataset']['image_size'],
                                        config['Dataset']['image_size']),
                                )
           
            for i in range(img.shape[0]):
                tvu.save_image(img[i],f"{file_name}/{num_img+i}.png")
        num_img += img.shape[0]
    