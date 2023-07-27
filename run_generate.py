import argparse
import yaml
import sys
import os
import numpy as np
import torch as th
import torchvision.utils as tvu

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
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.eval()

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
        )
    
    num_img = 0
    while num_img <= args.total_num_imgs:
        img = sampler.sample(S=args.sample_speed,
                            batch_size=args.batch_size,
                            method=args.method,
                            shape=(config['Dataset']['channels'],
                                    config['Dataset']['image_size'],
                                    config['Dataset']['image_size']),
                            )

        for i in range(img.shape[0]):
            tvu.save_image(img[i],f"ddim_imgs/50_steps_scale_step_0.0025/{num_img+i}.png")
        
        num_img += img.shape[0]
    