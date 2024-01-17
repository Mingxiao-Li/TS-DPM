"""SAMPLING ONLY."""

import torch as th
import numpy as np 
from sampler.dpm_solver_utils import NoiseScheduleVP, model_wrapper, DPM_Solver


class DPMSolverSampler(object):
    def __init__(self, 
        model, 
        diffusion_sampling_steps,
        beta_start,
        beta_end,
        device,
        schedule_type,
        model_type='noise',
        schedule='discrete',
        shift_time_step = False,
        window_size = None,
        cut_off_value = None,
        **kwargs):
        super().__init__()
        
        self.device = device 
        self.model = model
        self.model.to(device)
        
        if shift_time_step:
            print('Sampling with time shift <--->')
            self.shift_time_step=shift_time_step
        else:
            self.shift_time_step = False 
        if window_size is not None:
            self.window_shift = window_size // 2 
        else:
            self.window_shift = None
        self.cut_off_value = cut_off_value
        
        self._make_schedule(
            type=schedule_type,
            diffusion_step=diffusion_sampling_steps,
            beta_start=beta_start,
            beta_end=beta_end
        )

        to_torch = lambda x: x.clone().detach().to(th.float32).to(self.device)
        self.noise_schedule = NoiseScheduleVP(schedule=schedule, betas=to_torch(self.betas))
        
    

    def _make_schedule(self, type, diffusion_step, beta_start, beta_end):
        def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
            betas = []
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            return np.array(betas)
        
        if type == "quad":
            betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, diffusion_step, dtype=np.float64) ** 2)
        elif type == "linear":
            betas = np.linspace(beta_start, beta_end, diffusion_step, dtype=np.float64)
        elif type == "cosine":
            betas = betas_for_alpha_bar(diffusion_step, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2 )
        else:
            betas = None 
        
        betas= th.from_numpy(betas).float()
        alphas = 1.0 - betas 
        alphas_cump = alphas.cumprod(dim=0)
        
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cump = alphas_cump.to(self.device)
        self.alpha_list = [1-x for x in self.alphas_cump]
        self.alpha_list = [a.view(-1) for a in self.alpha_list]


    def register_buffer(self, name, attr):
        if type(attr) == th.Tensor:
            if attr.device != th.device("cuda"):
                attr = attr.to(th.device("cuda"))
        setattr(self, name, attr)

    @th.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,

               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        if x_T is None:
            img = th.randn(size, device=self.device)
        else:
            img = x_T

        #ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            self.model,
            self.noise_schedule,
            model_type="noise",
            guidance_type="uncond",
            # condition=conditioning,
            # unconditional_condition=unconditional_conditioning,
            # guidance_scale=unconditional_guidance_scale,
        )

        dpm_solver = DPM_Solver(model_fn, self.noise_schedule,  predict_x0=False, 
                                window_shift=self.window_shift,cut_off_value=self.cut_off_value,
                                apply_time_shift=self.shift_time_step)
        x = dpm_solver.sample(img, steps=S, skip_type="time_uniform", method='multistep', order=2, lower_order_final=True)
        x = self.inverse_img_transform(x)
        return x
    

    def inverse_img_transform(self, X):
        # if hasattr(config, "image_mean"):
        #     X = X + config.image_mean.to(X.device)[None, ...]
        X = (X + 1.0) / 2.0

        return th.clamp(X, 0.0, 1.0) 