import numpy as np 
import math 
import torch as th
from tqdm import tqdm 
from scipy import integrate

class PFlow:

    def __init__(self,
                 model, 
                 diffusion_sampling_steps,
                 beta_start,
                 beta_end,
                 schedule_type,
                 device):
        
        self.device = device
        self.model = model
        self.model.to(device)
        self.total_step = diffusion_sampling_steps

        self.beta_0 = beta_start
        self.beta_1 = beta_end
        
        self._make_schedule(
            type=schedule_type,
            diffusion_step=diffusion_sampling_steps,
            beta_start=beta_start,
            beta_end=beta_end
        )

    
    def _make_time_steps(self, discr_method, num_sampling_steps,verbose=True):
        if discr_method == "uniform":
            skip = self.total_step // num_sampling_steps
            timesteps = np.asarray(list(range(0, self.total_step, skip)))
        elif discr_method == "quad":
            timesteps = ((np.linspace(0, np.sqrt(num_sampling_steps * .8), num_sampling_steps)) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no discretization method called "{discr_method}"')
        
        steps_out = timesteps + 1
        if verbose:
            print(f"Selected timsteps for sampler : {steps_out}")
        return steps_out[:-1]
    

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
    

    def denoising(self,
                  img,
                  t,
                  t_next,
                  ):
        
        n = img.shape[0]
        beta_0, beta_1 = self.betas[0], self.betas[-1]

        t_start = th.ones(n, device=img.device) * t
        beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * self.total_step
        log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * self.total_step
        std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

        # drift, diffusion -> f(x,t), g(t)
        drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
        score = - self.model(img, t_start * (self.total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
        drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

        return drift
    
    def sample(self, 
               S,
               x_T=None,
               batch_size=None,
               shape=None,
               time_seq=None,
               method=None,
               discr_method = "uniform"):

        if time_seq is None:
            time_seq = self._make_time_steps(discr_method=discr_method, num_sampling_steps=S)
        time_seq_next = [0] + list(time_seq[:-1])

        if x_T is None:
            shape = (batch_size, )+shape
            x_T = th.randn(shape, device=self.device)

        device = self.device 
        tol = 1e-5 if S > 1 else S 
        with th.no_grad():
            def drift_func(t, x):
                x = th.from_numpy(x.reshape(shape)).to(device).type(th.float32)
                drift = self.denoising(x,t,None)
                drift = drift.cpu().numpy().reshape((-1,))
                return drift 
            
            solution = integrate.solve_ivp(drift_func, (1, 1e-3), x_T.cpu().numpy().reshape((-1,)),
                                        rtol=tol,atol=tol, method="RK45")
            img = th.tensor(solution.y[:, -1]).reshape(shape).type(th.float32)

        return img 
        


    

