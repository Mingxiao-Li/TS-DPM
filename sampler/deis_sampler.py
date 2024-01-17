import torch as th 
import numpy as np 
import jax.numpy as jnp
from sampler import deis 


class DEIS_Sampler:

    def __init__(self,
                 model,
                 device,
                 diffusion_sampling_steps,
                 beta_start,
                 beta_end,
                 schedule_type,
                 num_steps,
                 method='t_ab',
                 shift_time_step=False,
                 window_size = None,
                 cut_off_value = None,):
        # t2alpha_fn, alphat_fn = deis.get_linear_alpha_fns(beta_0=beta_start, beta_1=beta_end) 
        self.device=device 
        self._make_schedule(
            type=schedule_type,
            diffusion_step=diffusion_sampling_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        
        self.shift_time_step = shift_time_step
        self.window_size = window_size
        self.cut_off_value = cut_off_value

        vpsde = deis.DiscreteVPSDE(self.alphas_cump)
        self.eps_model = model 
        self.eps_model.to(self.device)


        def eps_fn(x_t, scalar_t, cut_off_value, window_size, shift_time_step):
            if shift_time_step:
                vec_t = self.apply_time_shift(x_t, scalar_t.int().item())
            else:
                vec_t = th.ones(x_t.shape[0]).to(x_t.device) * scalar_t
            return self.eps_model(x_t, vec_t)

        self.sampler = deis.get_sampler(
            vpsde,
            eps_fn,
            ts_phase="t",
            ts_order=2.0,
            num_step=num_steps,
            method=method,
            ab_order=2,
        )
    

    def apply_time_shift(self,img_list,t_next):
        x_pre = img_list #[-1]
        n = x_pre.shape[0]
        var = th.var(x_pre.view(x_pre.size()[0],-1),dim=-1)
        var.reshape(-1,1)
        if t_next - self.window_size > 0 and t_next+self.window_size+1<len(self.alpha_list):
            time_list = self.alpha_list[t_next-self.window_size:t_next+self.window_size+1]
        elif t_next-self.window_size <= 0:
            time_list = self.alpha_list[0:t_next+self.window_size+1]
        elif t_next+self.window_size+1 >= len(self.alpha_list):
            time_list = self.alpha_list[t_next-self.window_size:]

        time_list = th.tensor([time_list]*var.size()[0])
        var = var.unsqueeze(1).expand_as(time_list)
        
        dist = (var - time_list.to(x_pre.device)) ** 2
        next_t = th.argmin(dist,dim=1)
        if t_next - self.window_size > 0:
            n_t = next_t + t_next-self.window_size
        else:
            n_t = next_t 
        
        if t_next > self.cut_off_value:
            next_t = th.tensor(n_t).to(x_pre.device)
        else:
            next_t = (th.ones(n) * (t_next)).to(x_pre.device)
        th.cuda.empty_cache()
        return next_t

    
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

    def inverse_img_transform(self, X):
        # if hasattr(config, "image_mean"):
        #     X = X + config.image_mean.to(X.device)[None, ...]
        X = (X + 1.0) / 2.0

        return th.clamp(X, 0.0, 1.0) 

    @th.no_grad()
    def sample(self,
                S=None,
                x_T=None,
                batch_size=None,
                shape=None,
                method=None,
                ):

        if x_T is None:
            shape = (batch_size, ) + shape 
            x_T =th.randn(shape, device=self.device)
        img = x_T.to(self.device)
        img  = self.sampler(
            img,
            cut_off_value = self.cut_off_value,
            window_size = self.window_size,
            shift_time_step = self.shift_time_step
          )
        img = self.inverse_img_transform(img)
        return img 
