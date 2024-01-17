import numpy as np 
import math 
import torch as th
from tqdm import tqdm 

class PNMSolver:
    # includes ddim (1-order)

    def __init__(self, 
                 model, 
                 diffusion_sampling_steps,
                 beta_start,
                 beta_end,
                 schedule_type,
                 device,
                 shift_time_step=False,
                 window_size = None,
                 cut_off_value = None,
                 scale_method=False,
                 step_size = None,
                 fix_scale = None,
                 normalize_variance=False,
                 eta=0,
                ):
        
        self.device = device
        self.model = model
        self.model.to(device)
        self.normalize_variance = normalize_variance

        if shift_time_step:
            print("Sampling with time shift <--->")
        self.shift_time_step = shift_time_step
        if window_size is not None:
            self.window_shift = window_size // 2
        self.cut_off_value = cut_off_value
        self.eta = eta 

        if not scale_method:
            self.scale=[1.00 for i in range(diffusion_sampling_steps)]
        elif step_size is not None:
            print("Sampling with linear scale <-->")
            self.scale = [1.00+step_size*i for i in range(diffusion_sampling_steps)]
        elif fix_scale is not None:
            print("Sampling with fix scale <-->")
            self.scale = [fix_scale for i in range(diffusion_sampling_steps)]
    
        self.total_sampling_steps = diffusion_sampling_steps
        self.ets = []
        self._make_schedule(
            type=schedule_type,
            diffusion_step=diffusion_sampling_steps,
            beta_start=beta_start,
            beta_end=beta_end
        )

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


    def _make_time_steps(self, discr_method, num_sampling_steps,verbose=True):
        if discr_method == "uniform":
            skip = self.total_sampling_steps // num_sampling_steps
            timesteps = np.asarray(list(range(0, self.total_sampling_steps, skip)))
        elif discr_method == "quad":
            timesteps = ((np.linspace(0, np.sqrt(num_sampling_steps * .8), num_sampling_steps)) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no discretization method called "{discr_method}"')
        
        steps_out = timesteps + 1
        if verbose:
            print(f"Selected timsteps for sampler : {steps_out}")
        return steps_out[:-1]
    
    def apply_time_shift(self,img_list,t_next):
        x_pre = img_list[-1]
        n = x_pre.shape[0]
        var = th.var(x_pre.view(x_pre.size()[0],-1),dim=-1)
        var.reshape(-1,1)
        if t_next - self.window_shift > 0 and t_next+self.window_shift+1<len(self.alpha_list):
            time_list = self.alpha_list[t_next-self.window_shift:t_next+self.window_shift+1]
        elif t_next-self.window_shift <= 0:
            time_list = self.alpha_list[0:t_next+self.window_shift+1]
        elif t_next+self.window_shift+1 >= len(self.alpha_bar):
            time_list = self.alpha_list[t_next-self.window_shift:]

        time_list = th.tensor([time_list]*var.size()[0])
        var = var.unsqueeze(1).expand_as(time_list)
        
        dist = (var - time_list.to(x_pre.device)) ** 2
        next_t = th.argmin(dist,dim=1)
        if t_next - self.window_shift > 0:
            n_t = next_t + t_next-self.window_shift
        else:
            n_t = next_t 
        
        if t_next > self.cut_off_value:
            next_t = th.tensor(n_t).to(x_pre.device)
        else:
            next_t = (th.ones(n) * (t_next)).to(x_pre.device)
        th.cuda.empty_cache()
        return next_t



    def sample(self, 
               S = None,
               x_T=None,
               batch_size=None,
               shape=None,
               method='euler',
               discr_method = "uniform",
               time_seq = None):
        self.step = 0
        self.ets = []
        if time_seq is None:
            time_seq = self._make_time_steps(discr_method=discr_method, num_sampling_steps=S)
        time_seq_next = [0] + list(time_seq[:-1])
        if x_T is None:
            shape = (batch_size, )+shape
            x_T = th.randn(shape, device=self.device)
        
        img = x_T.to(self.device)

        img_list = [img]
        for i, j in tqdm(zip(reversed(time_seq), reversed(time_seq_next)),total=len(time_seq_next)):
            t = (th.ones(batch_size) * i).to(self.device)
        
            if not self.shift_time_step:
                t_next = (th.ones(batch_size) * j).to(self.device)
            else:
                if len(img_list) > 1:
                    t_next = self.apply_time_shift(img_list, j)
                else:
                    t_next = (th.ones(batch_size) * j).to(self.device)
            with th.no_grad():
                img = self.denoising(img, t, t_next, method=method)
                img_list.append(img)

        img = self.inverse_img_transform(img)
        return img 

    def denoising(self, x, t, t_next, method):

        if self.normalize_variance:
            batch_size = x.shape[0]
            x_ = x.view(batch_size, -1)
            var = th.var(x_, dim=-1)
            op_var = 1-self.alphas_cump[t.long()].view(-1)
            ratio = op_var / var 
            ratio = ratio.view(-1,1,1,1).expand_as(x)
            x = x * th.sqrt(ratio) 

        if method == "euler":
            x_next = self.euler_solver(x, t, t_next)
        elif method == 'improved_euler_solver':
            x_next = self.improved_euler_solver(x, t, t_next)
        elif method == 'runge_kutta':
            x_next = self.runge_kutta(x, t, t_next,)
        elif method == 's_pndm':
            x_next = self.s_pndm(x, t, t_next)
        elif method == 'f_pndm':
            x_next = self.f_pndm(x, t, t_next)
        elif method == 'ddpm':
            x_next = self.ddpm(x, t, t_next)
        else:
            raise ValueError(f" {method} is not supported !!")
        return x_next

    
    def ddpm(self, x, t, t_next):
        at = self.alphas_cump[t.long()].view(-1, 1, 1, 1)
        at_next = self.alphas_cump[t_next.long()].view(-1, 1, 1, 1)

        et = self.model(x, t)
        et = et / self.scale[self.step]
        self.step += 1

        x0_t = (x - et * (1-at).sqrt()) / at.sqrt()

        c1 = self.eta * ((1-at/at_next)*(1-at_next) / (1-at)).sqrt()
        c2 = ((1-at_next) - c1 ** 2).sqrt()

        xt_next = at_next.sqrt() * x0_t + c1 * th.randn_like(x) + c2 * et 
        return xt_next


    def transfer(self,x, t, t_next, et):
        at = self.alphas_cump[t.long()].view(-1, 1, 1, 1)
        at_next = self.alphas_cump[t_next.long()].view(-1, 1, 1, 1)

        x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                    1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)
        
        x_next = x + x_delta 
        return x_next


    # different solver 
    def euler_solver(self, x, t, t_next):
        # 1st order solver 
        # the same as ddim  (eular method)
        et = self.model(x, t)
        et = et / self.scale[self.step]
        self.step += 1 
        #self.ets.append(et)

        x_next = self.transfer(x, t, t_next, et)
        return x_next
    

    def improved_euler_solver(self, x, t, t_next,return_noise=False):
        # 2nd order solver  classical method
        e_1 = self.model(x, t)
        self.ets.append(e_1)
        x_2 = self.transfer(x, t, t_next, e_1)

        e_2 = self.model(x_2, t_next)
        et = (e_1 + e_2) / 2 
        x_next = self.transfer(x, t, t_next, et)
        
        if return_noise:
            return et 
        return x_next
    

    def s_pndm(self,x, t, t_next):
        # second-order pseudo numerical method
        # improved euler_solver + multi_linear step solver (2 steps)
        if len(self.ets) > 0:
            et = self.model(x, t)
            if len(self.ets) < 2:
                self.ets.append(et)
            else:
                self.ets[0] = self.ets[-1]
                self.ets[-1] = et  
            et = 0.5 * ( 3 * self.ets[-1] - self.ets[-2])
        else:
            et = self.improved_euler_solver(x, t, t_next, return_noise=True)
        # et = et / self.step_scale[self.step]
        # self.step += 1 
        #et += th.randn(et.shape,device=self.device)
        x_next = self.transfer(x, t, t_next, et)
        return x_next
    

    def f_pndm(self, x, t, t_next):
        # fourth-order pseudo numerical method
        # runge_kutta + multi_linear step solver (4 steps)
        if len(self.ets) > 2:
            et = self.model(x, t)
            if len(self.ets) < 4:
                self.ets.append(et)
            else:
                self.ets[0], self.ets[1], self.ets[2] = self.ets[1], self.ets[2], self.ets[3]
                self.ets[3] = et
            et = ( 1 / 24 ) * ( 55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])
        else:
            et = self.runge_kutta(x, t, t_next, return_noise=True)
        #et += th.randn(et.shape,device=self.device)
        x_next = self.transfer(x, t, t_next, et)
        return x_next


    def runge_kutta(self, x, t, t_next, return_noise=False):
        t_list = [t, (t + t_next) / 2.0, t_next]
        e_1 = self.model(x, t_list[0])
        self.ets.append(e_1)
        x_2 = self.transfer(x, t_list[0], t_list[1], e_1) 

        e_2 = self.model(x_2, t_list[1])
        x_3 = self.transfer(x, t_list[0], t_list[1], e_2)

        e_3 = self.model(x_3, t_list[1])
        x_4 = self.transfer(x, t_list[0], t_list[2], e_3)

        e_4 = self.model(x_4, t_list[2])
        et = (1 / 6) * ( e_1 + 2 * e_2 + 2 * e_3 + e_4)
        x_next = self.transfer(x, t_list[0], t_list[-1], et)
        
        if return_noise:
            return et 
        else:
            return x_next
    

    def classical_fourth_order_solver(self, x, t, t_next):
        t_list = [t, (t + t_next) / 2.0, t_next]

        if len(self.ets) > 2:
            et = self.model(x, t)
            x_next = self.transfer(x, t, t-1, et)
            delta1 = x_next - x 
            self.ets.append(delta1)
            delta = ( 1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])
        else:
            et = self.model(x, t_list[0])
            x_ = self.transfer(x, t, t-1, et)
            deltal_1 =x_ - x 

            x_2 = x + deltal_1 * (t - t_next).view(-1, 1, 1, 1) / 2.0
            et = self.model(x_2, t_list[1])
            x_ = self.transfer(x, t, t-1, et)
            deltal_2 = x_ - x 

            x_3 = x + deltal_2 * (t - t_next).view(-1, 1, 1, 1) / 2.0
            et = self.model(x_3, t_list[1])
            x_ = self.transfer(x, t, t-1, et)
            deltal_3 = x_ - x 

            x_4 = x + deltal_3 * (t - t_next).view(-1, 1, 1, 1)
            et = self.model(x_4, t_list[2])
            x_ = self.transfer(x, t, t-1, et)
            deltal_4 = x_ - x 
            delta = (1 / 6.0) * (deltal_1 + 2 * deltal_2 + 2 * deltal_3 + deltal_4)
        x_next = x + delta * (t-t_next).view(-1, 1, 1, 1)
        
        return x_next 
 
    def inverse_img_transform(self, X):
        # if hasattr(config, "image_mean"):
        #     X = X + config.image_mean.to(X.device)[None, ...]
        X = (X + 1.0) / 2.0

        return th.clamp(X, 0.0, 1.0) 
 