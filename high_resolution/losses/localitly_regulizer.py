import torch
import numpy as np
import wandb
from losses import l2_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Space_Regulizer:
    def __init__(self, config, original_G, lpips_net):
        self.original_G = original_G
        self.morphing_regulizer_alpha = 0.5
        self.lpips_loss = lpips_net
        self.config = config

    def get_morphed_w_code(self, new_w_code, fixed_w):

        return (1 - self.morphing_regulizer_alpha) * fixed_w + self.morphing_regulizer_alpha * new_w_code

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch):
        loss = 0.0

        with torch.no_grad():
            z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
            w_samples = self.original_G.mapping(torch.from_numpy(z_samples).to(device), None,
                                                truncation_psi=0.5)
            territory_indicator_ws = [self.get_morphed_w_code(new_G, w_code.unsqueeze(0), w_batch) for w_code in
                                      w_samples]

        for w_code in territory_indicator_ws:
            new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)

            with torch.no_grad():
                old_img = self.original_G.synthesis(w_code, noise_mode='none', force_fp32=True)

            if self.config.tuneG['regulizer_l2_lambda'] > 0:
                l2_loss_val = l2_loss.l2_loss(old_img, new_img)
                loss += l2_loss_val * self.config.tuneG['regulizer_l2_lambda']

            if self.config.tuneG['regulizer_lpips_lambda'] > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * self.config.tuneG['regulizer_lpips_lambda']

        return loss / len(territory_indicator_ws)

    def space_regulizer_loss(self, new_G, w_batch):
        ret_val = self.ball_holder_loss_lazy(new_G, self.config.tuneG['latent_ball_num_of_samples'], w_batch)
        return ret_val
