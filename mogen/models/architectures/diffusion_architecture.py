from ..builder import ARCHITECTURES, build_loss, build_submodule
from ..utils.gaussian_diffusion import (
    GaussianDiffusion, LossType, ModelMeanType, ModelVarType, SpacedDiffusion,
    create_named_schedule_sampler, get_named_beta_schedule, space_timesteps)
from .base_architecture import BaseArchitecture
import torch
from ..utils.vis import SMPLX_Skeleton
from ..utils.quaternion import ax_from_6v, ax_to_6v
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def build_diffusion(cfg, opt=None):
    beta_scheduler = cfg['beta_scheduler']
    diffusion_steps = cfg['diffusion_steps']

    betas = get_named_beta_schedule(beta_scheduler, diffusion_steps)
    model_mean_type = {
        'start_x': ModelMeanType.START_X,
        'previous_x': ModelMeanType.PREVIOUS_X,
        'epsilon': ModelMeanType.EPSILON
    }[cfg['model_mean_type']]
    model_var_type = {
        'learned': ModelVarType.LEARNED,
        'fixed_small': ModelVarType.FIXED_SMALL,
        'fixed_large': ModelVarType.FIXED_LARGE,
        'learned_range': ModelVarType.LEARNED_RANGE
    }[cfg['model_var_type']]
    if cfg.get('respace', None) is not None:
        diffusion = SpacedDiffusion(use_timesteps=space_timesteps(
            diffusion_steps, cfg['respace']),
                                    betas=betas,
                                    model_mean_type=model_mean_type,
                                    model_var_type=model_var_type,
                                    loss_type=LossType.MSE,
                                    opt=opt)
    else:
        diffusion = GaussianDiffusion(betas=betas,
                                      model_mean_type=model_mean_type,
                                      model_var_type=model_var_type,
                                      loss_type=LossType.MSE)
    return diffusion


@ARCHITECTURES.register_module()
class MotionDiffusion(BaseArchitecture):

    def __init__(self,
                 model=None,
                 loss_recon=None,
                 loss_reduction="frame",
                 diffusion_train=None,
                 diffusion_test=None,
                 sampler_type='uniform',
                 init_cfg=None,
                 inference_type='ddpm',
                 opt=None,
                 hand_loss_factor=1.0,
                 face_no_loss=False,
                 hand_no_loss=False,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.inference_type = inference_type
        self.loss_reduction = loss_reduction
        self.hand_loss_factor = hand_loss_factor
        self.face_no_loss = face_no_loss
        self.hand_no_loss = hand_no_loss
        self.smplx_fk = SMPLX_Skeleton(device="cuda", batch=256*196)
        
        if self.inference_type != 'gt':
            self.model = build_submodule(model)

        self.loss_recon = build_loss(loss_recon)
        self.diffusion_train = build_diffusion(diffusion_train)
        self.diffusion_test = build_diffusion(diffusion_test, opt=opt)
        self.sampler = create_named_schedule_sampler(sampler_type,
                                                     self.diffusion_train)

    def forward(self, **kwargs):
        motion = kwargs['motion'].float()
        motion_mask = kwargs['motion_mask'].float()
        motion_length = kwargs['motion_length']
        num_intervals = kwargs.get('num_intervals', 1)
        sample_idx = kwargs.get('sample_idx', None)
        clip_feat = kwargs.get('clip_feat', None)
        patch_size = kwargs.get('patch_size', 1)
        c = kwargs.get('c', None)
        y = kwargs.get('y', {})
        B, T = motion.shape[:2]
        text = []
        for i in range(B):
            text.append(kwargs['motion_metas'][i]['text'])

        if self.training:
            t, _ = self.sampler.sample(B, motion.device)
            output = self.diffusion_train.training_losses(model=self.model,
                                                          x_start=motion,
                                                          t=t,
                                                          model_kwargs={
                                                              'motion_mask':
                                                              motion_mask,
                                                              'motion_length':
                                                              motion_length,
                                                              'text':
                                                              text,
                                                              'clip_feat':
                                                              clip_feat,
                                                              'sample_idx':
                                                              sample_idx,
                                                              'num_intervals':
                                                              num_intervals,
                                                              'c':
                                                              c,
                                                          })
            pred, target = output['pred'], output['target']
            if self.face_no_loss and pred.shape[-1] == 322:
                face_no_loss_mask = torch.ones_like(pred)
                face_no_loss_mask[:, :, 159:309] = 0
                pred = pred * face_no_loss_mask
                target = target * face_no_loss_mask
            if self.hand_no_loss and pred.shape[-1] == 322:
                hand_no_loss_mask = torch.ones_like(pred)
                hand_no_loss_mask[:, :, 66:66+90] = 0
                pred = pred * hand_no_loss_mask
                target = target * hand_no_loss_mask
            recon_loss = self.loss_recon(pred,
                                         target,
                                         reduction_override='none')

            
            if self.hand_loss_factor > 1.0 and pred.shape[-1] == 322:
                recon_loss[:, :, 66:66+90] = recon_loss[:, :, 66:66+90] * self.hand_loss_factor

            

            recon_loss = recon_loss.mean(dim=-1) * motion_mask
            recon_loss_batch = \
                recon_loss.sum(dim=1) / motion_mask.sum(dim=1)
            recon_loss_frame = \
                recon_loss.sum() / motion_mask.sum()
            if self.loss_reduction == "frame":
                recon_loss = recon_loss_frame
            else:
                recon_loss = recon_loss_batch
            if hasattr(self.sampler, "update_with_local_losses"):
                self.sampler.update_with_local_losses(t, recon_loss_batch)
            loss = {'recon_loss': recon_loss.mean()}
            if hasattr(self.model, 'aux_loss'):
                loss.update(self.model.aux_loss())
            return loss
        else:
            dim_pose = kwargs['motion'].shape[-1]
            if self.inference_type != 'gt':
                model_kwargs = self.model.get_precompute_condition(
                    device=motion.device, text=text, **kwargs)
                model_kwargs['motion_mask'] = motion_mask
                model_kwargs['sample_idx'] = sample_idx
                model_kwargs['motion_length'] = motion_length
                model_kwargs['num_intervals'] = num_intervals
                model_kwargs['c'] = c
                model_kwargs['y'] = y
                model_kwargs['patch_size'] = patch_size
                inference_kwargs = kwargs.get('inference_kwargs', {})

            if self.inference_type == 'ddpm':
                output = self.diffusion_test.p_sample_loop(
                    self.model, (B, T, dim_pose),
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    **inference_kwargs)
            elif self.inference_type == 'ddim':
                output = self.diffusion_test.ddim_sample_loop(
                    self.model, (B, T, dim_pose),
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    eta=0,
                    **inference_kwargs)
            elif self.inference_type == 'gt':
                output = motion

            results = kwargs
            if self.inference_type != 'gt' and getattr(self.model, "post_process") is not None:
                output = self.model.post_process(output)
            
            results['pred_motion'] = output
            
            

            results = self.split_results(results)
            return results
