import argparse

import torch

from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (add_dict_to_argparser, args_to_dict,
                                          classifier_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)


class DiffAug:

    def __init__(self, gpu_id=0, low=0, high=1000):
        defaults = dict()
        defaults.update(classifier_and_diffusion_defaults())
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        MODEL_FLAGS = "--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
        args = parser.parse_args(MODEL_FLAGS.split())

        score_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args,
                           model_and_diffusion_defaults().keys()))
        self.schedule_sampler = create_named_schedule_sampler(
            "uniform", self.diffusion)

        score_model = score_model.cuda(gpu_id)
        self.gpu_id = gpu_id
        score_model.load_state_dict(
            torch.load('workdirs/256x256_diffusion_uncond.pt',
                       map_location='cpu'))
        score_model.convert_to_fp16()

        self.score_model = (score_model).eval()
        self.low = low
        self.high = high
        print("loaded score-model")

    def augment(self, batch):
        diffusion = self.diffusion

        t = torch.randint(low=self.low,
                          high=self.high,
                          size=(batch.shape[0], )).to(batch).long()
        z = torch.randn_like(batch)
        xt = diffusion.q_sample(x_start=batch, t=t, noise=z)
        with torch.autocast("cuda"):
            xstart = diffusion.p_sample(self.score_model,
                                        xt,
                                        t,
                                        clip_denoised=True)['pred_xstart']

        return xstart

    def test_time_augment(self, batch, t):
        diffusion = self.diffusion

        t = torch.ones(batch.shape[0], device=batch.device).long() * t
        z = torch.randn_like(batch)
        xt = diffusion.q_sample(x_start=batch, t=t, noise=z)
        with torch.autocast("cuda"):
            xstart = diffusion.p_sample(self.score_model,
                                        xt,
                                        t,
                                        clip_denoised=True)['pred_xstart']

        return xstart
