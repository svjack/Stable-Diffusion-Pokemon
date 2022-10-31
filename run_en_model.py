import torch
import pandas as pd

from torch import autocast
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
     beta_schedule="scaled_linear", num_train_timesteps=1000)

#pretrained_model_name_or_path = "en_model_26000"
pretrained_model_name_or_path = "svjack/Stable-Diffusion-Pokemon-en"
pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path,
                                                           scheduler=scheduler, use_auth_token=True)

pipe = pipe.to("cuda")

#### disable safety_checker
pipe.safety_checker = lambda images, clip_input: (images, False)

imgs = pipe("A cartoon character with a potted plant on his head",
                    num_inference_steps = 100
)
imgs.images[0]

imgs = pipe("cartoon bird",
                    num_inference_steps = 100
)
imgs.images[0]

imgs = pipe("yellow ball",
                    num_inference_steps = 100
)
imgs.images[0]

imgs = pipe("blue dragon illustration",
                    num_inference_steps = 100
)
imgs.images[0]

###### person "Zhuge Liang"
###### penis
