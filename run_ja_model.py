from japanese_stable_diffusion import JapaneseStableDiffusionPipeline
import torch
import pandas as pd

from torch import autocast
from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
     beta_schedule="scaled_linear", num_train_timesteps=1000)

#pretrained_model_name_or_path = "jap_model_26000"

#### sudo apt-get install git-lfs
#### git clone https://huggingface.co/svjack/Stable-Diffusion-Pokemon-ja
pretrained_model_name_or_path = "Stable-Diffusion-Pokemon-ja"

pipe = JapaneseStableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path,
                                                           scheduler=scheduler, use_auth_token=True)

pipe = pipe.to("cuda")

#### disable safety_checker
pipe.safety_checker = lambda images, clip_input: (images, False)

imgs = pipe("鉢植えの植物を頭に載せた漫画のキャラクター",
                    num_inference_steps = 100
)
imgs.images[0]

imgs = pipe("漫画の鳥",
                    num_inference_steps = 100
)
imgs.images[0]

imgs = pipe("黄色いボール",
                    num_inference_steps = 100
)
imgs.images[0]

imgs = pipe("ブルードラゴンのイラスト",
                    num_inference_steps = 100
)
imgs.images[0]

##### 緑のピエロ
