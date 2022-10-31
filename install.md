pip install git+https://github.com/rinnakk/japanese-stable-diffusion
####pip install diffusers
huggingface-cli login

sudo apt-get install git-lfs
git clone https://huggingface.co/rinna/japanese-stable-diffusion

#### from text_to_image_train.py requirements: https://github.com/huggingface/diffusers/tree/main/examples/text_to_image
git clone https://github.com/huggingface/diffusers
cd examples/text_to_image
pip install -r requirements.txt

diffusers==0.4.1
accelerate
torchvision
transformers>=4.21.0
ftfy
tensorboard
modelcards
jieba
pandas
datasets
