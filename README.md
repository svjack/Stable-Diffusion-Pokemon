<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Stable-Diffusion-Pokemon</h3>

  <p align="center">
      在宝可梦数据集（Pokemon-Blip-Captions）的英语、日语、中文版本上微调Stable Diffusion的示例
    <br />
  </p>
</p>

[In English](README_EN.md)

### 简要引述
[Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release)是现在一流的文本转图片生成模型。<br/>
现如今，借助于提供多模态能力的[diffusers](https://github.com/huggingface/diffusers)工程，人们可以自定义它们的条件或非条件图像（是否以文本提示作为条件）生成模型。<br/>
这个工程的目标是实现diffuser提供的基于[lambdalabs/pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)的[text to image example](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)微调任务，并将该任务迁移到日文及中文数据集上进行实践。<br/>
通过比较结论将会对Stable Diffusion在不同语言上的微调任务给出指导。
所有的代码都依据官方的[train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)进行修改，并使得其对于中文和日文是有效的。
得到的三种语言的训练模型如下：[English](https://huggingface.co/svjack/Stable-Diffusion-Pokemon-en) , [Japanese](https://huggingface.co/svjack/Stable-Diffusion-Pokemon-ja) 及 [Chinese](https://huggingface.co/svjack/Stable-Diffusion-Pokemon-zh).

### 安装和运行
运行install.sh将会安装所有的依赖并下载所有需要的模型（保证您注册并具有huggingface账号以及您的[token](https://huggingface.co/docs/hub/security-tokens）)
下载后可尝试运行[run_en_model.py](run_en_model.py), [run_ja_model.py](run_ja_model.py) 及 [run_zh_model.py](run_zh_model.py)

### 数据集准备
为了在日文及中文领域进行调试，我们需要将[lambdalabs/pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)。我已经使用 [DeepL](https://www.deepl.com/translator) 对其进行翻译并上传至 huggingface dataset hub。分别位于 [svjack/pokemon-blip-captions-en-ja](https://huggingface.co/datasets/svjack/pokemon-blip-captions-en-ja) 和 [svjack/pokemon-blip-captions-en-zh](https://huggingface.co/datasets/svjack/pokemon-blip-captions-en-zh).


### 微调预训练模型
英文版本是一个仅仅将[train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)的脚本由函数更改为notebook模式，位于[train_en_model.py](train_en_model.py)<br/>

日文版本使用[rinnakk/japanese-stable-diffusion](https://github.com/rinnakk/japanese-stable-diffusion)替换预训练模型，位于[train_ja_model.py](train_ja_model.py)<br/>

中文版本使用[IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese) 替换预训练分词器以及文本编码器，并将BertForTokenClassification的logit输出进行padding来代替CLIPTextModel，位于[train_zh_model.py](train_zh_model.py)。<br/>

为了在推断阶段使得所有结果可见，我关闭了safety_checker。

### 生成器结果比较
<table><caption>Images</caption>
<thead>
<tr>
<th>Prompt</th>
<th colspan="1">English</th>
<th colspan="1">Japanese</th>
<th colspan="1">Chinese</th>
</tr>
</thead>
<tbody>
<tr>
<td>A cartoon character with a potted plant on his head<br/><br/>鉢植えの植物を頭に載せた漫画のキャラクター<br/><br/>一个头上戴着盆栽的卡通人物</td>
<td><img src="imgs/en_plant.jpg" alt="Girl in a jacket" width="550" height="450"></td>
<td><img src="imgs/ja_plant.jpg" alt="Girl in a jacket" width="550" height="450"></td>
<td><img src="imgs/zh_plant.jpg" alt="Girl in a jacket" width="550" height="450"></td>
</tr>
<tr>
<td>cartoon bird<br/><br/>漫画の鳥<br/><br/>卡通鸟</td>
<td><img src="imgs/en_bird.jpg" alt="Girl in a jacket" width="550" height="450"></td>
<td><img src="imgs/ja_bird.jpg" alt="Girl in a jacket" width="550" height="450"></td>
<td><img src="imgs/zh_bird.jpg" alt="Girl in a jacket" width="550" height="450"></td>
</tr>
</tbody>
<tfoot>
<tr>
<td>blue dragon illustration<br/><br/>ブルードラゴンのイラスト<br/><br/>蓝色的龙图</td>
<td><img src="imgs/en_blue_dragon.jpg" alt="Girl in a jacket" width="550" height="450"></td>
<td><img src="imgs/ja_blue_dragon.jpg" alt="Girl in a jacket" width="550" height="450"></td>
<td><img src="imgs/zh_blue_dragon.jpg" alt="Girl in a jacket" width="550" height="450"></td>
</tr>
</tfoot>
</table>

### 讨论
在英文、日文、中文下的预训练模型分别训练了26000, 26000 及 20000步。<br/>
对于训练结果的解释是这样的：[rinnakk/japanese-stable-diffusion](https://github.com/rinnakk/japanese-stable-diffusion)由于是日文的原生模型，所以含有很多宝可梦的特征。[Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release)在英文领域微调的很好。[IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese) 如模型卡片所说在中文上能够起到基本的文本表征作用。

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/Stable-Diffusion-Pokemon](https://github.com/svjack/Stable-Diffusion-Pokemon)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
-->
* [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release)
* [diffusers](https://github.com/huggingface/diffusers)
* [DeepL](https://www.deepl.com/translator)
* [rinnakk/japanese-stable-diffusion](https://github.com/rinnakk/japanese-stable-diffusion)
* [IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese)
* [svjack](https://huggingface.co/svjack)
