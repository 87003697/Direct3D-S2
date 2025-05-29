
# Direct3D‑S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention

<div align="center">
  <a href=https://www.neural4d.com/research/direct3d-s2 target="_blank"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome height=22px></a>
  <a href=https://huggingface.co/spaces/wushuang98/Direct3D-S2-v1.0-demo target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href=https://huggingface.co/spaces/wushuang98/Direct3D-S2-v1.0-demo target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://arxiv.org/pdf/2505.17412 target="_blank"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
</div>

<div style="background: #fff; box-shadow: 0 4px 12px rgba(0,0,0,.15); display: inline-block; padding: 0px;">
    <img id="teaser" src="assets/teaserv6.png" alt="Teaser image of Direct3D-S2"/>
</div>

---

## ✨ News

- Feb 11, 2025: 🔨 We are working on the Gradio demo and will release it soon!
- Feb 11, 2025: 🎁 Enjoy our improved version of Direct3D with high quality geometry and texture at [https://www.neural4d.com](https://www.neural4d.com/).
- Feb 11, 2025: 🚀 Release inference code of Direct3D and the pretrained models are available at 🤗 [Hugging Face](https://huggingface.co/DreamTechAI/Direct3D/tree/main).

## 📝 Abstract

Generating high‑resolution 3D shapes using volumetric representations such as Signed Distance Functions (SDFs) presents substantial computational and memory challenges. We introduce <strong class="has-text-weight-bold">Direct3D‑S2</strong>, a scalable 3D generation framework based on sparse volumes that achieves superior output quality with dramatically reduced training costs. Our key innovation is the <strong class="has-text-weight-bold">Spatial Sparse Attention (SSA)</strong> mechanism, which greatly enhances the efficiency of Diffusion Transformer (DiT) computations on sparse volumetric data. SSA allows the model to effectively process large token sets within sparse volumes, yielding a <em>3.9&times;</em> speed‑up in the forward pass and a <em>9.6&times;</em> speed‑up in the backward pass. The framework also includes a variational autoencoder (VAE) that maintains a consistent sparse volumetric format across input, latent, and output stages. Compared with prior 3D VAEs that rely on heterogeneous representations, this unified design markedly improves training efficiency and stability. Trained on publicly available datasets, <strong class="has-text-weight-bold">Direct3D‑S2</strong> not only surpasses state‑of‑the‑art methods in generation quality and efficiency, but also enables <strong class="has-text-weight-bold">training at 1024<sup>3</sup>  resolution with just 8 GPUs</strong>—a task that previously required at least 32 GPUs for 256<sup>3</sup> volumetric training—making gigascale 3D generation both practical and accessible.

## 🚀 Getting Started

### Installation

```sh
git clone https://github.com/DreamTechAI/Direct3D.git

cd Direct3D

pip install -r requirements.txt

pip install -e .
```

### Usage

```python
from direct3d.pipeline import Direct3dPipeline
pipeline = Direct3dPipeline.from_pretrained("DreamTechAI/Direct3D")
pipeline.to("cuda")
mesh = pipeline(
    "assets/devil.png",
    remove_background=False, # set to True if the background of the image needs to be removed
    mc_threshold=-1.0,
    guidance_scale=4.0,
    num_inference_steps=50,
)["meshes"][0]
mesh.export("output.obj")
```

## 🤗 Acknowledgements

Thanks to the following repos for their great work, which helps us a lot in the development of Direct3D:

- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet/tree/master)
- [Michelangelo](https://github.com/NeuralCarver/Michelangelo)
- [Objaverse](https://objaverse.allenai.org/)
- [diffusers](https://github.com/huggingface/diffusers)

## 📖 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{wu2025direct3ds2gigascale3dgeneration,
  title={Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention}, 
  author={Shuang Wu and Youtian Lin and Feihu Zhang and Yifei Zeng and Yikang Yang and Yajie Bao and Jiachen Qian and Siyu Zhu and Philip Torr and Xun Cao and Yao Yao},
  journal={arXiv preprint arXiv:2505.17412},
  year={2025}
}
```

---
