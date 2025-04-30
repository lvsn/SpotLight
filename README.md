<div align="center">
<h1>SpotLight: Shadow-Guided Object Relighting via Diffusion</h1>

<a href="https://arxiv.org/pdf/2411.18665" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-PDF-blue" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2411.18665"><img src="https://img.shields.io/badge/arXiv-2411.18665-b31b1b" alt="arXiv"></a>
<a href="https://lvsn.github.io/spotlight"><img src="https://img.shields.io/badge/Project_page-purple" alt="Project Page"></a>

</div>

## Overview

SpotLight allows precise local lighting control by specifying the desired shadows of the inserted object. This approach accurately reshades the object and properly harmonizes the object with the target background, without any training.

In this repository, we provide the code for SpotLight applied to two _diffusion renderers_, that is, diffusion models trained to render realistic images conditioned on 2D intrinsic maps (e.g. normals, albedo, shading, etc.) :
- ZeroComp (Zhang et al., 2025). See the code in `src/zerocomp`.
- RGB↔X (Zeng et al., 2024). See the code in `src/rgbx`.

## Pre-requisites

- Blender 4.0 or 4.1: Make sure you have Blender installed on your machine and in the PATH of your machine, in order to be able to generate shadows. You can check by running `blender --version`. Right now, Blender 4.2 is not supported, since there were many changes to the EEVEE rendering engine.
- Python (tested on >= 3.10)
- NVIDIA GPU

## Quick Start

First, clone the repository, all its submodules and load all the data.

```bash
git clone --recursive git@github.com:lvsn/SpotLight.git
cd SpotLight
```

Then, you can install the dependencies using and run the project in a virtual environment using this:

```bash
python -m venv venv # create virtual environment
# activate on Windows:
venv\Scripts\activate
# activate on Linux/Mac:
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu126
pip install -r src/zerocomp/requirements.txt


python src/spotlight/estimate_background_meshes.py
python src/spotlight/main_generate_shadows.py # optional, you can use pre-computed shadows

python src/zerocomp/main_run_spotlight.py
```

Alternatively, if you want to run the project inside a Docker container, you can do so using the following command.

```bash
docker run -it --gpus all --mount type=bind,src=.,dst=/app ubuntu bash
```

## Acknowledgements

This research was supported by NSERC grants RGPIN 2020-04799 and ALLRP 586543-23, Mitacs and Depix. Computing resources were provided by the Digital Research Alliance of Canada. The authors thank Zheng Zeng, Yannick Hold-Geoffroy and Justine Giroux for their help as well as all members of the lab for discussions and proofreading help.

## Notes about non working versions

For some reason, there are NaN outputs in the conv_in module when running pytorch with torch-2.6.0+cu126 and torch-2.7.0+cu128.
With torch 2.7-cu118, the conv_in is fine, but the vae decoder has trouble

## Citing SpotLight

If you use this code, please cite SpotLight:
```bibtex
@misc{fortierchouinard2025spotlightshadowguidedobjectrelighting,
      title={SpotLight: Shadow-Guided Object Relighting via Diffusion}, 
      author={Frédéric Fortier-Chouinard and Zitian Zhang and Louis-Etienne Messier and Mathieu Garon and Anand Bhattad and Jean-François Lalonde},
      year={2025},
      eprint={2411.18665},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.18665}, 
}
```
And also cite the relevant diffusion renderers. For ZeroComp, cite:
```bibtex
@InProceedings{zhang2025zerocomp,
  author    = {Zhang, Zitian and Fortier-Chouinard, Fr\'ed\'eric and Garon, Mathieu and Bhattad, Anand and Lalonde, Jean-Fran\c{c}ois},
  title     = {ZeroComp: Zero-Shot Object Compositing from Image Intrinsics via Diffusion},
  booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
  month     = {February},
  year      = {2025},
  pages     = {483-494}
}
```
For RGB↔X, cite:
```bibtex
@inproceedings{10.1145/3641519.3657445,
author = {Zeng, Zheng and Deschaintre, Valentin and Georgiev, Iliyan and Hold-Geoffroy, Yannick and Hu, Yiwei and Luan, Fujun and Yan, Ling-Qi and Ha\v{s}an, Milo\v{s}},
title = {RGB↔X: Image decomposition and synthesis using material- and lighting-aware diffusion models},
year = {2024},
isbn = {9798400705250},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3641519.3657445},
doi = {10.1145/3641519.3657445},
abstract = {The three areas of realistic forward rendering, per-pixel inverse rendering, and generative image synthesis may seem like separate and unrelated sub-fields of graphics and vision. However, recent work has demonstrated improved estimation of per-pixel intrinsic channels (albedo, roughness, metallicity) based on a diffusion architecture; we call this the RGB → X problem. We further show that the reverse problem of synthesizing realistic images given intrinsic channels, X → RGB, can also be addressed in a diffusion framework. Focusing on the image domain of interior scenes, we introduce an improved diffusion model for RGB → X, which also estimates lighting, as well as the first diffusion X → RGB model capable of synthesizing realistic images from (full or partial) intrinsic channels. Our X → RGB model explores a middle ground between traditional rendering and generative models: We can specify only certain appearance properties that should be followed, and give freedom to the model to hallucinate a plausible version of the rest. This flexibility allows using a mix of heterogeneous training datasets that differ in the available channels. We use multiple existing datasets and extend them with our own synthetic and real data, resulting in a model capable of extracting scene properties better than previous work and of generating highly realistic images of interior scenes.},
booktitle = {ACM SIGGRAPH 2024 Conference Papers},
articleno = {75},
numpages = {11},
keywords = {Diffusion models, intrinsic decomposition, realistic rendering},
location = {Denver, CO, USA},
series = {SIGGRAPH '24}
}
```

## License
The codes, pretrained weights and test dataset are all for non-commercial use only.

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title"
        rel="cc:attributionURL" href="https://lvsn.github.io/spotlight/">ZeroComp: Zero-shot Object Compositing from
        Image Intrinsics via Diffusion</a> by
    <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://lefreud.github.io/">Frédéric Fortier-Chouinard</a>, Zitian Zhang, Louis-Etienne Messier, Mathieu Garon, Anand Bhattad,
    Jean-François Lalonde is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1"
        target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img
            style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
            src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img
            style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
            src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img
            style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
            src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img
            style="height:22px!important;margin-left:3px;vertical-align:text-bottom;"
            src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a>
</p>