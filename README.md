[![Awesome Logo](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![TechrXiv](https://img.shields.io/badge/T-TechRxiv-blue)](https://www.techrxiv.org) [![Visitors](https://komarev.com/ghpvc/?username=VWMSurvey&label=Welcome!%20&color=orange)]()

[![arXiv](https://img.shields.io/badge/arXiv-26XX.XXXXX-b31b1b?logo=arxiv)](https://arxiv.org/abs/25XX.XXXXX) [![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/papers/)

# :sunglasses: Awesome Vision World Models

<img width="100%" src="files/teaser.png">

## 📖 𝐀 𝐒𝐮𝐫𝐯𝐞𝐲 𝐨𝐧 𝐖𝐨𝐫𝐥𝐝 𝐌𝐨𝐝𝐞𝐥𝐬: 𝐀 𝐕𝐢𝐬𝐢𝐨𝐧 𝐏𝐞𝐫𝐬𝐩𝐞𝐜𝐭𝐢𝐯𝐞

This repository provides a curated list of works on Vision World Models, along with their corresponding **arXiv IDs, GitHub repositories**, and **Project pages**.

For more details, kindly refer to our [paper](./files/temp.pdf) :rocket:


### :books: Citation 

If you find this work helpful for your research, please kindly consider citing our paper:
```bib
WIP
```

### Table of Contents
- [0. Background](#0-background)
- [1. Designs](#1-designs)
  - [1.1. Sequential Generation](#11-sequential-generation)
    - [Autoregressive Transformers](#autoregressive-transformers)
    - [MLLM as Vision World Model Engine](#mllm-as-vision-world-model-engine)
  - [1.2. Diffusion-based Generation](#12-diffusion-based-generation)
    - [Latent Diffusion](#latent-diffusion)
    - [Autoregressive Diffusion](#autoregressive-diffusion)
  - [1.3. Predictive Architecture](#13-predictive-architecture)
    - [JEPA](#jepa)
  - [1.4. State Transition](#14-state-transition)
    - [Latent State-Space Modeling](#latent-state-space-modeling)
    - [Object-Centric Modeling](#object-centric-modeling)
  - [1.5. Other Architectures](#15-other-architectures)
- [2. Datasets \& Benchmarks](#2-datasets--benchmarks)
  - [2.1. General-Purpose World Modeling](#21-general-purpose-world-modeling)
    - [World Prediction and Simulation](#world-prediction-and-simulation)
    - [Physical and Causal Reasoning](#physical-and-causal-reasoning)
  - [2.2. Application-Specific World Modeling](#22-application-specific-world-modeling)
    - [Embodied AI and Robotics](#embodied-ai-and-robotics)
    - [Autonomous Driving](#autonomous-driving)
    - [Interactive Environments and Gaming](#interactive-environments-and-gaming)
- [3. Others](#3-others)
    - [Survey](#survey)
    - [GitHub Repo](#github-repo)
    - [Workshop](#workshop)
    - [Theory](#theory)
    - [World Models for Downstream Tasks](#world-models-for-downstream-tasks)
    - [Other Perspectives of World Modeling](#other-perspectives-of-world-modeling)
- [4. Acknowledgements](#4-acknowledgements)

# 0. Background

## About Vision World Model (VWM)
Definition:

>A vision world model is an AI model that learns to simulate the physical world through visual observation.

Formally, a VWM can be seen as a probabilistic model $f_{\theta}$ that predicts the distribution of future states given observed visual context and interactive conditions:

$$p(\mathcal{R}_{t+1:T}| v_{0:t}, c_{t}) = f_{\theta} (\mathcal{E}(v_{0:t}), c_{t})$$

where $v_{0:t}$ represents the sequence of visual observations from time $0$ to $t$, and $c_{t}$ represents current conditions (e.g., agent actions, language instructions, or control signals). $\mathcal{E}(\cdot)$ denotes the visual encoder that maps raw inputs into tokens or embeddings.
$\mathcal{R}_{t+1:T}$ represents the world representation, which encompasses a broad range of future modalities depending on the paradigm, including future observations ($v_{t+1:T}$), latent states ($s_{t+1:T}$), or other future properties (e.g., segmentation maps, depth, flow, 3D Gaussian splats, or trajectories). 

We further establish a conceptual framework that decomposes VWM into three essential components:

* (1) The Perceptual Foundation: How diverse visual signals are transformed into world representation.

* (2) The Dynamics Core: What "rules of the world" are learned, progressing from spatio-temporal coherence to physical dynamics and causal reasoning.

* (3) The Key Capability: How VWM performs controllable simulation conditioned on actions, language, or other interaction prompts.

<img width="100%" src="files/framework.png">

## Taxonomy of VWM Designs

We provide an in-depth analysis of VWMs' four major architectural families, applying our three-component framework to compare their underlying mechanisms.

<img width="100%" src="files/taxonomy.png">

## Evaluation Ecosystem

We provide an extensive review of the evaluation landscape, cataloging metrics and distinguishing between datasets and benchmarks types.

<img width="100%" src="files/evaluation.png">



# 1. Designs

## 1.1. Sequential Generation

### Autoregressive Transformers

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `iMoWM` | iMoWM: Taming Interactive Multi-Modal World Model for Robotic Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2510.09036-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.09036)[![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://xingyoujun.github.io/imowm/) |
| `PWM` | From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction | [![arXiv](https://img.shields.io/badge/arXiv-2510.19654-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.19654) [![GitHub](https://img.shields.io/github/stars/6550Zhao/Policy-World-Model)](https://github.com/6550Zhao/Policy-World-Model)  |
| `SAMPO` | SAMPO: Scale-wise Autoregression with Motion Prompt for Generative World Models | [![arXiv](https://img.shields.io/badge/arXiv-2509.15536-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.15536) |
| `RynnVLA-001` | RynnVLA-001: Using Human Demonstrations to Improve Robot Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2509.15212-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.15212) [![GitHub](https://img.shields.io/github/stars/alibaba-damo-academy/RynnVLA-001)](https://github.com/alibaba-damo-academy/RynnVLA-001) |
| `OccTENS` | OccTENS: 3D Occupancy World Model via Temporal Next-Scale Prediction | [![arXiv](https://img.shields.io/badge/arXiv-2509.03887-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.03887) |
| `Genie 3` | Genie 3: A new frontier for world models | [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) |
| `I²-world` | I²-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting | [![arXiv](https://img.shields.io/badge/arXiv-2507.09144-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.09144) [![GitHub](https://img.shields.io/github/stars/lzzzzzm/II-World)](https://github.com/lzzzzzm/II-World) |
| `UniVLA` | Unified Vision-Language-Action Model | [![arXiv](https://img.shields.io/badge/arXiv-2506.19850-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.19850) [![GitHub](https://img.shields.io/github/stars/baaivision/UniVLA)](https://github.com/baaivision/UniVLA) |
| `WorldVLA` | WorldVLA: Towards Autoregressive Action World Model | [![arXiv](https://img.shields.io/badge/arXiv-2506.21539-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.21539) [![GitHub](https://img.shields.io/github/stars/alibaba-damo-academy/WorldVLA)](https://github.com/alibaba-damo-academy/WorldVLA) |
| `RoboScape` | RoboScape: Physics-informed Embodied World Model | [![arXiv](https://img.shields.io/badge/arXiv-2506.23135-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.23135) [![GitHub](https://img.shields.io/github/stars/tsinghua-fib-lab/RoboScape)](https://github.com/tsinghua-fib-lab/RoboScape)|
| `Xray2Xray` | Xray2Xray: World Model from Chest X-rays with Volumetric Context | [![arXiv](https://img.shields.io/badge/arXiv-2506.19055-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.19055) |
| `RLVR-World` | RLVR-World: Training World Models with Reinforcement Learning | [![arXiv](https://img.shields.io/badge/arXiv-2505.13934-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.13934) [![GitHub](https://img.shields.io/github/stars/thuml/RLVR-World)](https://github.com/thuml/RLVR-World) |
| `MineWorld` | MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft | [![arXiv](https://img.shields.io/badge/arXiv-2504.08388-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.08388) [![GitHub](https://img.shields.io/github/stars/microsoft/mineworld)](https://github.com/microsoft/mineworld) |
| `UVA` | Unified Video Action Model | [![arXiv](https://img.shields.io/badge/arXiv-2503.00200-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.00200) [![GitHub](https://img.shields.io/github/stars/ShuangLI59/unified_video_action)](https://github.com/ShuangLI59/unified_video_action) |
| `SurgWM` | Surgical Vision World Model | [![arXiv](https://img.shields.io/badge/arXiv-2503.02904-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.02904) [![GitHub](https://img.shields.io/github/stars/bhattarailab/Surgical-Vision-World-Model)](https://github.com/bhattarailab/Surgical-Vision-World-Model)|
| `DWS` | Pre-Trained Video Generative Models as World Simulators | [![arXiv](https://img.shields.io/badge/arXiv-2502.07825-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.07825) |
| `VideoWorld` | VideoWorld: Exploring Knowledge Learning from Unlabeled Videos | [![arXiv](https://img.shields.io/badge/arXiv-2501.09781-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.09781) [![GitHub](https://img.shields.io/github/stars/ByteDance-Seed/VideoWorld)](https://github.com/ByteDance-Seed/VideoWorld) |
| `Drivingworld` | Drivingworld: Constructing world model for autonomous driving via video GPT | [![arXiv](https://img.shields.io/badge/arXiv-2412.19505-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.19505) [![GitHub](https://img.shields.io/github/stars/YvanYin/DrivingWorld)](https://github.com/YvanYin/DrivingWorld) |
| `Moto` | Moto: Latent motion token as the bridging language for learning robot manipulation from video | [![arXiv](https://img.shields.io/badge/arXiv-2412.04445-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.04445) [![GitHub](https://img.shields.io/github/stars/TencentARC/Moto)](https://github.com/TencentARC/Moto) |
| `WHALE` | WHALE: Towards Generalizable and Scalable World Models for Embodied Decision-making | [![arXiv](https://img.shields.io/badge/arXiv-2411.05619-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.05619) |
| `GR-2` | GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2410.06158-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.06158) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://gr2-manipulation.github.io/) |
| `LatentDriver` | Learning Multiple Probabilistic Decisions from Latent World Model in Autonomous Driving | [![arXiv](https://img.shields.io/badge/arXiv-2409.15730-b31b1b?logo=arxiv)](https://arxiv.org/abs/2409.15730) [![GitHub](https://img.shields.io/github/stars/Sephirex-X/LatentDriver)](https://github.com/Sephirex-X/LatentDriver) |
| `Renderworld` | Renderworld: World model with self-supervised 3D label | [![arXiv](https://img.shields.io/badge/arXiv-2409.11356-b31b1b?logo=arxiv)](https://arxiv.org/abs/2409.11356) |
| `VidIT` | Video In-context Learning: Autoregressive Transformers are Zero-Shot Video Imitators | [![arXiv](https://img.shields.io/badge/arXiv-2407.07356-b31b1b?logo=arxiv)](https://arxiv.org/abs/2407.07356) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://www.microsoft.com/en-us/research/project/ar-videos/)|
| `iVideoGPT` | iVideoGPT: Interactive VideoGPTs are Scalable World Models | [![arXiv](https://img.shields.io/badge/arXiv-2405.15223-b31b1b?logo=arxiv)](https://arxiv.org/abs/2405.15223) [![GitHub](https://img.shields.io/github/stars/thuml/iVideoGPT)](https://github.com/thuml/iVideoGPT)|
| `Genie` | Genie: Generative Interactive Environments | [![arXiv](https://img.shields.io/badge/arXiv-2402.15391-b31b1b?logo=arxiv)](https://arxiv.org/abs/2402.15391) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://sites.google.com/view/genie-2024/home)|
| `LWM` | World Model on Million-Length Video And Language With Blockwise RingAttention | [![arXiv](https://img.shields.io/badge/arXiv-2402.08268-b31b1b?logo=arxiv)](https://arxiv.org/abs/2402.08268) [![GitHub](https://img.shields.io/github/stars/LargeWorldModel/LWM)](https://github.com/LargeWorldModel/LWM)|
| `WHAM` | World and human action models towards gameplay ideation | [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://www.nature.com/articles/s41586-025-08600-3)|
| `WorldDreamer` | WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens | [![arXiv](https://img.shields.io/badge/arXiv-2401.09985-b31b1b?logo=arxiv)](https://arxiv.org/abs/2401.09985) [![GitHub](https://img.shields.io/github/stars/JeffWang987/WorldDreamer)](https://github.com/JeffWang987/WorldDreamer) |
| `OccWorld` | OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving | [![arXiv](https://img.shields.io/badge/arXiv-2311.16038-b31b1b?logo=arxiv)](https://arxiv.org/abs/2311.16038) [![GitHub](https://img.shields.io/github/stars/wzzheng/OccWorld)](https://github.com/wzzheng/OccWorld)|
| `GAIA-1` | Gaia-1: A generative world model for autonomous driving | [![arXiv](https://img.shields.io/badge/arXiv-2309.17080-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2309.17080) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://wayve.ai/science/gaia/)|

### MLLM as Vision World Model Engine

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `AstraNav-World` | AstraNav-World: World Model for Foresight Control and Consistency | [![arXiv](https://img.shields.io/badge/arXiv-2512.21714-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.21714)  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://astra-amap.github.io/AstraNav-World.github.io/) [![GitHub](https://img.shields.io/github/stars/amap-cvlab/AstraNav-World)](https://github.com/amap-cvlab/AstraNav-World)|
| `RynnVLA-002` | RynnVLA-002: A Unified Vision-Language-Action and World Model | [![arXiv](https://img.shields.io/badge/arXiv-2511.17502-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.17502)  [![GitHub](https://img.shields.io/github/stars/alibaba-damo-academy/RynnVLA-002)](https://github.com/alibaba-damo-academy/RynnVLA-002)|
| `SWM` | Semantic World Models | [![arXiv](https://img.shields.io/badge/arXiv-2510.19818-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.19818)  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://weirdlabuw.github.io/swm/)|
| `UniWM` | Unified World Models: Memory-Augmented Planning and Foresight for Visual Navigation | [![arXiv](https://img.shields.io/badge/arXiv-2510.08713-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.08713)   [![GitHub](https://img.shields.io/github/stars/F1y1113/UniWM)](https://github.com/F1y1113/UniWM) |
| `F1` | F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions | [![arXiv](https://img.shields.io/badge/arXiv-2509.06951-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.06951) [![GitHub](https://img.shields.io/github/stars/InternRobotics/F1-VLA)](https://github.com/InternRobotics/F1-VLA) |
| `OccVLA` | OccVLA: Vision-Language-Action Model with Implicit 3D Occupancy Supervision | [![arXiv](https://img.shields.io/badge/arXiv-2509.05578-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.05578) |
| `VLWM` | Planning with Reasoning using Vision Language World Model | [![arXiv](https://img.shields.io/badge/arXiv-2509.02722-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.02722)  |
| `DreamVLA` | DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge | [![arXiv](https://img.shields.io/badge/arXiv-2507.04447-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.04447) [![GitHub](https://img.shields.io/github/stars/Zhangwenyao1/DreamVLA)](https://github.com/Zhangwenyao1/DreamVLA) |
| `World4Omni` | World4Omni: A Zero-Shot Framework from Image Generation World Model to Robotic Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2506.23919-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.23919)[![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://nus-lins-lab.github.io/goalvlaweb/) |
| `WALL-E 2.0` | WALL-E 2.0: World Alignment by NeuroSymbolic Learning Improves World Model-based LLM Agent | [![arXiv](https://img.shields.io/badge/arXiv-2504.15785-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.15785)[![GitHub](https://img.shields.io/github/stars/elated-sawyer/WALL-E)](https://github.com/elated-sawyer/WALL-E) |
| `GR00T N1` | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | [![arXiv](https://img.shields.io/badge/arXiv-2503.14734-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.14734) [![GitHub](https://img.shields.io/github/stars/NVIDIA/Isaac-GR00T)](https://github.com/NVIDIA/Isaac-GR00T)|
| `HERMES` | HERMES: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation | [![arXiv](https://img.shields.io/badge/arXiv-2501.14729-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.14729) [![GitHub](https://img.shields.io/github/stars/LMD0311/HERMES)](https://github.com/LMD0311/HERMES) |
| `Doe-1` | Doe-1: Closed-Loop Autonomous Driving with Large World Model | [![arXiv](https://img.shields.io/badge/arXiv-2412.09627-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.09627)[![GitHub](https://img.shields.io/github/stars/wzzheng/Doe)](https://github.com/wzzheng/Doe) |
| `Owl-1` | Owl-1: Omni World Model for Consistent Long Video Generation | [![arXiv](https://img.shields.io/badge/arXiv-2412.09600-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.09600) [![GitHub](https://img.shields.io/github/stars/huang-yh/Owl)](https://github.com/huang-yh/Owl)|
| `Eva` | Eva: An Embodied World Model for Future Video Anticipation | [![arXiv](https://img.shields.io/badge/arXiv-2410.15461-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.15461) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://sites.google.com/view/icml-eva)|
| `PIVOT-R` | PIVOT-R: Primitive-Driven Waypoint-Aware World Model for Robotic Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2410.10394-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.10394) [![GitHub](https://img.shields.io/github/stars/abliao/PIVOT-R)](https://github.com/abliao/PIVOT-R)|
| `OccLLaMA` | OccLLaMA: An Occupancy-Language-Action Generative World Model for Autonomous Driving | [![arXiv](https://img.shields.io/badge/arXiv-2409.03272-b31b1b?logo=arxiv)](https://arxiv.org/abs/2409.03272) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://vilonge.github.io/OccLLaMA_Page/)|
| `WorldGPT` | WorldGPT: Empowering LLM as Multimodal World Model | [![arXiv](https://img.shields.io/badge/arXiv-2404.18202-b31b1b?logo=arxiv)](https://arxiv.org/abs/2404.18202) [![GitHub](https://img.shields.io/github/stars/DCDmllm/WorldGPT)](https://github.com/DCDmllm/WorldGPT)|
| `3D-VLA` | 3D-VLA: A 3D Vision-Language-Action Generative World Model | [![arXiv](https://img.shields.io/badge/arXiv-2403.09631-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.09631)[![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://vis-www.cs.umass.edu/3dvla)|
| `ADriver-I` | ADriver-I: A General World Model for Autonomous Driving | [![arXiv](https://img.shields.io/badge/arXiv-2311.13549-b31b1b?logo=arxiv)](https://arxiv.org/abs/2311.13549) |
||

## 1.2. Diffusion-based Generation

### Latent Diffusion

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `GrndCtrl`  | GrndCtrl: Grounding World Models via Self-Supervised Reward Alignment |  [![arXiv](https://img.shields.io/badge/arXiv-2512.01952-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.01952) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://rlwg-grndctrl.github.io/) |
| `C^3`  | World Models That Know When They Don’t Know: Controllable Video Generation with Calibrated Uncertainty |  [![arXiv](https://img.shields.io/badge/arXiv-2512.05927-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.05927) [![GitHub](https://img.shields.io/github/stars/irom-princeton/c-cubed)](https://github.com/irom-princeton/c-cubed) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://c-cubed-uq.github.io/)|
| `GAIA-3`  | GAIA-3: Scaling World Models to Power Safety and Evaluation |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://wayve.ai/thinking/gaia-3/) |
| `WristWorld`  | WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2510.07313-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.07313) [![GitHub](https://img.shields.io/github/stars/XuWuLingYu/WristWorld)](https://github.com/XuWuLingYu/WristWorld) |
| `WorldSplat`  | WorldSplat: Gaussian-Centric Feed-Forward 4D Scene Generation for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2509.23402-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.23402) [![GitHub](https://img.shields.io/github/stars/wm-research/worldsplat)](https://github.com/wm-research/worldsplat) |
| `WorldForge`  | WorldForge: Unlocking Emergent 3D/4D Generation in Video Diffusion Model via Training-Free Guidance |  [![arXiv](https://img.shields.io/badge/arXiv-2509.15130-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.15130) [![GitHub](https://img.shields.io/github/stars/Westlake-AGI-Lab/WorldForge)](https://github.com/Westlake-AGI-Lab/WorldForge) |
| `PEWM`  | Learning Primitive Embodied World Models: Towards Scalable Robotic Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2508.20840-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.20840) [![GitHub](https://img.shields.io/github/stars/qiaosun22/PrimitiveWorld)](https://github.com/qiaosun22/PrimitiveWorld) |
| `Video Policy`  | Video Generators are Robot Policies |  [![arXiv](https://img.shields.io/badge/arXiv-2508.00795-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.00795) [![GitHub](https://img.shields.io/github/stars/cvlab-columbia/videopolicy)](https://github.com/cvlab-columbia/videopolicy) |
| `Ego-PM`  | Ego-centric Predictive Model Conditioned on Hand Trajectories |  [![arXiv](https://img.shields.io/badge/arXiv-2508.19852-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.19852) [![GitHub](https://img.shields.io/github/stars/binjiezhang/Ego-PM)](https://github.com/binjiezhang/Ego-PM) |
| `GWM`  | GWM: Towards Scalable Gaussian World Models for Robotic Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2508.17600-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.17600) [![GitHub](https://img.shields.io/github/stars/Gaussian-World-Model/gaussianwm)](https://github.com/Gaussian-World-Model/gaussianwm) |
| `M3arsSynth`  | Martian World Models: Controllable Video Synthesis with Physically Accurate 3D Reconstructions |  [![arXiv](https://img.shields.io/badge/arXiv-2507.07978-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.07978) [![GitHub](https://img.shields.io/github/stars/loongfeili/Martian-World-Model)](https://github.com/loongfeili/Martian-World-Model) |
| `AirScape`  | AirScape: An Aerial Generative World Model with Motion Controllability |  [![arXiv](https://img.shields.io/badge/arXiv-2507.08885-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.08885) [![GitHub](https://img.shields.io/github/stars/EmbodiedCity/AirScape.code)](https://github.com/EmbodiedCity/AirScape.code) |
| `robot4dgen`  | Geometry-aware 4D Video Generation for Robot Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2507.01099-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.01099) [![GitHub](https://img.shields.io/github/stars/lzylucy/4dgen)](https://github.com/lzylucy/4dgen) |
| `Genesis`  | Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency |  [![arXiv](https://img.shields.io/badge/arXiv-2506.07497-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.07497) [![GitHub](https://img.shields.io/github/stars/xiaomi-research/genesis)](https://github.com/xiaomi-research/genesis) |
| `MinD`  | MinD: Learning A Dual-System World Model for Real-Time Planning and Implicit Risk Analysis |  [![arXiv](https://img.shields.io/badge/arXiv-2506.18897-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.18897) [![GitHub](https://img.shields.io/github/stars/manipulate-in-dream/MinD)](https://github.com/manipulate-in-dream/MinD) |
| `RealPlay`  | PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2506.18901-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.18901) [![GitHub](https://img.shields.io/github/stars/wenqsun/Real-Play)](https://github.com/wenqsun/Real-Play) |
| `COME`  | COME: Adding Scene-Centric Forecasting Control to Occupancy World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2506.13260-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.13260) [![GitHub](https://img.shields.io/github/stars/synsin0/COME)](https://github.com/synsin0/COME) |
| `MeWM`  | Medical World Model:  Generative Simulation of Tumor Evolution for Treatment Planning |  [![arXiv](https://img.shields.io/badge/arXiv-2506.02327-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.02327) [![GitHub](https://img.shields.io/github/stars/scott-yjyang/MeWM)](https://github.com/scott-yjyang/MeWM) |
| `StateSpaceDiffuser`  | StateSpaceDiffuser: Bringing Long Context to Diffusion World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2505.22246-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.22246) [![GitHub](https://img.shields.io/github/stars/insait-institute/StateSpaceDiffuser)](https://github.com/insait-institute/StateSpaceDiffuser)|
| `GeoDrive`  | GeoDrive: 3D Geometry-Informed Driving World Model with Precise Action Control |  [![arXiv](https://img.shields.io/badge/arXiv-2505.22421-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.22421) [![GitHub](https://img.shields.io/github/stars/antonioo-c/GeoDrive)](https://github.com/antonioo-c/GeoDrive) |
| `3DPEWM`  | Learning 3d persistent embodied world models |  [![arXiv](https://img.shields.io/badge/arXiv-2505.05495-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.05495)|
| `RoboTransfer`  | RoboTransfer: Geometry-Consistent Video Diffusion for Robotic Visual Policy Transfer |  [![arXiv](https://img.shields.io/badge/arXiv-2505.23171-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.23171) [![GitHub](https://img.shields.io/github/stars/HorizonRobotics/RoboTransfer)](https://github.com/HorizonRobotics/RoboTransfer) |
| `FlowDreamer`  | FlowDreamer: A RGB-D World Model with Flow-based Motion Representations  for Robot Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2505.10075-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.10075) [![GitHub](https://img.shields.io/github/stars/sharinka0715/FlowDreamer)](https://github.com/sharinka0715/FlowDreamer) |
| `LaDi-WM`  | LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2505.11528-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.11528) [![GitHub](https://img.shields.io/github/stars/GuHuangAI/LaDiWM)](https://github.com/GuHuangAI/LaDiWM) |
| `LangToMo`  | Pixel motion as universal representation for robot control |  [![arXiv](https://img.shields.io/badge/arXiv-2505.07817-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.07817) [![GitHub](https://img.shields.io/github/stars/kahnchana/LangToMo)](https://github.com/kahnchana/LangToMo) |
| `DreamGen`  | DreamGen: Unlocking Generalization in Robot Learning through Video World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2505.12705-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.12705) [![GitHub](https://img.shields.io/github/stars/nvidia/GR00T-dreams)](https://github.com/nvidia/GR00T-dreams) |
| ` Learning to Drive`  |  Learning to Drive from a World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2504.19077-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.19077) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://blog.comma.ai/mlsim)|
| `Tesseract`  | Tesseract: Learning 4d embodied world models |  [![arXiv](https://img.shields.io/badge/arXiv-2504.20995-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.20995) [![GitHub](https://img.shields.io/github/stars/UMass-Embodied-AGI/TesserAct)](https://github.com/UMass-Embodied-AGI/TesserAct) |
| `UWM`  | Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets |  [![arXiv](https://img.shields.io/badge/arXiv-2504.02792-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.02792) [![GitHub](https://img.shields.io/github/stars/WEIRDLabUW/unified-world-model)](https://github.com/WEIRDLabUW/unified-world-model) |
| `ViMo`  | ViMo: A Generative Visual GUI World Model for App Agents|  [![arXiv](https://img.shields.io/badge/arXiv-2504.13936-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.13936) [![GitHub](https://img.shields.io/github/stars/ai-agents-2030/ViMo)](https://github.com/ai-agents-2030/ViMo) |
| `DiST-4D`  | DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth  for 4D Driving Scene Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2503.15208-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.15208) [![GitHub](https://img.shields.io/github/stars/royalmelon0505/dist4d)](https://github.com/royalmelon0505/dist4d) |
| `Aether`  | Aether: Geometric-aware unified world modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2503.18945-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.18945) [![GitHub](https://img.shields.io/github/stars/InternRobotics/Aether)](https://github.com/InternRobotics/Aether) |
| `Cosmos transfer1`  | Cosmos transfer1: Conditional world generation with adaptive multimodal control |  [![arXiv](https://img.shields.io/badge/arXiv-2503.14492-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.14492) [![GitHub](https://img.shields.io/github/stars/nvidia-cosmos/cosmos-transfer1)](https://github.com/nvidia-cosmos/cosmos-transfer1) |
| `GAIA-2`  | GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2503.20523-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.20523) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://wayve.ai/thinking/gaia-2/) |
| `EDELINE`  | EDELINE: Enhancing Memory in Diffusion-based World Models via Linear-Time Sequence Modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2502.00466-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.00466) [![GitHub](https://img.shields.io/github/stars/LJH-coding/EDELINE)](https://github.com/LJH-coding/EDELINE) |
| `MaskGWM`  | MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction |  [![arXiv](https://img.shields.io/badge/arXiv-2502.11663-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.11663) [![GitHub](https://img.shields.io/github/stars/SenseTime-FVG/OpenDWM)](https://github.com/SenseTime-FVG/OpenDWM) |
| `Dreamdrive`  | Dreamdrive: Generative 4d scene modeling from street view images |  [![arXiv](https://img.shields.io/badge/arXiv-2501.00601-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.00601) [![GitHub](https://img.shields.io/github/stars/NVlabs/DreamDrive)](https://github.com/NVlabs/DreamDrive) |
| `Cosmos`  | Cosmos World Foundation Model Platform for Physical AI |  [![arXiv](https://img.shields.io/badge/arXiv-2501.03575-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.03575) [![GitHub](https://img.shields.io/github/stars/nvidia-cosmos)](https://github.com/nvidia-cosmos) |
| `VPP`  | Video prediction policy: A generalist robot policy with predictive visual representations |  [![arXiv](https://img.shields.io/badge/arXiv-2412.14803-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.14803) [![GitHub](https://img.shields.io/github/stars/roboterax/video-prediction-policy)](https://github.com/roboterax/video-prediction-policy) |
| `Imagine-2-drive`  | Imagine-2-drive: High-fidelity world modeling in carla for autonomous vehicles |  [![arXiv](https://img.shields.io/badge/arXiv-2411.10171-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.10171) [![GitHub](https://img.shields.io/github/stars/anantagrg/Imagine-2-Drive)](https://github.com/anantagrg/Imagine-2-Drive) |
| `GenEx`  | Generative World Explorer |  [![arXiv](https://img.shields.io/badge/arXiv-2411.11844-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.11844) [![GitHub](https://img.shields.io/github/stars/GenEx-world/genex)](https://github.com/GenEx-world/genex) |
| `DOME`  | DOME: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2410.10429-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.10429) [![GitHub](https://img.shields.io/github/stars/gusongen/DOME)](https://github.com/gusongen/DOME) |
| `Panacea+`  | Panacea+: Panoramic and Controllable Video Generation for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2408.07605-b31b1b?logo=arxiv)](https://arxiv.org/abs/2408.07605) [![GitHub](https://img.shields.io/github/stars/wenyuqing/panacea)](https://github.com/wenyuqing/panacea) |
| `Bevworld`  | Bevworld: A multimodal world model for autonomous driving via unified bev latent space |  [![arXiv](https://img.shields.io/badge/arXiv-2407.05679-b31b1b?logo=arxiv)](https://arxiv.org/abs/2407.05679) [![GitHub](https://img.shields.io/github/stars/zympsyche/BevWorld)](https://github.com/zympsyche/BevWorld) |
| `Delphi`  | Unleashing generalization of end-to-end autonomous driving with controllable long video generation |  [![arXiv](https://img.shields.io/badge/arXiv-2406.01349-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.01349) [![GitHub](https://img.shields.io/github/stars/westlake-autolab/Delphi)](https://github.com/westlake-autolab/Delphi) |
| `Occsora`  | Occsora: 4d occupancy generation models as world simulators for autonomous driving |  [![arXiv](https://img.shields.io/badge/arXiv-2405.20337-b31b1b?logo=arxiv)](https://arxiv.org/abs/2405.20337) [![GitHub](https://img.shields.io/github/stars/wzzheng/OccSora)](https://github.com/wzzheng/OccSora) |
| `GenAD`  | Generalized Predictive Model for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2403.09630-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.09630) [![GitHub](https://img.shields.io/github/stars/OpenDriveLab/DriveAGI)](https://github.com/OpenDriveLab/DriveAGI) |
| `WorldGPT`  | Worldgpt: a sora-inspired video ai agent as rich world models from text and image inputs |  [![arXiv](https://img.shields.io/badge/arXiv-2403.07944-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.07944)|
| `DriveDreamer-2`  | DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2403.06845-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.06845) [![GitHub](https://img.shields.io/github/stars/f1yfisher/DriveDreamer2)](https://github.com/f1yfisher/DriveDreamer2) |
| `WoVoGen`  | WoVoGen: World Volume-Aware Diffusion for Controllable Multi-Camera Driving Scene Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2312.02934-b31b1b?logo=arxiv)](https://arxiv.org/abs/2312.02934) [![GitHub](https://img.shields.io/github/stars/fudan-zvg/WoVoGen)](https://github.com/fudan-zvg/WoVoGen) |
| `Drive-WM`  | Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2311.17918-b31b1b?logo=arxiv)](https://arxiv.org/abs/2311.17918) [![GitHub](https://img.shields.io/github/stars/BraveGroup/Drive-WM)](https://github.com/BraveGroup/Drive-WM) |
| `Panacea`  | Panacea: Panoramic and controllable video generation for autonomous driving |  [![arXiv](https://img.shields.io/badge/arXiv-2311.16813-b31b1b?logo=arxiv)](https://arxiv.org/abs/2311.16813) [![GitHub](https://img.shields.io/github/stars/wenyuqing/panacea)](https://github.com/wenyuqing/panacea) |
| `Drivingdiffusion`  | DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model |  [![arXiv](https://img.shields.io/badge/arXiv-2310.07771-b31b1b?logo=arxiv)](https://arxiv.org/abs/2310.07771) [![GitHub](https://img.shields.io/github/stars/shalfun/DrivingDiffusion)](https://github.com/shalfun/DrivingDiffusion) |
| `DriveDreamer`  | DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2309.09777-b31b1b?logo=arxiv)](https://arxiv.org/abs/2309.09777) [![GitHub](https://img.shields.io/github/stars/JeffWang987/DriveDreamer)](https://github.com/JeffWang987/DriveDreamer) |
| `UniPi`  | Learning universal policies via text-guided video generation |  [![arXiv](https://img.shields.io/badge/arXiv-2302.00111-b31b1b?logo=arxiv)](https://arxiv.org/abs/2302.00111) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://universal-policy.github.io/unipi/) |
||


### Autoregressive Diffusion

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `Yume1.5`  | Yume1.5: A Text-Controlled Interactive World Generation Model |  [![arXiv](https://img.shields.io/badge/arXiv-2512.22096-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.22096) [![GitHub](https://img.shields.io/github/stars/stdstu12/YUME)](https://github.com/stdstu12/YUME) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://stdstu12.github.io/YUME-Project/)|
| `HY-World 1.5`  | HY-World 1.5: A Systematic Framework for Interactive World Modeling with Real-Time Latency and Geometric Consistency |  [![GitHub](https://img.shields.io/github/stars/Tencent-Hunyuan/HY-WorldPlay)](https://github.com/Tencent-Hunyuan/HY-WorldPlay)|
| `Astra`  | Astra: General Interactive World Model with Autoregressive Denoising |  [![arXiv](https://img.shields.io/badge/arXiv-2512.08931-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.08931) [![GitHub](https://img.shields.io/github/stars/EternalEvan/Astra)](https://github.com/EternalEvan/Astra) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://eternalevan.github.io/Astra-project/)|
| `RELIC`  | RELIC: Interactive Video World Model with Long-Horizon Memory |  [![arXiv](https://img.shields.io/badge/arXiv-2512.04040-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.04040) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://relic-worldmodel.github.io/)|
| `ANWM`  | Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space |  [![arXiv](https://img.shields.io/badge/arXiv-2512.21887-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.21887)|
| `WorldPack`  | WorldPack: Compressed Memory Improves Spatial Consistency in Video World Modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2512.02473-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.02473)|
| `Hunyuan-GameCraft-2`  | Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2511.23429-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.23429) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://hunyuan-gamecraft-2.github.io/)|
| `PAN` | PAN: A World Model for General, Interactable, and Long-Horizon World Simulation | [![arXiv](https://img.shields.io/badge/arXiv-2511.09057-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.09057) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://ifm.mbzuai.ac.ae/pan/) |
| `Emu3.5` | Emu3.5: Native Multimodal Models are World Learners | [![arXiv](https://img.shields.io/badge/arXiv-2510.26583-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.26583) [![GitHub](https://img.shields.io/github/stars/baaivision/Emu3.5)](https://github.com/baaivision/Emu3.5) |
| `OmniNWM`  | OmniNWM: Omniscient Driving Navigation World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2510.18313-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.18313) [![GitHub](https://img.shields.io/github/stars/Ma-Zhuang/OmniNWM)](https://github.com/Ma-Zhuang/OmniNWM) |
| `WoW`  | WoW: Towards a World omniscient World model Through Embodied Interaction |  [![arXiv](https://img.shields.io/badge/arXiv-2509.22642-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.22642) [![GitHub](https://img.shields.io/github/stars/wow-world-model/wow-world-model)](https://github.com/wow-world-model/wow-world-model) |
| `LongScape`  | PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2509.21790-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.21790) [![GitHub](https://img.shields.io/github/stars/tsinghua-fib-lab/Longscape)](https://github.com/tsinghua-fib-lab/Longscape) |
| `Dreamer V4`  | Training Agents Inside of Scalable World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2509.24527-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.24527) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://danijar.com/project/dreamer4/) |
| `Genie Envisioner`  | Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2508.05635-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.05635) [![GitHub](https://img.shields.io/github/stars/AgibotTech/Genie-Envisioner)](https://github.com/AgibotTech/Genie-Envisioner) |
| `LiDARCrafter`  | LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences |  [![arXiv](https://img.shields.io/badge/arXiv-2508.03692-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.03692) [![GitHub](https://img.shields.io/github/stars/worldbench/lidarcrafter)](https://github.com/worldbench/lidarcrafter) |
| `Yan`  | Yan: Foundational Interactive Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2508.08601-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.08601) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://greatx3.github.io/Yan/)|
| `Matrix-Game 2.0`  | Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2508.13009-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.13009) [![GitHub](https://img.shields.io/github/stars/SkyworkAI/Matrix-Game)](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) |
| `Yume`  | Yume: An Interactive World Generation Model |  [![arXiv](https://img.shields.io/badge/arXiv-2507.17744-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.17744) [![GitHub](https://img.shields.io/github/stars/stdstu12/YUME)](https://github.com/stdstu12/YUME) |
| `Geometry Forcing`  | Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2507.07982-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.07982) [![GitHub](https://img.shields.io/github/stars/CIntellifusion/GeometryForcing)](https://github.com/CIntellifusion/GeometryForcing) |
| `spmem`  | Video World Models with Long-term Spatial Memory |  [![arXiv](https://img.shields.io/badge/arXiv-2506.05284-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.05284) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://spmem.github.io/)|
| `STAGE`  | STAGE: A Stream-Centric Generative World Model for Long-Horizon Driving-Scene Simulation |  [![arXiv](https://img.shields.io/badge/arXiv-2506.13138-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.13138) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://4dvlab.github.io/STAGE/) |
| `CaM`  | Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval |  [![arXiv](https://img.shields.io/badge/arXiv-2506.03141-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.03141) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://context-as-memory.github.io/) |
| `DeepVerse`  | DeepVerse: 4D Autoregressive Video Generation as a World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2506.01103-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.01103) [![GitHub](https://img.shields.io/github/stars/SOTAMak1r/DeepVerse)](https://github.com/SOTAMak1r/DeepVerse) |
| `Epona`  | Epona: Autoregressive Diffusion World Model for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2506.24113-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.24113) [![GitHub](https://img.shields.io/github/stars/Kevin-thu/Epona)](https://github.com/Kevin-thu/Epona) |
| `SceneDiffuser++`  | SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2506.21976-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.21976) |
| `VMem`  | VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory |  [![arXiv](https://img.shields.io/badge/arXiv-2506.18903-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.18903) [![GitHub](https://img.shields.io/github/stars/runjiali-rl/vmem)](https://github.com/runjiali-rl/vmem) |
| `Hunyuan-GameCraft`  | Hunyuan-GameCraft: High-dynamic Interactive Game Video Generation with Hybrid History Condition |  [![arXiv](https://img.shields.io/badge/arXiv-2506.17201-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.17201) [![GitHub](https://img.shields.io/github/stars/Tencent-Hunyuan/Hunyuan-GameCraft-1.0)](https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0) |
| `Matrix-Game`  | Matrix-Game: Interactive World Foundation Model |  [![arXiv](https://img.shields.io/badge/arXiv-2506.18701-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.18701) [![GitHub](https://img.shields.io/github/stars/SkyworkAI/Matrix-Game)](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-1) |
| `PEVA`  | Whole-Body Conditioned Egocentric Video Prediction |  [![arXiv](https://img.shields.io/badge/arXiv-2506.21552-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.21552)[![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://dannytran123.github.io/PEVA/)  |
| `NFD`  | Playing with Transformer at 30+ FPS via Next-Frame Diffusion |  [![arXiv](https://img.shields.io/badge/arXiv-2506.01380-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.01380)[![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://nextframed.github.io/)  |
| `VRAG`  | Learning World Models for Interactive Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2505.21996-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.21996) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://sites.google.com/view/vrag)|
| `DriVerse`  | DriVerse: Navigation World Model for Driving Simulation via Multimodal Trajectory Prompting and Motion Alignment |  [![arXiv](https://img.shields.io/badge/arXiv-2504.18576-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.18576) [![GitHub](https://img.shields.io/github/stars/shalfun/DriVerse)](https://github.com/shalfun/DriVerse) |
| `WORLDMEM`  | WORLDMEM: Long-term Consistent World Simulation with Memory |  [![arXiv](https://img.shields.io/badge/arXiv-2504.12369-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.12369) [![GitHub](https://img.shields.io/github/stars/xizaoqu/WorldMem)](https://github.com/xizaoqu/WorldMem) |
| `Adaworld`  | Adaworld: Learning adaptable world models with latent actions |  [![arXiv](https://img.shields.io/badge/arXiv-2503.18938-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.18938) [![GitHub](https://img.shields.io/github/stars/Little-Podi/AdaWorld)](https://github.com/Little-Podi/AdaWorld) |
| `TSWM`  | Toward Stable World Models: Measuring and Addressing World Instability in Generative Environments |  [![arXiv](https://img.shields.io/badge/arXiv-2503.08122-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.08122)|
| `Gamefactory`  | Gamefactory: Creating new games with generative interactive videos |  [![arXiv](https://img.shields.io/badge/arXiv-2501.08325-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.08325) [![GitHub](https://img.shields.io/github/stars/KwaiVGI/GameFactory)](https://github.com/KwaiVGI/GameFactory) |
| `PlayGen`  | Playable Game Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2412.00887-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.00887) [![GitHub](https://img.shields.io/github/stars/GreatX3/Playable-Game-Generation)](https://github.com/GreatX3/Playable-Game-Generation) |
| `InfinityDrive`  | InfinityDrive: Breaking Time Limits in Driving World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2412.01522-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.01522) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://metadrivescape.github.io/papers_project/InfinityDrive/page.html) |
| `GEM`  | Gem: A generalizable ego-vision multimodal world model for fine-grained ego-motion, object dynamics, and scene composition control |  [![arXiv](https://img.shields.io/badge/arXiv-2412.11198-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.11198) [![GitHub](https://img.shields.io/github/stars/vita-epfl/GEM)](https://github.com/vita-epfl/GEM) |
| `UniMLVG`  | UniMLVG: Unified Framework for Multi-view Long Video Generation with Comprehensive Control Capabilities for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2412.04842-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.04842) [![GitHub](https://img.shields.io/github/stars/SenseTime-FVG/OpenDWM)](https://github.com/SenseTime-FVG/OpenDWM) |
| `The Matrix`  | The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control |  [![arXiv](https://img.shields.io/badge/arXiv-2412.03568-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.03568)[![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://thematrix1999.github.io/index.html) |
| `NWM`  | Navigation World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2412.03572-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.03572) [![GitHub](https://img.shields.io/github/stars/facebookresearch/nwm)](https://github.com/facebookresearch/nwm) |
| `Copilot4D`  | Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion |  [![arXiv](https://img.shields.io/badge/arXiv-2311.01017-b31b1b?logo=arxiv)](https://arxiv.org/abs/2311.01017) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://waabi.ai/research/copilot-4d) |
| `Drivingsphere`  | Drivingsphere: Building a high-fidelity 4d world for closed-loop simulation |  [![arXiv](https://img.shields.io/badge/arXiv-2411.11252-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.11252) [![GitHub](https://img.shields.io/github/stars/yanty123/DrivingSphere)](https://github.com/yanty123/DrivingSphere) |
| `Gamegen-x`  | Gamegen-x: Interactive open-world game video generation |  [![arXiv](https://img.shields.io/badge/arXiv-2411.00769-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.00769) [![GitHub](https://img.shields.io/github/stars/GameGen-X/GameGen-X)](https://github.com/GameGen-X/GameGen-X) |
| `Genie 2`  | Genie 2: A large-scale foundation world model |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)|
| `oasis`  | Oasis: A Universe in a Transformer | [![GitHub](https://img.shields.io/github/stars/etched-ai/open-oasis)](https://github.com/etched-ai/open-oasis) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://oasis-model.github.io/)|
| `GameNGen`  | Diffusion models are real-time game engines |  [![arXiv](https://img.shields.io/badge/arXiv-2408.14837-b31b1b?logo=arxiv)](https://arxiv.org/abs/2408.14837)[![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://gamengen.github.io/)|
| `IRAsim`  | IRASim: A Fine-Grained World Model for Robot Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2406.14540-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.14540) [![GitHub](https://img.shields.io/github/stars/bytedance/IRASim)](https://github.com/bytedance/IRASim) |
| `diamond`  | Diffusion for World Modeling: Visual Details Matter in Atari |  [![arXiv](https://img.shields.io/badge/arXiv-2405.12399-b31b1b?logo=arxiv)](https://arxiv.org/abs/2405.12399) [![GitHub](https://img.shields.io/github/stars/eloialonso/diamond)](https://github.com/eloialonso/diamond) |
| `Vista`  | Vista: A generalizable driving world model with high fidelity and versatile controllability |  [![arXiv](https://img.shields.io/badge/arXiv-2405.17398-b31b1b?logo=arxiv)](https://arxiv.org/abs/2405.17398) [![GitHub](https://img.shields.io/github/stars/OpenDriveLab/Vista)](https://github.com/OpenDriveLab/Vista) |
| `UniSim`  | UniSim: Learning Interactive Real-World Simulators |  [![arXiv](https://img.shields.io/badge/arXiv-2310.06114-b31b1b?logo=arxiv)](https://arxiv.org/abs/2310.06114)[![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://universal-simulator.github.io/unisim//)  |
||


## 1.3. Predictive Architecture

### JEPA

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
| `DINO-world`  | Back to the Features: DINO as a Foundation for Video World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2507.19468-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.19468) |
| `V-JEPA 2`  | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning |  [![arXiv](https://img.shields.io/badge/arXiv-2506.09985-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.09985) [![GitHub](https://img.shields.io/github/stars/facebookresearch/vjepa2)](https://github.com/facebookresearch/vjepa2) |
| `SIVWM`  | Sparse Imagination for Efficient Visual World Model Planning |  [![arXiv](https://img.shields.io/badge/arXiv-2506.01392-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.01392) |
| `seq-JEPA`  | seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2505.03176-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.03176) [![GitHub](https://img.shields.io/github/stars/hafezgh/seq-jepa)](https://github.com/hafezgh/seq-jepa) |
| `Flare`  | Flare: Robot learning with implicit world modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2505.15659-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.15659) [![GitHub](https://img.shields.io/github/stars/JiahengHu/FLaRe)](https://github.com/JiahengHu/FLaRe) |
| `OSVI-WM`  | OSVI-WM: One-Shot Visual Imitation for Unseen Tasks using World-Model-Guided Trajectory Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2505.20425-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.20425) [![GitHub](https://img.shields.io/github/stars/raktimgg/osvi-wm)](https://github.com/raktimgg/osvi-wm) |
| `EchoWorld`  | EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance |  [![arXiv](https://img.shields.io/badge/arXiv-2504.13065-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.13065) [![GitHub](https://img.shields.io/github/stars/LeapLabTHU/EchoWorld)](https://github.com/LeapLabTHU/EchoWorld) |
| `AD-L-JEPA`  | AD-L-JEPA: Self-Supervised Spatial World Models with Joint Embedding Predictive Architecture for Autonomous Driving with LiDAR Data |  [![arXiv](https://img.shields.io/badge/arXiv-2501.04969-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.04969) [![GitHub](https://img.shields.io/github/stars/HaoranZhuExplorer/AD-L-JEPA-Release)](https://github.com/HaoranZhuExplorer/AD-L-JEPA-Release) |
| `DINO-Foresight`  | DINO-Foresight: Looking into the Future with DINO |  [![arXiv](https://img.shields.io/badge/arXiv-2412.11673-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.11673) [![GitHub](https://img.shields.io/github/stars/Sta8is/DINO-Foresight)](https://github.com/Sta8is/DINO-Foresight) |
| `DINO-WM`  | DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning |  [![arXiv](https://img.shields.io/badge/arXiv-2411.04983-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.04983) [![GitHub](https://img.shields.io/github/stars/gaoyuezhou/dino_wm)](https://github.com/gaoyuezhou/dino_wm) |
| `LAW`  | Enhancing End-to-End Autonomous Driving with Latent World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2406.08481-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.08481) [![GitHub](https://img.shields.io/github/stars/BraveGroup/LAW)](https://github.com/BraveGroup/LAW) |
| `V-JEPA`  | Revisiting Feature Prediction for Learning Visual Representations from Video |  [![arXiv](https://img.shields.io/badge/arXiv-2404.08471-b31b1b?logo=arxiv)](https://arxiv.org/abs/2404.08471) [![GitHub](https://img.shields.io/github/stars/facebookresearch/jepa)](https://github.com/facebookresearch/jepa) |
| `IWM`  | Learning and Leveraging World Models in Visual Representation Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2403.00504-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.00504) |
||

## 1.4. State Transition

### Latent State-Space Modeling

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `BOOM`  | Bootstrap Off-policy with World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2511.00423-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.00423) [![GitHub](https://img.shields.io/github/stars/molumitu/BOOM_MBRL)](https://github.com/molumitu/BOOM_MBRL)|
| `GASv2`  | Visuomotor Grasping with World Models for Surgical Robots |  [![arXiv](https://img.shields.io/badge/arXiv-2508.11200-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.11200) |
| `LPS`  | Latent Policy Steering with Embodiment-Agnostic Pretrained World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2507.13340-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.13340) |
| `EMERALD`  | Accurate and Efficient World Modeling with Masked Latent Transformers |  [![arXiv](https://img.shields.io/badge/arXiv-2507.04075-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.04075) [![GitHub](https://img.shields.io/github/stars/burchim/EMERALD)](https://github.com/burchim/EMERALD) |
| `FOUNDER`  | FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making |  [![arXiv](https://img.shields.io/badge/arXiv-2507.12496-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.12496) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://sites.google.com/view/founder-rl)|
| `NavMorph`  | NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments |  [![arXiv](https://img.shields.io/badge/arXiv-2506.23468-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.23468) [![GitHub](https://img.shields.io/github/stars/Feliciaxyao/NavMorph)](https://github.com/Feliciaxyao/NavMorph) |
| `Raw2Drive`  | Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2505.16394-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.16394)  |
| `SSVWM`  | Long-Context State-Space Video World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2505.20171-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.20171) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://ryanpo.com/ssm_wm/)|
| `PIN-WM`  | PIN-WM: Learning Physics-Informed World Models for Non-Prehensile Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2504.16693-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.16693) [![GitHub](https://img.shields.io/github/stars/XuAdventurer/PIN-WM)](https://github.com/XuAdventurer/PIN-WM)|
| `ReDRAW`  | Adapting World Models with Latent-State Dynamics Residuals |  [![arXiv](https://img.shields.io/badge/arXiv-2504.02252-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.02252) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://redraw.jblanier.net/)|
| `WoTE`  | End-to-End Driving with Online Trajectory Evaluation via BEV World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2504.01941-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.01941) [![GitHub](https://img.shields.io/github/stars/liyingyanUCAS/WoTE)](https://github.com/liyingyanUCAS/WoTE) |
| `DyWA`  | DyWA: Dynamics-Adaptive World Action Model for Generalizable Non-Prehensile Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2503.16806-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.16806) [![GitHub](https://img.shields.io/github/stars/jiangranlv/DyWA)](https://github.com/jiangranlv/DyWA)|
| `DMWM`  | DMWM: Dual-Mind World Model with Long-Term Imagination |  [![arXiv](https://img.shields.io/badge/arXiv-2502.07591-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.07591) [![GitHub](https://img.shields.io/github/stars/news-vt/DMWM)](https://github.com/news-vt/DMWM) |
| `Simulus`  | Uncovering Untapped Potential in Sample-Efficient World Model Agents |  [![arXiv](https://img.shields.io/badge/arXiv-2502.11537-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.11537) [![GitHub](https://img.shields.io/github/stars/leor-c/Simulus)](https://github.com/leor-c/Simulus) |
| `S5WM`  | Accelerating Model-Based Reinforcement Learning with State-Space World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2502.20168-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.20168) |
| `AdaWM`  | AdaWM: Adaptive World Model based Planning for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2501.13072-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.13072) |
| `RoboHorizon`  | RoboHorizon: An LLM-Assisted Multi-View World Model for Long-Horizon Robotic Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2501.06605-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.06605) |
| `LS-Imagine`  | Open-World Reinforcement Learning over Long Short-Term Imagination |  [![arXiv](https://img.shields.io/badge/arXiv-2410.03618-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.03618) [![GitHub](https://img.shields.io/github/stars/qiwang067/LS-Imagine)](https://github.com/qiwang067/LS-Imagine) |
| `-`  | Mitigating Covariate Shift in Imitation Learning for Autonomous Vehicles Using Latent Space Generative World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2409.16663-b31b1b?logo=arxiv)](https://arxiv.org/abs/2409.16663) |
| `GenRL`  | GenRL: Multimodal-Foundation World Models for Generalization in Embodied Agents |  [![arXiv](https://img.shields.io/badge/arXiv-2406.18043-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.18043) [![GitHub](https://img.shields.io/github/stars/mazpie/genrl)](https://github.com/mazpie/genrl)|
| `DriveWorld`  | DriveWorld: 4D Pre-Trained Scene Understanding via World Models for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2405.04390-b31b1b?logo=arxiv)](https://arxiv.org/abs/2405.04390) |
| `Puppeteer`  | Hierarchical World Models as Visual Whole-Body Humanoid Controllers |  [![arXiv](https://img.shields.io/badge/arXiv-2405.18418-b31b1b?logo=arxiv)](https://arxiv.org/abs/2405.18418) [![GitHub](https://img.shields.io/github/stars/nicklashansen/puppeteer)](https://github.com/nicklashansen/puppeteer)|
| `R2I`  | Mastering Memory Tasks with World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2403.04253-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.04253) [![GitHub](https://img.shields.io/github/stars/chandar-lab/Recall2Imagine)](https://github.com/chandar-lab/Recall2Imagine)|
| `REM`  | Improving Token-Based World Models with Parallel Observation Prediction |  [![arXiv](https://img.shields.io/badge/arXiv-2402.05643-b31b1b?logo=arxiv)](https://arxiv.org/abs/2402.05643) [![GitHub](https://img.shields.io/github/stars/leor-c/REM)](https://github.com/leor-c/REM) |
| `Think2Drive`  | Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2402.16720-b31b1b?logo=arxiv)](https://arxiv.org/abs/2402.16720) [![GitHub](https://img.shields.io/github/stars/Thinklab-SJTU/Bench2Drive)](https://github.com/Thinklab-SJTU/Bench2Drive)|
| `MUVO`  | MUVO: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations |  [![arXiv](https://img.shields.io/badge/arXiv-2311.11762-b31b1b?logo=arxiv)](https://arxiv.org/abs/2311.11762) [![GitHub](https://img.shields.io/github/stars/fzi-forschungszentrum-informatik/muvo)](https://github.com/fzi-forschungszentrum-informatik/muvo) |
| `STORM`  | STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2310.09615-b31b1b?logo=arxiv)](https://arxiv.org/abs/2310.09615) [![GitHub](https://img.shields.io/github/stars/weipu-zhang/STORM)](https://github.com/weipu-zhang/STORM) |
| `HarmonyDream`  | HarmonyDream: Task Harmonization Inside World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2310.00344-b31b1b?logo=arxiv)](https://arxiv.org/abs/2310.00344) [![GitHub](https://img.shields.io/github/stars/thuml/HarmonyDream)](https://github.com/thuml/HarmonyDream) |
| `TD-MPC2`  | TD-MPC2: Scalable, Robust World Models for Continuous Control |  [![arXiv](https://img.shields.io/badge/arXiv-2310.16828-b31b1b?logo=arxiv)](https://arxiv.org/abs/2310.16828) [![GitHub](https://img.shields.io/github/stars/nicklashansen/tdmpc2)](https://github.com/nicklashansen/tdmpc2) |
| `MoDem-V2`  | MoDem-V2: Visuo-Motor World Models for Real-World Robot Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2309.14236-b31b1b?logo=arxiv)](https://arxiv.org/abs/2309.14236) [![GitHub](https://img.shields.io/github/stars/facebookresearch/modemv2)](https://github.com/facebookresearch/modemv2)|
| `SWIM`  | Structured World Models from Human Videos |  [![arXiv](https://img.shields.io/badge/arXiv-2308.10901-b31b1b?logo=arxiv)](https://arxiv.org/abs/2308.10901) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://human-world-model.github.io/)|
| `Dynalang`  | Learning to Model the World with Language |  [![arXiv](https://img.shields.io/badge/arXiv-2308.01399-b31b1b?logo=arxiv)](https://arxiv.org/abs/2308.01399) [![GitHub](https://img.shields.io/github/stars/jlin816/dynalang)](https://github.com/jlin816/dynalang)|
| `SafeDreamer`  | SafeDreamer: Safe Reinforcement Learning with World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2307.07176-b31b1b?logo=arxiv)](https://arxiv.org/abs/2307.07176) [![GitHub](https://img.shields.io/github/stars/PKU-Alignment/SafeDreamer)](https://github.com/PKU-Alignment/SafeDreamer) |
| `CoWorld`  | Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2305.15260-b31b1b?logo=arxiv)](https://arxiv.org/abs/2305.15260) [![GitHub](https://img.shields.io/github/stars/qiwang067/CoWorld)](https://github.com/qiwang067/CoWorld) |
| `TWM`  | Transformer-Based World Models Are Happy With 100k Interactions |  [![arXiv](https://img.shields.io/badge/arXiv-2303.07109-b31b1b?logo=arxiv)](https://arxiv.org/abs/2303.07109) [![GitHub](https://img.shields.io/github/stars/jrobine/twm)](https://github.com/jrobine/twm) |
| `MV-MWM`  | Multi-View Masked World Models for Visual Robotic Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2302.02408-b31b1b?logo=arxiv)](https://arxiv.org/abs/2302.02408) [![GitHub](https://img.shields.io/github/stars/younggyoseo/MV-MWM)](https://github.com/younggyoseo/MV-MWM) |
| `DreamV3`  | Mastering Diverse Domains through World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2301.04104-b31b1b?logo=arxiv)](https://arxiv.org/abs/2301.04104) [![GitHub](https://img.shields.io/github/stars/danijar/dreamerv3)](https://github.com/danijar/dreamerv3) |
| `IRIS`  | Transformers are Sample-Efficient World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2209.00588-b31b1b?logo=arxiv)](https://arxiv.org/abs/2209.00588) [![GitHub](https://img.shields.io/github/stars/eloialonso/iris)](https://github.com/eloialonso/iris) |
| `MWM`  | Masked World Models for Visual Control |  [![arXiv](https://img.shields.io/badge/arXiv-2206.14244-b31b1b?logo=arxiv)](https://arxiv.org/abs/2206.14244) [![GitHub](https://img.shields.io/github/stars/younggyoseo/MWM)](https://github.com/younggyoseo/MWM) |
| `DayDreamer`  | DayDreamer: World Models for Physical Robot Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2206.14176-b31b1b?logo=arxiv)](https://arxiv.org/abs/2206.14176) [![GitHub](https://img.shields.io/github/stars/danijar/daydreamer)](https://github.com/danijar/daydreamer) |
| `CADDY`  | Playable Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2101.12195-b31b1b?logo=arxiv)](https://arxiv.org/abs/2101.12195) [![GitHub](https://img.shields.io/github/stars/willi-menapace/PlayableVideoGeneration)](https://github.com/willi-menapace/PlayableVideoGeneration) |
| `DreamV2`  | Mastering Atari with Discrete World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2010.02193-b31b1b?logo=arxiv)](https://arxiv.org/abs/2010.02193) [![GitHub](https://img.shields.io/github/stars/danijar/dreamerv2)](https://github.com/danijar/dreamerv2) |
| `DreamV1`  | Dream to Control: Learning Behaviors by Latent Imagination |  [![arXiv](https://img.shields.io/badge/arXiv-1912.01603-b31b1b?logo=arxiv)](https://arxiv.org/abs/1912.01603) [![GitHub](https://img.shields.io/github/stars/danijar/dreamer)](https://github.com/danijar/dreamer) |
| `PlaNet`  | Learning Latent Dynamics for Planning from Pixels |  [![arXiv](https://img.shields.io/badge/arXiv-1811.04551-b31b1b?logo=arxiv)](https://arxiv.org/abs/1811.04551) [![GitHub](https://img.shields.io/github/stars/google-research/planet)](https://github.com/google-research/planet) |
||

### Object-Centric Modeling

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `FIOC-WM`  | Learning Interactive World Model for Object-Centric Reinforcement Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2511.02225-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.02225)  |
| `Dyn-o`  | Dyn-O: Building Structured World Models with Object-Centric Representations |  [![arXiv](https://img.shields.io/badge/arXiv-2507.03298-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.03298) [![GitHub](https://img.shields.io/github/stars/wangzizhao/dyn-O)](https://github.com/wangzizhao/dyn-O) |
| `SlotPi`  | SlotPi: Physics-informed Object-centric Reasoning Models |  [![arXiv](https://img.shields.io/badge/arXiv-2506.10778-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.10778) |
| `-`  | Object-Centric World Model for Language-Guided Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2503.06170-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.06170) |
| `DisWM`  | Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2503.08751-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.08751) [![GitHub](https://img.shields.io/github/stars/qiwang067/DisWM)](https://github.com/qiwang067/DisWM) |
| `Objects matter`  | Objects matter: object-centric world models improve reinforcement learning in visually complex environments |  [![arXiv](https://img.shields.io/badge/arXiv-2501.16443-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.16443) |
| `Dreamweaver`  | Dreamweaver: Learning Compositional World Models from Pixels |  [![arXiv](https://img.shields.io/badge/arXiv-2501.14174-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.14174) [![GitHub](https://img.shields.io/github/stars/ahn-ml/dreamweaver-release)](https://github.com/ahn-ml/dreamweaver-release) |
| `MEAD`  | Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction |  [![arXiv](https://img.shields.io/badge/arXiv-2408.11816-b31b1b?logo=arxiv)](https://arxiv.org/abs/2408.11816) |
| `slotSSM`  | Slot State Space Models |  [![arXiv](https://img.shields.io/badge/arXiv-2406.12272-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.12272) [![GitHub](https://img.shields.io/github/stars/JindongJiang/SlotSSMs)](https://github.com/JindongJiang/SlotSSMs) |
| `RoboDreamer`  | RoboDreamer: Learning Compositional World Models for Robot Imagination |  [![arXiv](https://img.shields.io/badge/arXiv-2404.12377-b31b1b?logo=arxiv)](https://arxiv.org/abs/2404.12377) [![GitHub](https://img.shields.io/github/stars/rainbow979/robodreamer)](https://github.com/rainbow979/robodreamer) |
| `SSWM`  | Slot Structured World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2402.03326-b31b1b?logo=arxiv)](https://arxiv.org/abs/2402.03326) [![GitHub](https://img.shields.io/github/stars/JonathanCollu/Slot-Structured-World-Models)](https://github.com/JonathanCollu/Slot-Structured-World-Models) |
| `cosmos`  | Neurosymbolic Grounding for Compositional World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2310.12690-b31b1b?logo=arxiv)](https://arxiv.org/abs/2310.12690) [![GitHub](https://img.shields.io/github/stars/trishullab/cosmos)](https://github.com/trishullab/cosmos) |
| `FOCUS`  | FOCUS: Object-Centric World Models for Robotics Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2307.02427-b31b1b?logo=arxiv)](https://arxiv.org/abs/2307.02427) [![GitHub](https://img.shields.io/github/stars/StefanoFerraro/FOCUS)](https://github.com/StefanoFerraro/FOCUS)|
| `SlotFormer`  | SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models |  [![arXiv](https://img.shields.io/badge/arXiv-2210.05861-b31b1b?logo=arxiv)](https://arxiv.org/abs/2210.05861) [![GitHub](https://img.shields.io/github/stars/pairlab/SlotFormer)](https://github.com/pairlab/SlotFormer) |
| `HOWM`  | Toward Compositional Generalization in Object-Oriented World Modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2204.13661-b31b1b?logo=arxiv)](https://arxiv.org/abs/2204.13661) [![GitHub](https://img.shields.io/github/stars/linfeng-z/HOWM)](https://github.com/linfeng-z/HOWM) |
| `G-SWM`  | Improving Generative Imagination in Object-Centric World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2010.02054-b31b1b?logo=arxiv)](https://arxiv.org/abs/2010.02054) [![GitHub](https://img.shields.io/github/stars/zhixuan-lin/G-SWM)](https://github.com/zhixuan-lin/G-SWM) |
||

## 1.5. Other Architectures

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `PhysWorld` | PhysWorld: From Real Videos to World Models of Deformable Objects via Physics-Aware Demonstration Synthesis | [![arXiv](https://img.shields.io/badge/arXiv-2510.21447-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.21447) [![GitHub](https://img.shields.io/github/stars/AlanYoung123/PhysWorld)](https://github.com/AlanYoung123/PhysWorld) |
| `FASTopoWM` | FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models | [![arXiv](https://img.shields.io/badge/arXiv-2507.23325-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.23325)  [![GitHub](https://img.shields.io/github/stars/YimingYang23/FASTopoWM)](https://github.com/YimingYang23/FASTopoWM)|
| `GWM` | Graph World Model | [![arXiv](https://img.shields.io/badge/arXiv-2507.10539-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.10539) [![GitHub](https://img.shields.io/github/stars/ulab-uiuc/GWM)](https://github.com/ulab-uiuc/GWM) |
| `World4Drive` | World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model | [![arXiv](https://img.shields.io/badge/arXiv-2507.00603-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.00603) [![GitHub](https://img.shields.io/github/stars/ucaszyp/World4Drive)](https://github.com/ucaszyp/World4Drive) |
| `Orbis` | Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models | [![arXiv](https://img.shields.io/badge/arXiv-2507.13162-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.13162) [![GitHub](https://img.shields.io/github/stars/lmb-freiburg/orbis)](https://github.com/lmb-freiburg/orbis) |
| `LiDARWM` | Towards foundational LiDAR world models with efficient latent flow matching | [![arXiv](https://img.shields.io/badge/arXiv-2506.23434-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.23434)  [![GitHub](https://img.shields.io/github/stars/Orbis36/OccFM-NeurIPS2025)](https://github.com/Orbis36/OccFM-NeurIPS2025)|
| `ManiGaussian++` | ManiGaussian++: General Robotic Bimanual Manipulation with Hierarchical Gaussian World Model | [![arXiv](https://img.shields.io/badge/arXiv-2506.19842-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.19842) [![GitHub](https://img.shields.io/github/stars/April-Yz/ManiGaussian_Bimanual)](https://github.com/April-Yz/ManiGaussian_Bimanual) |
| `GAF` | GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2506.14135-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.14135) |
| `HWM` | Humanoid World Models: Open World Foundation Models for Humanoid Robotics | [![arXiv](https://img.shields.io/badge/arXiv-2506.01182-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.01182) |
| `DriveX` | DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving | [![arXiv](https://img.shields.io/badge/arXiv-2505.19239-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.19239) |
| `OccProphet` | OccProphet: Pushing Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with Observer-Forecaster-Refiner Framework | [![arXiv](https://img.shields.io/badge/arXiv-2502.15180-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.15180) [![GitHub](https://img.shields.io/github/stars/JLChen-C/OccProphet)](https://github.com/JLChen-C/OccProphet)|
| `FleetWM` | Multi-Task Interactive Robot Fleet Learning with Visual World Models | [![arXiv](https://img.shields.io/badge/arXiv-2410.22689-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.22689)  [![GitHub](https://img.shields.io/github/stars/UT-Austin-RPL/sirius-fleet)](https://github.com/UT-Austin-RPL/sirius-fleet)|
| `GaussianWorld` | GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction | [![arXiv](https://img.shields.io/badge/arXiv-2412.10373-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.10373) [![GitHub](https://img.shields.io/github/stars/zuosc19/GaussianWorld)](https://github.com/zuosc19/GaussianWorld) |
| `NeMo` | Neural volumetric world models for autonomous driving | [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02571.pdf) |
||


# 2. Datasets \& Benchmarks

## 2.1. General-Purpose World Modeling

### World Prediction and Simulation

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `DynamicVerse`  | DynamicVerse: A Physically-Aware Multimodal Framework for 4D World Modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2512.03000-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.03000) [![GitHub](https://img.shields.io/github/stars/Dynamics-X/DynamicVerse)](https://github.com/Dynamics-X/DynamicVerse) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://dynamic-verse.github.io/)|
| `4DWorldBench`  | 4DWorldBench: A Comprehensive Evaluation Framework for 3D/4D World Generation Models |  [![arXiv](https://img.shields.io/badge/arXiv-2511.19836-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.19836) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://yeppp27.github.io/4DWorldBench.github.io/)|
| `Gen-ViRe`  | Can World Simulators Reason? Gen-ViRe: A Generative Visual Reasoning Benchmark |  [![arXiv](https://img.shields.io/badge/arXiv-2511.13853-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.13853) [![GitHub](https://img.shields.io/github/stars/L-CodingSpace/GVR)](https://github.com/L-CodingSpace/GVR) |
| `World-in-World`  | World-in-World: World Models in a Closed-Loop World |  [![arXiv](https://img.shields.io/badge/arXiv-2510.18135-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.18135) [![GitHub](https://img.shields.io/github/stars/World-In-World/world-in-world)](https://github.com/World-In-World/world-in-world) |
| `OmniWorld`  | OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling |  [![arXiv](https://img.shields.io/badge/arXiv-2509.12201-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.12201) [![GitHub](https://img.shields.io/github/stars/yangzhou24/OmniWorld)](https://github.com/yangzhou24/OmniWorld) |
| `Sekai`  | Sekai: A Video Dataset towards World Exploration |  [![arXiv](https://img.shields.io/badge/arXiv-2506.15675-b31b1b?&logo=arxiv)](https://arxiv.org/abs/2506.15675) [![GitHub](https://img.shields.io/github/stars/Lixsp11/sekai-codebase)](https://github.com/Lixsp11/sekai-codebase) |
| `WorldPrediction`  | WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning |  [![arXiv](https://img.shields.io/badge/arXiv-2506.04363-b31b1b?logo=arxiv)](https://www.arxiv.org/abs/2506.04363) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://worldprediction.github.io/)|
| `WorldScore`  | WorldScore: A Unified Evaluation Benchmark for World Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2504.00983-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.00983) [![GitHub](https://img.shields.io/github/stars/haoyi-duan/WorldScore)](https://github.com/haoyi-duan/WorldScore) |
| `MM-OR`  | MM-OR: A Large Multimodal Operating Room Dataset for Semantic Understanding of High-Intensity Surgical Environments |  [![arXiv](https://img.shields.io/badge/arXiv-2503.02579-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.02579) [![GitHub](https://img.shields.io/github/stars/egeozsoy/MM-OR)](https://github.com/egeozsoy/MM-OR) |
| `WorldModelBench`  | WorldModelBench: Judging Video Generation Models As World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2502.20694-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.20694) [![GitHub](https://img.shields.io/github/stars/WorldModelBench-Team/WorldModelBench)](https://github.com/WorldModelBench-Team/WorldModelBench) |
| `OpenHumanVid`  | OpenHumanVid: A Large-Scale High-Quality Dataset for Enhancing Human-Centric Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2412.00115-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.00115) [![GitHub](https://img.shields.io/github/stars/fudan-generative-vision/OpenHumanVid)](https://github.com/fudan-generative-vision/OpenHumanVid) |
| `EgoVid-5M`  | EgoVid-5M: A Large-Scale Video-Action Dataset for Egocentric Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2411.08380-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.08380) [![GitHub](https://img.shields.io/github/stars/JeffWang987/EgoVid)](https://github.com/JeffWang987/EgoVid) |
| `Ego-Exo4D`  | Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives |  [![arXiv](https://img.shields.io/badge/arXiv-2311.18259-b31b1b?logo=arxiv)](https://arxiv.org/abs/2311.18259) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://ego-exo4d-data.org/) |
| `InternVid`  | InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2307.06942-b31b1b?logo=arxiv)](https://arxiv.org/abs/2307.06942) [![GitHub](https://img.shields.io/github/stars/OpenGVLab/InternVideo)](https://github.com/OpenGVLab/InternVideo) |
| `Ego4D`  | Ego4D: Around the World in 3,000 Hours of Egocentric Video |  [![arXiv](https://img.shields.io/badge/arXiv-2110.07058-b31b1b?logo=arxiv)](http://arxiv.org/abs/2110.07058) [![GitHub](https://img.shields.io/github/stars/EGO4D)](https://github.com/EGO4D) |
| `WebVid-2M`  | Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval |  [![arXiv](https://img.shields.io/badge/arXiv-2104.00650-b31b1b?logo=arxiv)](https://arxiv.org/abs/2104.00650) [![GitHub](https://img.shields.io/github/stars/m-bain/webvid)](https://github.com/m-bain/webvid) |
| `EPIC-KITCHENS-100`  | Rescaling Egocentric Vision |  [![arXiv](https://img.shields.io/badge/arXiv-2006.13256-b31b1b?logo=arxiv)](https://arxiv.org/abs/2006.13256) [![GitHub](https://img.shields.io/github/stars/epic-kitchens)](https://github.com/epic-kitchens) |
| `HowTo100M`  | HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips |  [![arXiv](https://img.shields.io/badge/arXiv-1906.03327-b31b1b?logo=arxiv)](https://arxiv.org/abs/1906.03327) [![GitHub](https://img.shields.io/github/stars/antoine77340/howto100m)](https://github.com/antoine77340/howto100m) |
| `COIN`  | COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis |  [![arXiv](https://img.shields.io/badge/arXiv-1903.02874-b31b1b?logo=arxiv)](https://arxiv.org/abs/1903.02874) [![GitHub](https://img.shields.io/github/stars/coin-dataset)](https://github.com/coin-dataset/) |
| `SSV2`  | The "something something" video database for learning and evaluating visual common sense |  [![arXiv](https://img.shields.io/badge/arXiv-1706.04261-b31b1b?logo=arxiv)](https://arxiv.org/abs/1706.04261) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) |
| `Kinetics`  | The Kinetics Human Action Video Dataset |  [![arXiv](https://img.shields.io/badge/arXiv-1705.06950-b31b1b?logo=arxiv)](https://arxiv.org/abs/1705.06950) [![GitHub](https://img.shields.io/github/stars/cvdfoundation/kinetics-dataset)](https://github.com/cvdfoundation/kinetics-dataset) |
| `YouTube-8M`  | YouTube-8M: A Large-Scale Video Classification Benchmark |  [![arXiv](https://img.shields.io/badge/arXiv-1609.08675-b31b1b?logo=arxiv)](https://arxiv.org/abs/1609.08675) [![GitHub](https://img.shields.io/github/stars/google/youtube-8m)](https://github.com/google/youtube-8m) |
| `UCF101`  | UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild |  [![arXiv](https://img.shields.io/badge/arXiv-1212.0402-b31b1b?logo=arxiv)](https://arxiv.org/abs/1212.0402) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://www.crcv.ucf.edu/data/UCF101.php)|
||

### Physical and Causal Reasoning

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `VideoVerse`  | VideoVerse: How Far is Your T2V Generator from a World Model? |  [![arXiv](https://img.shields.io/badge/arXiv-2510.08398-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.08398) [![GitHub](https://img.shields.io/github/stars/Zeqing-Wang/VideoVerse)](https://github.com/Zeqing-Wang/VideoVerse) |
| `PAI-Bench`  | Physical ai bench: A comprehensive benchmark for physical ai generation and understanding |  [![GitHub](https://img.shields.io/github/stars/SHI-Labs/physical-ai-bench)](https://github.com/SHI-Labs/physical-ai-bench) |
| `PhysVidBench`  | Can Your Model Separate Yolks with a Water Bottle? Benchmarking Physical Commonsense Understanding in Video Generation Models |  [![arXiv](https://img.shields.io/badge/arXiv-2507.15824-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.15824) [![GitHub](https://img.shields.io/github/stars/ensanli/PhysVidBenchCode)](https://github.com/ensanli/PhysVidBenchCode) |
| `PBench`  | PBench: A benchmark for evaluating generative models |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://research.nvidia.com/labs/dir/pbench/) |
| `IntPhys 2`  | IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments |  [![arXiv](https://img.shields.io/badge/arXiv-2506.09849-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.09849) [![GitHub](https://img.shields.io/github/stars/facebookresearch/IntPhys2)](https://github.com/facebookresearch/IntPhys2) |
| `T2VPhysBench`  | T2VPhysBench: A First-Principles Benchmark for Physical Consistency in Text-to-Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2505.00337-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.00337) |
| `PisaBench`  | PISA Experiments: Exploring Physics Post-Training for Video Diffusion Models by Watching Stuff Drop |  [![arXiv](https://img.shields.io/badge/arXiv-2503.09595-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.09595) [![GitHub](https://img.shields.io/github/stars/vision-x-nyu/pisa-experiments)](https://github.com/vision-x-nyu/pisa-experiments) |
| `WISA-32K`  | WISA: World Simulator Assistant for Physics-Aware Text-to-Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2503.08153-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.08153) [![GitHub](https://img.shields.io/github/stars/360CVGroup/WISA)](https://github.com/360CVGroup/WISA) |
| `VideoPhy-2`  | VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2503.06800-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.06800) [![GitHub](https://img.shields.io/github/stars/Hritikbansal/videophy)](https://github.com/Hritikbansal/videophy) |
| `VBench-2.0`  | VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness |  [![arXiv](https://img.shields.io/badge/arXiv-2503.21755-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.21755) [![GitHub](https://img.shields.io/github/stars/Vchitect/VBench)](https://github.com/Vchitect/VBench) |
| `PhyCoBench`  | A Physical Coherence Benchmark for Evaluating Video Generation Models via Optical Flow-guided Frame Prediction |  [![arXiv](https://img.shields.io/badge/arXiv-2502.05503-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.05503) [![GitHub](https://img.shields.io/github/stars/Jeckinchen/PhyCoBench)](https://github.com/Jeckinchen/PhyCoBench) |
| `Physics-IQ`  | Do generative video models understand physical principles? |  [![arXiv](https://img.shields.io/badge/arXiv-2501.09038-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.09038) [![GitHub](https://img.shields.io/github/stars/google-deepmind/physics-IQ-benchmark)](https://github.com/google-deepmind/physics-IQ-benchmark) |
| `PhyGenBench`  | Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2410.05363-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.05363) [![GitHub](https://img.shields.io/github/stars/OpenGVLab/PhyGenBench)](https://github.com/OpenGVLab/PhyGenBench) |
| `PhyBench`  | PhyBench: A Physical Commonsense Benchmark for Evaluating Text-to-Image Models |  [![arXiv](https://img.shields.io/badge/arXiv-2406.11802-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.11802) [![GitHub](https://img.shields.io/github/stars/phybench-official/phybench)](https://github.com/phybench-official/phybench) |
| `VideoPhy`  | VideoPhy: Evaluating Physical Commonsense for Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2406.03520-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.03520) [![GitHub](https://img.shields.io/github/stars/Hritikbansal/videophy)](https://github.com/Hritikbansal/videophy) |
| `Physion++`  | Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties |  [![arXiv](https://img.shields.io/badge/arXiv-2306.15668-b31b1b?logo=arxiv)](https://arxiv.org/abs/2306.15668) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://dingmyu.github.io/physion_v2/) |
| `InfLevel`  | Benchmarking Progress to Infant-Level Physical Reasoning in AI |  [![GitHub](https://img.shields.io/github/stars/allenai/inflevel)](https://github.com/allenai/inflevel) |
| `VoE`  | A Benchmark for Modeling Violation-of-Expectation in Physical Reasoning Across Event Categories |  [![arXiv](https://img.shields.io/badge/arXiv-2111.08826-b31b1b?logo=arxiv)](https://arxiv.org/abs/2111.08826) |
| `Physion`  | Physion: Evaluating Physical Prediction from Vision in Humans and Machines |  [![arXiv](https://img.shields.io/badge/arXiv-2106.08261-b31b1b?logo=arxiv)](https://arxiv.org/abs/2106.08261) [![GitHub](https://img.shields.io/github/stars/cogtoolslab/physics-benchmarking-neurips2021)](https://github.com/cogtoolslab/physics-benchmarking-neurips2021) |
| `CoPhy`  | CoPhy: Counterfactual Learning of Physical Dynamics |  [![arXiv](https://img.shields.io/badge/arXiv-1909.12000-b31b1b?logo=arxiv)](https://arxiv.org/abs/1909.12000)  [![GitHub](https://img.shields.io/github/stars/fabienbaradel/cophy)](https://github.com/fabienbaradel/cophy)|
| `IntPhys`  | IntPhys: A Framework and Benchmark for Visual Intuitive Physics Reasoning |  [![arXiv](https://img.shields.io/badge/arXiv-1803.07616-b31b1b?logo=arxiv)](https://arxiv.org/abs/1803.07616) [![GitHub](https://img.shields.io/github/stars/rronan/IntPhys-Baselines)](https://github.com/rronan/IntPhys-Baselines) |
||


## 2.2. Application-Specific World Modeling


### Embodied AI and Robotics

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `GigaWorld-0`  | GigaWorld-0: World Models as Data Engine to Empower Embodied AI |  [![arXiv](https://img.shields.io/badge/arXiv-2511.19861-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.19861) [![GitHub](https://img.shields.io/github/stars/open-gigaai/giga-world-0)](https://github.com/open-gigaai/giga-world-0) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://giga-world-0.github.io/)|
| `Target-Bench`  | Target-Bench: Can World Models Achieve Mapless Path Planning with Semantic Targets |  [![arXiv](https://img.shields.io/badge/arXiv-2511.17792-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.17792) [![GitHub](https://img.shields.io/github/stars/TUM-AVS/target-bench)](https://github.com/TUM-AVS/target-bench) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://target-bench.github.io/)|
| `WoW`  | WoW: Towards a world omniscient world model through embodied interaction |  [![arXiv](https://img.shields.io/badge/arXiv-2509.22642-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.22642) [![GitHub](https://img.shields.io/github/stars/wow-world-model/wow-world-model)](https://github.com/wow-world-model/wow-world-model) |
| `Meta-World+`  | Meta-World+: An improved, standardized, rl benchmark |  [![arXiv](https://img.shields.io/badge/arXiv-2505.11289-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.11289) [![GitHub](https://img.shields.io/github/stars/Farama-Foundation/Metaworld)](https://github.com/Farama-Foundation/Metaworld) |
| `AgiBot-World`  | AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems |  [![arXiv](https://img.shields.io/badge/arXiv-2503.06669-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.06669) [![GitHub](https://img.shields.io/github/stars/OpenDriveLab/AgiBot-World)](https://github.com/OpenDriveLab/AgiBot-World) |
| `RoboCasa`  | RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots |  [![arXiv](https://img.shields.io/badge/arXiv-2406.02523-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.02523) [![GitHub](https://img.shields.io/github/stars/robocasa/robocasa)](https://github.com/robocasa/robocasa) |
| `DROID`  | DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset |  [![arXiv](https://img.shields.io/badge/arXiv-2403.12945-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.12945) [![GitHub](https://img.shields.io/github/stars/droid-dataset/droid_policy_learning)](https://github.com/droid-dataset/droid_policy_learning) |
| `BEHAVIOR-1K`  | BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation |  [![arXiv](https://img.shields.io/badge/arXiv-2403.09227-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.09227) [![GitHub](https://img.shields.io/github/stars/StanfordVL/BEHAVIOR-1K)](https://github.com/StanfordVL/BEHAVIOR-1K) |
| `MimicGen`  | MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations |  [![arXiv](https://img.shields.io/badge/arXiv-2310.17596-b31b1b?logo=arxiv)](https://arxiv.org/abs/2310.17596) [![GitHub](https://img.shields.io/github/stars/NVlabs/mimicgen)](https://github.com/NVlabs/mimicgen) |
| `OXE`  | Open X-Embodiment: Robotic Learning Datasets and RT-X Models |  [![arXiv](https://img.shields.io/badge/arXiv-2310.08864-b31b1b?logo=arxiv)](https://arxiv.org/abs/2310.08864) [![GitHub](https://img.shields.io/github/stars/google-deepmind/open_x_embodiment)](https://github.com/google-deepmind/open_x_embodiment) |
| `RH20T`  | RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot |  [![arXiv](https://img.shields.io/badge/arXiv-2307.00595-b31b1b?logo=arxiv)](https://arxiv.org/abs/2307.00595) [![GitHub](https://img.shields.io/github/stars/rh20t/rh20t_api)](https://github.com/rh20t/rh20t_api) |
| `VP²`  | A Control-Centric Benchmark for Video Prediction |  [![arXiv](https://img.shields.io/badge/arXiv-2304.13723-b31b1b?logo=arxiv)](https://arxiv.org/abs/2304.13723) [![GitHub](https://img.shields.io/github/stars/s-tian/vp2)](https://github.com/s-tian/vp2) |
| `RT-1`  | RT-1: Robotics Transformer for Real-World Control at Scale |  [![arXiv](https://img.shields.io/badge/arXiv-2212.06817-b31b1b?logo=arxiv)](https://arxiv.org/abs/2212.06817) [![GitHub](https://img.shields.io/github/stars/google-research/robotics_transformer)](https://github.com/google-research/robotics_transformer) |
| `BC-Z`  | BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2202.02005-b31b1b?logo=arxiv)](https://arxiv.org/abs/2202.02005) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://sites.google.com/view/bc-z/home?pli=1) |
| `CALVIN`  | CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks |  [![arXiv](https://img.shields.io/badge/arXiv-2112.03227-b31b1b?logo=arxiv)](https://arxiv.org/abs/2112.03227) [![GitHub](https://img.shields.io/github/stars/mees/calvin)](https://github.com/mees/calvin) |
| `BridgeData V2`  | BridgeData V2: A Dataset for Robot Learning at Scale |  [![arXiv](https://img.shields.io/badge/arXiv-2308.12952-b31b1b?logo=arxiv)](https://arxiv.org/abs/2308.12952) [![GitHub](https://img.shields.io/github/stars/rail-berkeley/bridge_data_v2)](https://github.com/rail-berkeley/bridge_data_v2) |
| `LIBERO`  | LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2306.03310-b31b1b?logo=arxiv)](https://arxiv.org/abs/2306.03310) [![GitHub](https://img.shields.io/github/stars/Lifelong-Robot-Learning/LIBERO)](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| `Isaac Gym`  | Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2108.10470-b31b1b?logo=arxiv)](https://arxiv.org/abs/2108.10470) [![GitHub](https://img.shields.io/github/stars/isaac-sim/IsaacGymEnvs)](https://github.com/isaac-sim/IsaacGymEnvs) |
| `RoboNet`  | RoboNet: Large-Scale Multi-Robot Learning |  [![arXiv](https://img.shields.io/badge/arXiv-1910.11215-b31b1b?logo=arxiv)](https://arxiv.org/abs/1910.11215) [![GitHub](https://img.shields.io/github/stars/SudeepDasari/RoboNet)](https://github.com/SudeepDasari/RoboNet) |
| `Meta-World`  | Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning |  [![arXiv](https://img.shields.io/badge/arXiv-1910.10897-b31b1b?logo=arxiv)](https://arxiv.org/abs/1910.10897) [![GitHub](https://img.shields.io/github/stars/Farama-Foundation/Metaworld)](https://github.com/Farama-Foundation/Metaworld) |
| `RLBench`  | RLBench: The Robot Learning Benchmark & Learning Environment |  [![arXiv](https://img.shields.io/badge/arXiv-1909.12271-b31b1b?logo=arxiv)](https://arxiv.org/abs/1909.12271) [![GitHub](https://img.shields.io/github/stars/stepjam/RLBench)](https://github.com/stepjam/RLBench) |
||

### Autonomous Driving

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `WorldLens`  | WorldLens: Full-Spectrum Evaluations of Driving World Models in Real World |  [![arXiv](https://img.shields.io/badge/arXiv-2512.10958-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.10958) [![GitHub](https://img.shields.io/github/stars/worldbench/WorldLens)](https://github.com/worldbench/WorldLens) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://worldbench.github.io/worldlens)|
| `ACT-Bench`  | ACT-Bench: Towards Action Controllable World Models for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2412.05337-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.05337) [![GitHub](https://img.shields.io/github/stars/turingmotors/ACT-Bench)](https://github.com/turingmotors/ACT-Bench) |
| `DrivingDojo`  | DrivingDojo Dataset: Advancing Interactive and Knowledge-Enriched Driving World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2410.10738-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.10738) [![GitHub](https://img.shields.io/github/stars/Robertwyq/Drivingdojo)](https://github.com/Robertwyq/Drivingdojo) |
| `DriveArena`  | DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2408.00415-b31b1b?logo=arxiv)](https://arxiv.org/abs/2408.00415) [![GitHub](https://img.shields.io/github/stars/PJLab-ADG/DriveArena)](https://github.com/PJLab-ADG/DriveArena) |
| `NAVSIM`  | NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking |  [![arXiv](https://img.shields.io/badge/arXiv-2406.15349-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.15349) [![GitHub](https://img.shields.io/github/stars/autonomousvision/navsim)](https://github.com/autonomousvision/navsim) |
| `CarDreamer`  | CarDreamer: Open-Source Learning Platform for World Model based Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2405.09111-b31b1b?logo=arxiv)](https://arxiv.org/abs/2405.09111) [![GitHub](https://img.shields.io/github/stars/ucd-dare/CarDreamer)](https://github.com/ucd-dare/CarDreamer) |
| `OpenDV-2K`  | GenAD: Generalized Predictive Model for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2403.09630-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.09630) [![GitHub](https://img.shields.io/github/stars/OpenDriveLab/DriveAGI)](https://github.com/OpenDriveLab/DriveAGI) |
| `ZOD`  | Zenseact Open Dataset: A Large-Scale and Diverse Multimodal Dataset for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2305.02008-b31b1b?logo=arxiv)](https://arxiv.org/abs/2305.02008) [![GitHub](https://img.shields.io/github/stars/zenseact/zod)](https://github.com/zenseact/zod) |
| `Occ3D`  | Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2304.14365-b31b1b?logo=arxiv)](https://arxiv.org/abs/2304.14365) [![GitHub](https://img.shields.io/github/stars/Tsinghua-MARS-Lab/Occ3D)](https://github.com/Tsinghua-MARS-Lab/Occ3D) |
| `OpenOccupancy`  | OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic Occupancy Perception |  [![arXiv](https://img.shields.io/badge/arXiv-2303.03991-b31b1b?logo=arxiv)](https://arxiv.org/abs/2303.03991) [![GitHub](https://img.shields.io/github/stars/JeffWang987/OpenOccupancy)](https://github.com/JeffWang987/OpenOccupancy) |
| `Argoverse 2`  | Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting |  [![arXiv](https://img.shields.io/badge/arXiv-2301.00493-b31b1b?logo=arxiv)](https://arxiv.org/abs/2301.00493) [![GitHub](https://img.shields.io/github/stars/argoverse/av2-api)](https://github.com/argoverse/av2-api) |
| `KITTI-360`  | KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D |  [![arXiv](https://img.shields.io/badge/arXiv-2109.13410-b31b1b?logo=arxiv)](https://arxiv.org/abs/2109.13410) [![GitHub](https://img.shields.io/github/stars/autonomousvision/kitti360Scripts)](https://github.com/autonomousvision/kitti360Scripts) |
| `NuPlan`  | NuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles |  [![arXiv](https://img.shields.io/badge/arXiv-2106.11810-b31b1b?logo=arxiv)](https://arxiv.org/abs/2106.11810) [![GitHub](https://img.shields.io/github/stars/motional/nuplan-devkit)](https://github.com/motional/nuplan-devkit) |
| `ONCE`  | One Million Scenes for Autonomous Driving: ONCE Dataset |  [![arXiv](https://img.shields.io/badge/arXiv-2106.11037-b31b1b?logo=arxiv)](https://arxiv.org/abs/2106.11037) [![GitHub](https://img.shields.io/github/stars/PointsCoder/ONCE_Benchmark)](https://github.com/PointsCoder/ONCE_Benchmark) |
| `WOMD`  | Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset |  [![arXiv](https://img.shields.io/badge/arXiv-2104.10133-b31b1b?logo=arxiv)](https://arxiv.org/abs/2104.10133) [![GitHub](https://img.shields.io/github/stars/waymo-research/waymo-open-dataset)](https://github.com/waymo-research/waymo-open-dataset) |
| `Lyft Level 5`  | One Thousand and One Hours: Self-driving Motion Prediction Dataset |  [![arXiv](https://img.shields.io/badge/arXiv-2006.14480-b31b1b?logo=arxiv)](https://arxiv.org/abs/2006.14480) [![GitHub](https://img.shields.io/github/stars/JerryIshihara/lyft-motion-prediction-for-autonomous-vehicle)](https://github.com/JerryIshihara/lyft-motion-prediction-for-autonomous-vehicle) |
| `A2D2`  | A2D2: Audi Autonomous Driving Dataset |  [![arXiv](https://img.shields.io/badge/arXiv-2004.06320-b31b1b?logo=arxiv)](https://arxiv.org/abs/2004.06320) [![GitHub](https://img.shields.io/github/stars/tum-gis/a2d2_ros_preparer)](https://github.com/tum-gis/a2d2_ros_preparer) |
| `Waymo`  | Scalability in Perception for Autonomous Driving: Waymo Open Dataset |  [![arXiv](https://img.shields.io/badge/arXiv-1912.04838-b31b1b?logo=arxiv)](https://arxiv.org/abs/1912.04838) [![GitHub](https://img.shields.io/github/stars/waymo-research/waymo-open-dataset)](https://github.com/waymo-research/waymo-open-dataset) |
| `Argoverse`  | Argoverse: 3D Tracking and Forecasting With Rich Maps |  [![arXiv](https://img.shields.io/badge/arXiv-1911.02620-b31b1b?logo=arxiv)](https://arxiv.org/abs/1911.02620) [![GitHub](https://img.shields.io/github/stars/argoverse/argoverse-api)](https://github.com/argoverse/argoverse-api) |
| `INTERACTION`  | INTERACTION Dataset: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps |  [![arXiv](https://img.shields.io/badge/arXiv-1910.03088-b31b1b?logo=arxiv)](https://arxiv.org/abs/1910.03088) [![GitHub](https://img.shields.io/github/stars/interaction-dataset/interaction-dataset)](https://github.com/interaction-dataset/interaction-dataset) |
| `SemanticKITTI`  | SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences |  [![arXiv](https://img.shields.io/badge/arXiv-1904.01416-b31b1b?logo=arxiv)](https://arxiv.org/abs/1904.01416) [![GitHub](https://img.shields.io/github/stars/PRBonn/semantic-kitti-api)](https://github.com/PRBonn/semantic-kitti-api) |
| `nuScenes`  | nuScenes: A Multimodal Dataset for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-1903.11027-b31b1b?logo=arxiv)](https://arxiv.org/abs/1903.11027) [![GitHub](https://img.shields.io/github/stars/nutonomy/nuscenes-devkit)](https://github.com/nutonomy/nuscenes-devkit) |
| `BDD100K`  | BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning |  [![arXiv](https://img.shields.io/badge/arXiv-1805.04687-b31b1b?logo=arxiv)](https://arxiv.org/abs/1805.04687) [![GitHub](https://img.shields.io/github/stars/bdd100k/bdd100k)](https://github.com/bdd100k/bdd100k) |
| `ApolloScape`  | The ApolloScape Dataset for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-1803.06184-b31b1b?logo=arxiv)](https://arxiv.org/abs/1803.06184) [![GitHub](https://img.shields.io/github/stars/ApolloScapeAuto/dataset-api)](https://github.com/ApolloScapeAuto/dataset-api) |
| `CARLA`  | CARLA: An Open Urban Driving Simulator |  [![arXiv](https://img.shields.io/badge/arXiv-1711.03938-b31b1b?logo=arxiv)](https://arxiv.org/abs/1711.03938) [![GitHub](https://img.shields.io/github/stars/carla-simulator/carla)](https://github.com/carla-simulator/carla) |
| `KITTI`  | Are we ready for autonomous driving? The KITTI vision benchmark suite |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://ieeexplore.ieee.org/document/6248074) |
||

### Interactive Environments and Gaming

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `Matrix-Game-MC`  | Matrix-Game: Interactive World Foundation Model |  [![arXiv](https://img.shields.io/badge/arXiv-2506.18701-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.18701) [![GitHub](https://img.shields.io/github/stars/SkyworkAI/Matrix-Game)](https://github.com/SkyworkAI/Matrix-Game) |
| `LOOPNAV`  | Toward Memory-Aided World Models: Benchmarking via Spatial Consistency |  [![arXiv](https://img.shields.io/badge/arXiv-2505.22976-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.22976) [![GitHub](https://img.shields.io/github/stars/Kevin-lkw/LoopNav)](https://github.com/Kevin-lkw/LoopNav) |
| `JARVIS-VLA`  | JARVIS-VLA: Post-Training Large-Scale Visual Language Models to Play Visual Games with Keyboards and Mouse |  [![arXiv](https://img.shields.io/badge/arXiv-2503.16365-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.16365) [![GitHub](https://img.shields.io/github/stars/CraftJarvis/JarvisVLA)](https://github.com/CraftJarvis/JarvisVLA) |
| `GF-Minecraft`  | GameFactory: Creating New Games with Generative Interactive Videos |  [![arXiv](https://img.shields.io/badge/arXiv-2501.08325-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.08325) [![GitHub](https://img.shields.io/github/stars/KwaiVGI/GameFactory)](https://github.com/KwaiVGI/GameFactory) |
| `The Matrix`  | The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control |  [![arXiv](https://img.shields.io/badge/arXiv-2412.03568-b31b1b?logo=arxiv)](https://arxiv.org/abs/2412.03568) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://thematrix1999.github.io/) |
| `GameGen-X`  | GameGen-X: Interactive Open-world Game Video Generation |  [![arXiv](https://img.shields.io/badge/arXiv-2411.00769-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.00769) [![GitHub](https://img.shields.io/github/stars/GameGen-X/GameGen-X)](https://github.com/GameGen-X/GameGen-X) |
| `Mars`  | Mars: Situated Inductive Reasoning in an Open-World Environment |  [![arXiv](https://img.shields.io/badge/arXiv-2410.08126-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.08126) [![GitHub](https://img.shields.io/github/stars/XiaojuanTang/Mars)](https://github.com/XiaojuanTang/Mars) |
| `Crafter`  | Benchmarking the Spectrum of Agent Capabilities |  [![arXiv](https://img.shields.io/badge/arXiv-2109.06780-b31b1b?logo=arxiv)](https://arxiv.org/abs/2109.06780) [![GitHub](https://img.shields.io/github/stars/danijar/crafter)](https://github.com/danijar/crafter) |
| `CS-Deathmatch`  | Counter-Strike Deathmatch with Large-Scale Behavioural Cloning |  [![arXiv](https://img.shields.io/badge/arXiv-2104.04258-b31b1b?logo=arxiv)](https://arxiv.org/abs/2104.04258) [![GitHub](https://img.shields.io/github/stars/TeaPearce/Counter-Strike_Behavioural_Cloning)](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning) |
| `MineRL`  | The MineRL 2020 Competition on Sample Efficient Reinforcement Learning using Human Priors |  [![arXiv](https://img.shields.io/badge/arXiv-2101.11071-b31b1b?logo=arxiv)](https://arxiv.org/abs/2101.11071) [![GitHub](https://img.shields.io/github/stars/minerllabs/minerl)](https://github.com/minerllabs/minerl) |
| `NLE`  | The NetHack Learning Environment |  [![arXiv](https://img.shields.io/badge/arXiv-2006.13760-b31b1b?logo=arxiv)](https://arxiv.org/abs/2006.13760) [![GitHub](https://img.shields.io/github/stars/facebookresearch/nle)](https://github.com/facebookresearch/nle) |
| `Procgen`  | Leveraging Procedural Generation to Benchmark Reinforcement Learning |  [![arXiv](https://img.shields.io/badge/arXiv-1912.01588-b31b1b?logo=arxiv)](https://arxiv.org/abs/1912.01588) [![GitHub](https://img.shields.io/github/stars/openai/train-procgen)](https://github.com/openai/train-procgen) |
| `DMC`  | DeepMind Control Suite |  [![arXiv](https://img.shields.io/badge/arXiv-1801.00690-b31b1b?logo=arxiv)](https://arxiv.org/abs/1801.00690) [![GitHub](https://img.shields.io/github/stars/google-deepmind/dm_control)](https://github.com/google-deepmind/dm_control) |
| `ALE`  | The Arcade Learning Environment: An Evaluation Platform for General Agents |  [![arXiv](https://img.shields.io/badge/arXiv-1207.4708-b31b1b?logo=arxiv)](https://arxiv.org/abs/1207.4708) [![GitHub](https://img.shields.io/github/stars/Farama-Foundation/Arcade-Learning-Environment)](https://github.com/Farama-Foundation/Arcade-Learning-Environment) |
||

# 3. Others

###  Survey

> :timer_clock: In chronological order, from the latest to the earliest.

| Paper | Link | 
|:-|:-:|
||
| Progressive Robustness-Aware World Models in Autonomous Driving: A Review and Outlook |  [![![TechrXiv](https://img.shields.io/badge/T-TechRxiv-blue)](https://www.techrxiv.org/users/1003906/articles/1364209-progressive-robustness-aware-world-models-in-autonomous-driving-a-review-and-outlook) |
| Simulating the Visual World with Artificial Intelligence: A Roadmap |  [![arXiv](https://img.shields.io/badge/arXiv-2511.08585-b31b1b?logo=arxiv)](http://arxiv.org/abs/2511.08585) [![GitHub](https://img.shields.io/github/stars/ziqihuangg/Awesome-From-Video-Generation-to-World-Model)](https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model) |
| A Step Toward World Models: A Survey on Robotic Manipulation |  [![arXiv](https://img.shields.io/badge/arXiv-2511.02097-b31b1b?logo=arxiv)](http://arxiv.org/abs/2511.02097) |
| The Safety Challenge of World Models for Embodied AI Agents: A Review |  [![arXiv](https://img.shields.io/badge/arXiv-2510.05865-b31b1b?logo=arxiv)](http://arxiv.org/abs/2510.05865) |
| A Comprehensive Survey on World Models for Embodied AI |  [![arXiv](https://img.shields.io/badge/arXiv-2510.16732-b31b1b?logo=arxiv)](https://www.arxiv.org/abs/2510.16732) [![GitHub](https://img.shields.io/github/stars/Li-Zn-H/AwesomeWorldModels)](https://github.com/Li-Zn-H/AwesomeWorldModels) |
| 3D and 4D World Modeling: A Survey |  [![arXiv](https://img.shields.io/badge/arXiv-2509.07996-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.07996) [![GitHub](https://img.shields.io/github/stars/worldbench/survey)](https://github.com/worldbench/survey) |
| Large Model Empowered Embodied AI: A Survey on Decision-Making and Embodied Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2508.10399-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.10399) |
| A Survey: Learning Embodied Intelligence from Physical Simulators and World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2507.00917-b31b1b?logo=arxiv)](http://arxiv.org/abs/2507.00917) [![GitHub](https://img.shields.io/github/stars/NJU3DV-LoongGroup/Embodied-World-Models-Survey)](https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey) |
| From 2D to 3D Cognition: A Brief Survey of General World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2506.20134-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.20134) |
| A Survey on World Models Grounded in Acoustic Physical Information |  [![arXiv](https://img.shields.io/badge/arXiv-2506.13833-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.13833) [![GitHub](https://img.shields.io/github/stars/soundai2016/survey_acoustic_world_models)](https://github.com/soundai2016/survey_acoustic_world_models) |
| World Models in Artificial Intelligence: Sensing, Learning, and Reasoning Like a Child |  [![arXiv](https://img.shields.io/badge/arXiv-2503.15168-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.15168) |
| The Role of World Models in Shaping Autonomous Driving: A Comprehensive Survey |  [![arXiv](https://img.shields.io/badge/arXiv-2502.10498-b31b1b?logo=arxiv)](https://arxiv.org/abs/2502.10498) [![GitHub](https://img.shields.io/github/stars/LMD0311/Awesome-World-Model)](https://github.com/LMD0311/Awesome-World-Model) |
| A Survey of World Models for Autonomous Driving |  [![arXiv](https://img.shields.io/badge/arXiv-2501.11260-b31b1b?logo=arxiv)](https://arxiv.org/abs/2501.11260) [![GitHub](https://img.shields.io/github/stars/FengZicai/WMAD-Benchmarks)](https://github.com/FengZicai/WMAD-Benchmarks) |
| Understanding World or Predicting Future? A Comprehensive Survey of World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2411.14499-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.14499) [![GitHub](https://img.shields.io/github/stars/tsinghua-fib-lab/World-Model)](https://github.com/tsinghua-fib-lab/World-Model) |
| Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond |  [![arXiv](https://img.shields.io/badge/arXiv-2405.03520-b31b1b?logo=arxiv)](https://arxiv.org/abs/2405.03520) [![GitHub](https://img.shields.io/github/stars/GigaAI-research/General-World-Models-Survey)](https://github.com/GigaAI-research/General-World-Models-Survey) |
| World Models for Autonomous Driving: An Initial Survey |  [![arXiv](https://img.shields.io/badge/arXiv-2403.02622-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.02622) |
| A Path Towards Autonomous Machine Intelligence | [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://yschoe.github.io/2022/08/24/LeCun-A-path-towards-autonomous-machine-intelligence.html) |
||

###  GitHub Repo
This part overlaps somewhat with the `Survey` section.


| Repo | Link |
|:-|:-:|
||
| Awesome From Video Generation to World Model | [![GitHub](https://img.shields.io/github/stars/ziqihuangg/Awesome-From-Video-Generation-to-World-Model)](https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model) |
| A Curated List of Amazing Works in World Modeling, spanning applications in Embodied AI, Autonomous Driving, Natural Laguage Processing and Agents | [![GitHub](https://img.shields.io/github/stars/knightnemo/Awesome-World-Models)](https://github.com/knightnemo/Awesome-World-Models) |
| A curated list of papers for World Models for General Video Generation, Embodied AI, and Autonomous Driving | [![GitHub](https://img.shields.io/github/stars/leofan90/Awesome-World-Models)](https://github.com/leofan90/Awesome-World-Models) |
| World Models for Autonomous Driving (and Robotic) papers | [![GitHub](https://img.shields.io/github/stars/LMD0311/Awesome-World-Model)](https://github.com/LMD0311/Awesome-World-Model) |
| Awesome 3D and 4D World Models | [![GitHub](https://img.shields.io/github/stars/worldbench/survey)](https://github.com/worldbench/survey) |
| Simulating the Real World: Survey & Resources | [![GitHub](https://img.shields.io/github/stars/ALEEEHU/World-Simulator)](https://github.com/ALEEEHU/World-Simulator) |
| A curated list of world models for autonomous driving | [![GitHub](https://img.shields.io/github/stars/HaoranZhuExplorer/World-Models-Autonomous-Driving-Latest-Survey)](https://github.com/HaoranZhuExplorer/World-Models-Autonomous-Driving-Latest-Survey) |
| A curated list of resources related to World Models for Autonomous Driving (WMAD) | [![GitHub](https://img.shields.io/github/stars/FengZicai/AwesomeWMAD)](https://github.com/FengZicai/AwesomeWMAD) |
| A Survey: Learning Embodied Intelligence from Physical Simulators and World Models | [![GitHub](https://img.shields.io/github/stars/NJU3DV-LoongGroup/Embodied-World-Models-Survey)](https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey) |
| A Comprehensive Survey on World Models for Embodied AI | [![GitHub](https://img.shields.io/github/stars/Li-Zn-H/AwesomeWorldModels)](https://github.com/Li-Zn-H/AwesomeWorldModels) |
| A Survey on World Models Grounded in Acoustic Physical Information | [![GitHub](https://img.shields.io/github/stars/soundai2016/survey_acoustic_world_models)](https://github.com/soundai2016/survey_acoustic_world_models) |
| A Comprehensive Survey on General World Models and Beyond | [![GitHub](https://img.shields.io/github/stars/GigaAI-research/General-World-Models-Survey)](https://github.com/GigaAI-research/General-World-Models-Survey) |
| Understanding World or Predicting Future? A Comprehensive Survey of World Models | [![GitHub](https://img.shields.io/github/stars/tsinghua-fib-lab/World-Model)](https://github.com/tsinghua-fib-lab/World-Model) |
| From Masks to Worlds: A Hitchhiker's Guide to World Models | [![GitHub](https://img.shields.io/github/stars/M-E-AGI-Lab/Awesome-World-Models)](https://github.com/M-E-AGI-Lab/Awesome-World-Models) |
||

### Workshop

| Venue  | Workshop | Link | 
|:-:|:-|:-:|
||
| `ICLR 2026` | Workshop on World Models: Understanding, Modelling and Scaling |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://sites.google.com/view/iclr-2026-workshop-world-model/home)|
| `NeurIPS 2025` | Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://sites.google.com/view/law-2025)|
| `NeurIPS 2025` | Embodied World Models for Decision Making |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://embodied-world-models.github.io/)|
| `ICCV 2025` | Reliable and Interactable World Models: Geometry, Physics, Interactivity and Real-World Generalization |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://riwm-2025.github.io/RIWM-2025/)|
| `ICML 2025` | Building Physically Plausible World Models |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://physical-world-modeling.github.io/)|
| `CVPR 2025` | Benchmarking World Models |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://worldmodelbench.github.io/)|
| `ICLR 2025` | World Models: Understanding, Modelling, and Scaling |  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://sites.google.com/view/worldmodel-iclr2025/)|





###  Theory

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `-` | What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? |  [![arXiv](https://img.shields.io/badge/arXiv-2512.24497-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.24497) |
| `-` | Closing the Train-Test Gap in World Models for Gradient-Based Planning |  [![arXiv](https://img.shields.io/badge/arXiv-2512.09929-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.09929) [![GitHub](https://img.shields.io/github/stars/qw3rtman/robust-world-model-planning)](https://github.com/qw3rtman/robust-world-model-planning)|
| `WWM` | Web World Model |  [![arXiv](https://img.shields.io/badge/arXiv-2512.23676-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.23676) [![GitHub](https://img.shields.io/github/stars/Princeton-AI2-Lab/Web-World-Models)](https://github.com/Princeton-AI2-Lab/Web-World-Models) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://princeton-ai2-lab.github.io/Web-World-Models/)|
| `DWM` | Dexterous World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2512.17907-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.17907) [![GitHub](https://img.shields.io/github/stars/snuvclab/dwm)](https://github.com/snuvclab/dwm) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://snuvclab.github.io/dwm/)|
| `PRISM-WM` | Prismatic World Model: Learning Compositional Dynamics for Planning in Hybrid Systems |  [![arXiv](https://img.shields.io/badge/arXiv-2512.08411-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.08411) |
| `-` | From Word to World: Can Large Language Models be Implicit Text-based World Models? |  [![arXiv](https://img.shields.io/badge/arXiv-2512.18832-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.18832) [![GitHub](https://img.shields.io/github/stars/X1AOX1A/Word2World)](https://github.com/X1AOX1A/Word2World)|
| `-` | Beyond World Models: Rethinking Understanding in AI Models |  [![arXiv](https://img.shields.io/badge/arXiv-2511.12239-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.12239) |
| `Cambrian-S` | Cambrian-S: Towards Spatial Supersensing in Video |  [![arXiv](https://img.shields.io/badge/arXiv-2511.04670-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.04670)  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://cambrian-mllm.github.io/)|
| `-` | World Models Should Prioritize the Unification of Physical and Social Dynamics |  [![arXiv](https://img.shields.io/badge/arXiv-2510.21219-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.21219) |
| `GRWM` | Clone Deterministic 3D Worlds with Geometrically-Regularized World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2510.26782-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.26782)  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://github.com/M-E-AGI-Lab/Awesome-World-Models)|
| `-` | From Masks to Worlds: A Hitchhiker’s Guide to World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2510.20668-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.20668) |
| `AutumnBench` | Benchmarking World-Model Learning |  [![arXiv](https://img.shields.io/badge/arXiv-2510.19788-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.19788)|
| `S3AP` | Social World Models |  [![arXiv](https://img.shields.io/badge/arXiv-2509.00559-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.00559) |
| `UniVid` | UniVid: Unifying Vision Tasks with Pre-trained Video Generation Models |  [![arXiv](https://img.shields.io/badge/arXiv-2509.21760-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.21760) [![GitHub](https://img.shields.io/github/stars/CUC-MIPG/UniVid)](https://github.com/CUC-MIPG/UniVid) |
| `Veo 3`| Video models are zero-shot learners and reasoners | [![arXiv](https://img.shields.io/badge/arXiv-2509.20328-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.20328)  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://video-zero-shot.github.io/) |
| `PSI`| World Modeling with Probabilistic Structure Integration |  [![arXiv](https://img.shields.io/badge/arXiv-2509.09737-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.09737) |
|`-`| What Does it Mean for a Neural Network to Learn a “World Model”? | [![arXiv](https://img.shields.io/badge/arXiv-2507.21513-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.21513)|
| `PAN`| Critiques of World Models | [![arXiv](https://img.shields.io/badge/arXiv-2507.05169-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.05169)|
|`-`| What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models |[![arXiv](https://img.shields.io/badge/arXiv-2507.06952-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.06952) [![GitHub](https://img.shields.io/github/stars/keyonvafa/inductive-bias-probes)](https://github.com/keyonvafa/inductive-bias-probes)|
|`PIN-WM`| PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation |[![arXiv](https://img.shields.io/badge/arXiv-2504.16693-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.16693) [![GitHub](https://img.shields.io/github/stars/XuAdventurer/PIN-WM)](https://github.com/XuAdventurer/PIN-WM)|
| `-`| Position: Interactive Generative Video as Next-Generation Game Engine | [![arXiv](https://img.shields.io/badge/arXiv-2503.17359-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.17359)|
|`-`| Four Principles for Physically Interpretable World Models |[![arXiv](https://img.shields.io/badge/arXiv-2503.02143-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.02143) [![GitHub](https://img.shields.io/github/stars/trustworthy-engineered-autonomy-lab/piwm-principles)](https://github.com/trustworthy-engineered-autonomy-lab/piwm-principles)|
|`-`| How Far is Video Generation from World Model: A Physical Law Perspective | [![arXiv](https://img.shields.io/badge/arXiv-2411.02385-b31b1b?logo=arxiv)](https://arxiv.org/abs/2411.02385) [![GitHub](https://img.shields.io/github/stars/phyworld/phyworld)](https://github.com/phyworld/phyworld)|
|`CWM`| Unifying (Machine) Vision via Counterfactual World Modeling | [![arXiv](https://img.shields.io/badge/arXiv-2306.01828-b31b1b?logo=arxiv)](https://arxiv.org/abs/2306.01828)|
||

###  World Models for Downstream Tasks

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
| `Act2Goal`| Act2Goal: From World Model To General Goal-conditioned Policy | [![arXiv](https://img.shields.io/badge/arXiv-2512.23541-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.23541) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://act2goal.github.io/) |
| `DexWM`| World Models Can Leverage Human Videos for Dexterous Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2512.13644-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.13644) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://raktimgg.github.io/dexwm/)|
| `WorldLM`| Can World Models Benefit VLMs for World Dynamics? | [![arXiv](https://img.shields.io/badge/arXiv-2510.00855-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.00855) [![GitHub](https://img.shields.io/github/stars/zyzkevin/dyva-worldlm)](https://github.com/zyzkevin/dyva-worldlm) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://dyva-worldlm.github.io/)|
| `WMPO`| WMPO: World Model-based Policy Optimization for Vision-Language-Action Models | [![arXiv](https://img.shields.io/badge/arXiv-2511.09515-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.09515) [![GitHub](https://img.shields.io/github/stars/WM-PO/WMPO)](https://github.com/WM-PO/WMPO) |
| `PhysWorld`| Robot Learning from a Physical World Model | [![arXiv](https://img.shields.io/badge/arXiv-2511.07416-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.07416)  |
| `-`| Scalable Policy Evaluation with Video World Models | [![arXiv](https://img.shields.io/badge/arXiv-2511.11520-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.11520) |
| `NORA-1.5`| NORA-1.5: A Vision-Language-Action Model Trained using World Model and Action-based Preference Rewards | [![arXiv](https://img.shields.io/badge/arXiv-2511.14659-b31b1b?logo=arxiv)](https://arxiv.org/abs/2511.14659) [![GitHub](https://img.shields.io/github/stars/declare-lab/nora-1.5)](https://github.com/declare-lab/nora-1.5) |
| `GigaBrain-0`| GigaBrain-0: A World Model-Powered Vision-Language Action Model | [![arXiv](https://img.shields.io/badge/arXiv-2510.19430-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.19430) [![GitHub](https://img.shields.io/github/stars/open-gigaai/giga-brain-0)](https://github.com/open-gigaai/giga-brain-0) |
| `DriveVLA-W0`| DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving | [![arXiv](https://img.shields.io/badge/arXiv-2510.12796-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.12796) [![GitHub](https://img.shields.io/github/stars/BraveGroup/DriveVLA-W0)](https://github.com/BraveGroup/DriveVLA-W0) |
| `Ctrl-World`| Ctrl-World: A Controllable Generative World Model for Robot Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2510.10125-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.10125) [![GitHub](https://img.shields.io/github/stars/Robert-gyj/Ctrl-World)](https://github.com/Robert-gyj/Ctrl-World) |
| `WorldLM`| Can World Models Benefit VLMs for World Dynamics? | [![arXiv](https://img.shields.io/badge/arXiv-2510.00855-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.00855) [![GitHub](https://img.shields.io/github/stars/zyzkevin/dyva-worldlm)](https://github.com/zyzkevin/dyva-worldlm) |
| `World-Env`| World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training | [![arXiv](https://img.shields.io/badge/arXiv-2509.24948-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.24948) [![GitHub](https://img.shields.io/github/stars/amap-cvlab/world-env)](https://github.com/amap-cvlab/world-env)|
| `GVF-TAPE`| Generative Visual Foresight Meets Task-Agnostic Pose Estimation in Robotic Table-Top Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2509.00361-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.00361) [![GitHub](https://img.shields.io/github/stars/Xiaoxiongzzzz/GVF-TAPE-Video-Generation)](https://github.com/Xiaoxiongzzzz/GVF-TAPE-Video-Generation) |
| `DiWA`| DiWA: Diffusion Policy Adaptation with World Models | [![arXiv](https://img.shields.io/badge/arXiv-2508.03645-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.03645) [![GitHub](https://img.shields.io/github/stars/acl21/diwa)](https://github.com/acl21/diwa) |
| `RIGVid`| Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations | [![arXiv](https://img.shields.io/badge/arXiv-2507.00990-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.00990) [![GitHub](https://img.shields.io/github/stars/shivanshpatel35/rigvid)](https://github.com/shivanshpatel35/rigvid) |
| `KL-tracing`| Taming generative video models for zero-shot optical flow extraction | [![arXiv](https://img.shields.io/badge/arXiv-2507.09082-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.09082) [![GitHub](https://img.shields.io/github/stars/neuroailab/kl_tracing)](https://github.com/neuroailab/kl_tracing) |
| `MindJourney`| MindJourney: Test-Time Scaling with World Models for Spatial Reasoning | [![arXiv](https://img.shields.io/badge/arXiv-2507.12508-b31b1b?logo=arxiv)](https://arxiv.org/abs/2507.12508) [![GitHub](https://img.shields.io/github/stars/UMass-Embodied-AGI/MindJourney)](https://github.com/UMass-Embodied-AGI/MindJourney) |
|`Cosmos-Drive-Dreams`| Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models | [![arXiv](https://img.shields.io/badge/arXiv-2506.09042-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.09042) [![GitHub](https://img.shields.io/github/stars/nv-tlabs/Cosmos-Drive-Dreams)](https://github.com/nv-tlabs/Cosmos-Drive-Dreams) |
|`WoMAP`| WoMAP: World Models For Embodied Open-Vocabulary Object Localization | [![arXiv](https://img.shields.io/badge/arXiv-2506.01600-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.01600) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://robot-womap.github.io/) |
|`-`| General agents contain world models | [![arXiv](https://img.shields.io/badge/arXiv-2506.01622-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.01622) |
|`WorldGym`| WorldGym: World Model as An Environment for Policy Evaluation | [![arXiv](https://img.shields.io/badge/arXiv-2506.00613-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.00613) [![GitHub](https://img.shields.io/github/stars/world-model-eval/world-model-eval)](https://github.com/world-model-eval/world-model-eval) |
|`-`| World Models for Anomaly Detection during Model-Based Reinforcement Learning Inference | [![arXiv](https://img.shields.io/badge/arXiv-2503.02552-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.02552) |
| `WorldEval`| WorldEval: World Model as Real-World Robot Policies Evaluator |  [![arXiv](https://img.shields.io/badge/arXiv-2505.19017-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.19017) [![GitHub](https://img.shields.io/github/stars/liyaxuanliyaxuan/Worldeval)](https://github.com/liyaxuanliyaxuan/Worldeval)|
|`DEMO3`| Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning | [![arXiv](https://img.shields.io/badge/arXiv-2503.01837-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.01837) [![GitHub](https://img.shields.io/github/stars/adrialopezescoriza/demo3)](https://github.com/adrialopezescoriza/demo3) |
|`ReconDreamer`| ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration | [![arXiv](https://img.shields.io/badge/arXiv-2411.19548-b31b1b?logo=arxiv)](http://arxiv.org/abs/2411.19548) [![GitHub](https://img.shields.io/github/stars/GigaAI-research/ReconDreamer)](https://github.com/GigaAI-research/ReconDreamer) |
|`DriveDreamer4D`| DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation | [![arXiv](https://img.shields.io/badge/arXiv-2410.13571-b31b1b?logo=arxiv)](http://arxiv.org/abs/2410.13571) [![GitHub](https://img.shields.io/github/stars/GigaAI-research/DriveDreamer4D)](https://github.com/GigaAI-research/DriveDreamer4D) |
|`JOWA`| Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining | [![arXiv](https://img.shields.io/badge/arXiv-2410.00564-b31b1b?logo=arxiv)](https://arxiv.org/abs/2410.00564) [![GitHub](https://img.shields.io/github/stars/CJReinforce/JOWA)](https://github.com/CJReinforce/JOWA) |
|`PWM`| PWM: Policy Learning with Multi-Task World Models | [![arXiv](https://img.shields.io/badge/arXiv-2407.02466-b31b1b?logo=arxiv)](https://arxiv.org/abs/2407.02466) [![GitHub](https://img.shields.io/github/stars/imgeorgiev/PWM)](https://github.com/imgeorgiev/PWM) |
|`VLP`| Video Language Planning | [![arXiv](https://img.shields.io/badge/arXiv-2310.10625-b31b1b?logo=arxiv)](https://arxiv.org/abs/2310.10625) [![GitHub](https://img.shields.io/github/stars/video-language-planning/vlp_code)](https://github.com/video-language-planning/vlp_code) |
|`Surfer`| Surfer: Progressive Reasoning with World Models for Robotic Manipulation | [![arXiv](https://img.shields.io/badge/arXiv-2306.11335-b31b1b?logo=arxiv)](https://arxiv.org/abs/2306.11335) [![GitHub](https://img.shields.io/github/stars/pzhren/Surfer)](https://github.com/pzhren/Surfer) |
||

### Other Perspectives of World Modeling

> :timer_clock: In chronological order, from the latest to the earliest.

| Design  | Paper | Link | 
|:-:|:-|:-:|
||
|`Motus`|Motus: A Unified Latent Action World Model|[![arXiv](https://img.shields.io/badge/arXiv-2512.13030-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.13030) [![GitHub](https://img.shields.io/github/stars/thu-ml/Motus)](https://github.com/thu-ml/Motus) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://motus-robotics.github.io/motus)|
|`Visionary`|Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform|[![arXiv](https://img.shields.io/badge/arXiv-2512.08478-b31b1b?logo=arxiv)](https://arxiv.org/abs/2512.08478) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://visionary-laboratory.github.io/visionary/)|
|`R-WoM`|R-WoM: Retrieval-augmented World Model For Computer-use Agents|[![arXiv](https://img.shields.io/badge/arXiv-2510.11892-b31b1b?logo=arxiv)](https://arxiv.org/abs/2510.11892)|
|`-`|Context and Diversity Matter: The Emergence of In-Context Learning in World Models|[![arXiv](https://img.shields.io/badge/arXiv-2509.22353-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.22353) [![GitHub](https://img.shields.io/github/stars/airs-cuhk/airsoul)](https://github.com/airs-cuhk/airsoul)|
|`UnifoLM-WMA-0`|UnifoLM-WMA-0: A World-Model-Action (WMA) Framework under UnifoLM Family|[![GitHub](https://img.shields.io/github/stars/unitreerobotics/unifolm-world-model-action)](https://github.com/unitreerobotics/unifolm-world-model-action)  [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://unigen-x.github.io/unifolm-world-model-action.github.io/)|
|`FantasyWorld`|FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction|[![arXiv](https://img.shields.io/badge/arXiv-2509.21657-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.21657) |
|`KeyWorld`|KeyWorld: Key Frame Reasoning Enables Effective and Efficient World Models|[![arXiv](https://img.shields.io/badge/arXiv-2509.21027-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.21027) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://anonymous.4open.science/r/Keyworld-E43D/) |
|`HERO`|HERO: Hierarchical Extrapolation and Refresh for Efficient World Models|[![arXiv](https://img.shields.io/badge/arXiv-2508.17588-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.17588)|
|`ReOI`|Reimagination with Test-time Observation Interventions: Distractor-Robust World Model Predictions for Visual Model Predictive Control|[![arXiv](https://img.shields.io/badge/arXiv-2506.16565-b31b1b?logo=arxiv)](https://arxiv.org/abs/2506.16565)
|`Vid2World`|Vid2World: Crafting Video Diffusion Models to Interactive World Models|[![arXiv](https://img.shields.io/badge/arXiv-2505.14357-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.14357) [![Website](https://img.shields.io/badge/Link-yellow?logo=gitbook)](https://knightnemo.github.io/vid2world/) |
|`SWIFT`|Can Test-Time Scaling Improve World Foundation Model|[![arXiv](https://img.shields.io/badge/arXiv-2503.24320-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.24320) [![GitHub](https://img.shields.io/github/stars/VITA-Group/WFM-TTS)](https://github.com/VITA-Group/WFM-TTS)|
||




# 4. Acknowledgements
This template is inspired by [3D and 4D World Modeling: A Survey](https://github.com/worldbench/survey) and [Simulating the Real World: Survey & Resources](https://github.com/ALEEEHU/World-Simulator).
