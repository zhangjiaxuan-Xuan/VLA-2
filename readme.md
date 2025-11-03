<div align="center">

# VLA<sup>2</sup>: Empowering Vision-Language-Action Models with an Agentic Framework for Unseen Concept Manipulation

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2&2.3-orange.svg)](https://pytorch.org/)
[![VLA](https://img.shields.io/badge/VLA-Vision--Language--Action-green.svg)]()
[![Agent](https://img.shields.io/badge/Agent-Robotics-red.svg)]()
[![LIBERO](https://img.shields.io/badge/LIBERO-Environment-purple.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

</div>

## 📄 Paper & Resources
- 📝 **Paper**: https://arxiv.org/abs/2510.14902
- 🌐 **Project Page**: https://vla-2.github.io

## 📣 News
- 10.27.25 initial upload.
- 11.03.25 update Deployment.

## 📁 Project Structure

```
VLA-2/
├── experiments/                    # Main experimental codes
│   ├── robot/                    # Core VLA-2 implementation
│   │   ├── openvla_utils.py      # OpenVLA utility functions
│   │   ├── robot_utils.py        # Robot interaction utilities
│   │   └── libero_run/           # Main scripts for LIBERO environment
│   │       ├── main_agent_clean.py        # 🎯 Main execution script, use client to get service from vision_planner_service
│   │       ├── vision_planner_service.py  # Vision & planning service
│   │       ├── qwenvl.py                  # Verification module wrapper
│   │       ├── libero_utils.py            # LIBERO environment utilities
│   │       ├── regenerate_libero_dataset.py  # Dataset regeneration
│   │       ├── mps_start.sh               # Multi-process service start
│   │       └── mps_stop.sh                # Multi-process service stop
│   └── val_zsh/                  # Validation shell scripts
│       ├── 0.sh, 10.sh           # 0 and 10 test scenarios
│       ├── goal.sh, goal_new.sh  # Goal-based evaluations
│       ├── objects.sh            # Object manipulation tests
│       ├── orange.sh             # Specific object tests
│       └── spatial.sh            # Spatial reasoning tests
├── script/                       # Tool and utility scripts
│   ├── __init__.py              # Package initialization
│   ├── auto_DL.py              # Automatic searching utilities
│   ├── color.json              # Color configuration
│   ├── Judge_simple.py         # Simple judgment module
│   ├── mmgdino.py              # MM-GroundingDINO integration, including Vision and Language understanding
│   ├── mmgdino_simple.py       # Simplified MM-GroundingDINO
│   ├── qwenvl_meg.py           # QwenVL model enhancement
│   ├── SAM2_1.py               # Segment Anything Model 2.1
│   ├── SAPdivision.py          # SAP (Sub-Action Planning) division
│   ├── segvideo.py             # Video segmentation
│   ├── segvideo_simple.py      # Simplified video segmentation
│   ├── Wholebody.py            # A media function
│   └── test_images/            # Test images and configurations
│       ├── info.json           # Image metadata
│       ├── replacetest.py      # Replacement testing
│       ├── smoke_results.json  # Smoke test results
│       └── test.py             # Test runner
├── prismatic/                  # OpenVLA codebase (original)
└── vla-scripts/                # Model testing
    ├── deploy.py               # Model deployment script
    ├── finetune.py             # Fine-tuning script
    ├── train.py                # Training script
    └── extern/                 # External conversion utilities
        ├── convert_openvla_weights_to_hf.py  # Weight conversion
        ├── test_openvla.py                   # OpenVLA testing
        └── verify_openvla.py                 # OpenVLA verification
```

## 🔧 Core Components

### 🎯 Main Execution (`libero_run/`)
- **`main_agent_clean.py`**: Main execution script containing all tool module calls and agent logic implementation
- **`vision_planner_service.py`**: Service server for planner, Vision, and Language modules. Due to library version compatibility issues, we run the execution and verification module code in a separate process, communicating with the main process through socket communication. For module naming and content details, please refer to the paper.
- **`qwenvl.py`**: Wrapper function for the verification module

### 🛠️ Tool Scripts (`script/`)
- **Computer Vision**: `SAM2_1.py`, `segvideo.py`, `mmgdino.py` - Advanced vision processing
- **Language Models**: `qwenvl_meg.py`, `Judge_simple.py` - Language understanding and judgment
- **Planning**: `SAPdivision.py` - Sub-action planning and task decomposition
- **Utilities**: `auto_DL.py`, `Wholebody.py` - Automation and analysis tools

### 🏗️ Architecture (`prismatic/`)
The remaining code in the experiments folder is based on OpenVLA codebase
- **Backbone Models**: Support for various LLM and vision architectures
- **VLA Integration**: Specialized vision-language-action model implementations
- **Training Infrastructure**: Distributed training with DDP/FSDP support
- **Data Processing**: RLDS dataset integration and preprocessing

## 📊 Evaluation Scripts (`val_zsh/`)
- Comprehensive test scenarios covering different aspects of robot manipulation
- Goal-oriented tasks, object manipulation, and spatial reasoning evaluations

## 🚀 Installation & Deployment

### Overview
This project uses a dual conda environment setup to avoid library version conflicts, particularly with transformers. We recommend using OpenVLA's recommended configuration for the main environment and our specified requirements for the server environment.

### Prerequisites
- **Anaconda/Miniconda**: Latest version
- **Git**: For repository cloning
- **NVIDIA Driver**: 550.54.14+
- **CUDA**: Compatible with PyTorch 2.2/2.3

### Environment Architecture

#### Client Environment Dependencies
- **OpenVLA**: [Core VLA framework](https://github.com/openvla/openvla)
- **LIBERO_ZERO**: [Evaluation benchmark](https://github.com/zhangjiaxuan-Xuan/LIBERO_ZERO)
- **Bulk-Bing-Image-downloader**: [Image downloading utility](https://github.com/ostrolucky/Bulk-Bing-Image-downloader)
- **Cutie**: [Video object segmentation](https://github.com/hkchengrex/Cutie)

#### Server Environment Dependencies
- **MM-GroundingDINO**: [Grounding DINO integration](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino)
- **SAM 2.1**: [Segment Anything Model](https://docs.ultralytics.com/models/sam-2/#interactive-segmentation)
- **Qwen-VL**: [Vision-Language model](https://github.com/QwenLM/Qwen3-VL)
- **GLM-4.1V**: [Thinking model](https://github.com/zai-org/GLM-V)

### Installation Steps

#### Step 1: Client Environment Setup

```bash
# Create and activate client environment
conda env create -f client.yml
conda activate client

# Install video segmentation library
git clone https://github.com/hkchengrex/Cutie
cd Cutie && pip install -e .
cd ..

# Install robot learning benchmark
git clone https://github.com/zhangjiaxuan-Xuan/LIBERO_ZERO 
# Optional: cd LIBERO_ZERO && pip install -e .
# Recommended: Import LIBERO_ZERO by absolute path

# Install OpenVLA dependencies
pip install dlimp@git+https://github.com/moojink/dlimp_openvla
pip install thinplate@git+https://github.com/cheind/py-thin-plate-spline

# Optional: Install Flash Attention for performance
pip install flash-attn==2.5.5
```

#### Step 2: Server Environment Setup

```bash
# Create and activate server environment
conda env create -f server.yml
conda activate server

# Install bulk image downloader
pip install git+https://github.com/ostrolucky/Bulk-Bing-Image-downloader

# Install latest transformers (includes tokenizers)
pip install git+https://github.com/huggingface/transformers.git

# Optional: Install Flash Attention for performance
pip install flash-attn==2.6.1
```

#### Step 3: Model Configuration

1. Download required model weights to local storage
2. Update model paths in all files in experiments and scripts as needed
3. Use validation scripts in `val_zsh/` folder for initial testing

### Quick Start
Enter the 'val_zsh' directory and run a test script, e.g.,

```bash
cd val_zsh
zsh 0.sh
``` 


##   Citation
if you find this project useful in your research, please consider citing:

```bibtex
@misc{zhaozhang2025vla2,
  title={VLA²: Empowering Vision-Language-Action Models with an Agentic Framework for Unseen Concept Manipulation},
  author={Han Zhao, Jiaxuan Zhang, Wenxuan Song, Pengxiang Ding, Donglin Wang},
  eprint={2510.14902},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  year={2025}
}
```

## 🎖️ References
- **OpenVLA**: Open Vision-Language Agents (https://arxiv.org/abs/2304.09103, https://github.com/openvla/openvla)
- **Agentic-Robot**: Referenced codebase (https://github.com/Agentic-Robot/agentic-robot)
- **LIBERO**: Lifelong Robot Learning Benchmark (https://arxiv.org/abs/2307.01620)
- **Qwen-VL**: Qwen Vision-Language Model (https://github.com/QwenLM/Qwen3-VL)
- **MM-GroundingDINO**: Grounding DINO Model (https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino)
- **Segment Anything Model 2.1**: (https://docs.ultralytics.com/zh/models/sam-2/#interactive-segmentation)
- **GLM-V**: GLM Vision-Language Model (https://github.com/zai-org/GLM-V)

## 🔧 todo:
- Updating, new features coming soon.
