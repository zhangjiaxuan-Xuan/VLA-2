<div align="center">

# VLA^2: Vision-Language-Action Agent

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![VLA](https://img.shields.io/badge/VLA-Vision--Language--Action-green.svg)]()
[![Agent](https://img.shields.io/badge/Agent-Robotics-red.svg)]()
[![LIBERO](https://img.shields.io/badge/LIBERO-Environment-purple.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

</div>

## 📄 Paper & Resources
- 📝 **Paper**: https://arxiv.org/abs/2510.14902
- 🌐 **Project Page**: https://vla-2.github.io

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

## 🎖️ Citation & References
- **OpenVLA**: Open Vision-Language Agents (https://arxiv.org/abs/2304.09103)
- **Agentic-Robot**: Referenced codebase (https://github.com/Agentic-Robot/agentic-robot)

## 🚀 Deployment
- coming soon

## 🔧 todo:
- Updating, new features coming soon.
