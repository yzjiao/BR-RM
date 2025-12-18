# Code for Branch-and-Rethink Reasoning Reward Model

## Introduction

**BR-RM (Branch-and-Rethink Reasoning Reward Model)** ([Paper Link](https://arxiv.org/pdf/2510.23596)) is a state-of-the-art two-turn reward modeling framework that addresses the problem of *judgment diffusion* in traditional reward models. Instead of compressing all quality dimensions into a single scalar in one pass, BR-RM uses a structured two-stage approach:

- **Turn 1: Adaptive Branching** - The model selects 1-3 instance-critical evaluation dimensions (e.g., factuality, safety, logical reasoning) and generates focused issue analyses for each response.
- **Turn 2: Branch-Conditioned Rethinking** - Using findings from Turn 1, the model performs a targeted re-evaluation through the lens of the selected dimensions, applying task-specific evaluation hierarchies to produce a final preference judgment.

This approach achieves state-of-the-art performance on three challenging reward modeling benchmarks:
- **RewardBench**: 92.1% (BR-RM-Qwen-14B)
- **RM-Bench**: 85.9% (BR-RM-Qwen-14B) 
- **RMB**: 74.7% (BR-RM-Qwen-14B)

### Key Features
- ✅ Two-turn reasoning for focused evaluation
- ✅ Adaptive dimension selection based on instance
- ✅ GRPO-style reinforcement learning with binary outcome rewards
- ✅ Built on NeMo-RL for scalability (1B to 100B+ parameters)
- ✅ Compatible with standard RLHF pipelines

## How to Run

### Prerequisites

This project is built on [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). Please follow the NeMo-RL setup instructions first:

1. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/BR-RM.git
cd BR-RM
git submodule update --init --recursive
```

2. **Install uv** (for environment management):
```bash
# Follow instructions at https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Create virtual environment**:
```bash
uv venv
```

4. **Build flash-attention** (optional, for faster training):
```bash
bash tools/build-flash-attn-in-uv-cache.sh
```

5. **Set up environment variables**:
```bash
export HF_HOME=/path/to/your/huggingface/cache
export WANDB_API_KEY=your_wandb_key  # Optional, for logging
huggingface-cli login  # For accessing gated models like Llama
```

### Data Preparation

BR-RM is trained on a diverse mix of preference datasets. Use the provided preprocessing script to prepare the training and validation data:
```bash
cd dataset
uv run python preprocess_data.py
```

This script processes and combines:
- **HelpSteer3**: Human-annotated preference data across diverse tasks
- **Skywork-Reward-Preference-80K**: Focus on instruction-following and safety
- **Code-Preference-Pairs**: Programming-related preferences (8K subset)
- **Math-Step-DPO-10K**: Mathematical reasoning preferences

The script generates:
- `train_data.jsonl`: Training dataset (~97K samples)
- `val_data.jsonl`: Validation dataset from HelpSteer3

### Train the Model

#### Single-Node Training (8B Model)

For training on a single node with 8 GPUs using Qwen3-8B:
```bash
uv run python examples/run_grpo_brrm.py \
    --config examples/configs/grpo_brrm.yaml \
    policy.model_name="Qwen/Qwen3-8B" \
    cluster.gpus_per_node=8 \
    cluster.num_nodes=1 \
    checkpointing.checkpoint_dir="results/brrm_qwen3_8b" \
    logger.wandb_enabled=True \
    logger.wandb.name="brrm-qwen3-8b"
```

#### Multi-Node Training

For distributed training across multiple nodes, use the provided SLURM scripts:

**8B Model (2 nodes)**:
```bash
# Edit train_brrm_multinode_8b.sh to set:
# - CONTAINER: Path to your NeMo-RL container
# - MOUNTS: Your filesystem mounts
# - Cluster account and partition
./train_brrm_multinode_8b.sh
```

**14B Model (8 nodes)**:
```bash
# Edit train_brrm_multinode_14b.sh similarly
./train_brrm_multinode_14b.sh
```

#### Training Configuration

Key hyperparameters (see `examples/configs/grpo_brrm.yaml`):
- **Learning rate**: 5e-7 (8B), 1e-6 (14B)
- **Temperature**: 1.0
- **KL penalty**: 0.001
- **Batch size**: 256
- **Rollouts per prompt**: 8
- **Max sequence length**: 16384
- **Training steps**: 1000 (with validation every 10 steps)

The training uses:
- GRPO (Group Relative Policy Optimization) for RL
- Binary outcome reward with format penalties
- Two-turn structured generation
- FSDP2 for model parallelism

#### Existing Model Checkpoints on Hugging Face

Pre-trained BR-RM models will be available on Hugging Face (coming soon):
- [`nvidia/Qwen3-Nemotron-14B-BRRM`](https://huggingface.co/nvidia/Qwen3-Nemotron-14B-BRRM)
- [`nvidia/Qwen3-Nemotron-8B-BRRM`](https://huggingface.co/nvidia/Qwen3-Nemotron-8B-BRRM)

To use pre-trained checkpoints:
```bash
uv run python examples/run_eval_brrm.py \
    ++generation.model_name="nvidia/Qwen3-Nemotron-8B-BRRM" \
    --dataset rewardbench
```

### Evaluate the Model

#### Convert Checkpoints

First, convert your trained checkpoint from DCP format to Hugging Face format:
```bash
# Convert a single checkpoint
uv run python examples/converters/convert_dcp_to_hf.py \
    --config results/brrm_qwen3_8b/step_240/config.yaml \
    --dcp-ckpt-path results/brrm_qwen3_8b/step_240/policy/weights/ \
    --hf-ckpt-path results/brrm_qwen3_8b/hf/step_240

# Or convert all checkpoints using the batch script
./convert_dcp_to_hf.sh results/brrm_qwen3_8b latest
```

#### Run Evaluation

Evaluate on individual datasets:
```bash
# RewardBench
uv run python examples/run_eval_brrm.py \
    --dataset rewardbench \
    ++generation.model_name=results/brrm_qwen3_8b/hf/step_240 \
    ++eval.output_file=results/brrm_qwen3_8b/outputs/step_240_rewardbench_results.json \
    ++cluster.gpus_per_node=2

# RM-Bench
uv run python examples/run_eval_brrm.py \
    --dataset rmbench \
    ++generation.model_name=results/brrm_qwen3_8b/hf/step_240 \
    ++eval.output_file=results/brrm_qwen3_8b/outputs/step_240_rmbench_results.json \
    ++cluster.gpus_per_node=2

# RMB
uv run python examples/run_eval_brrm.py \
    --dataset rmb \
    ++generation.model_name=results/brrm_qwen3_8b/hf/step_240 \
    ++eval.output_file=results/brrm_qwen3_8b/outputs/step_240_rmb_results.json \
    ++cluster.gpus_per_node=2
```

For multi-node evaluation or batch evaluation across checkpoints:
```bash
# Evaluate a single checkpoint on all datasets (multi-node)
./evaluate_brrm_one_step.sh \
    results/brrm_qwen3_8b/hf/step_240 \
    240 \
    results/brrm_qwen3_8b/outputs \
    "rewardbench,rmbench,rmb"

# Evaluate all checkpoints
./evaluate_brrm_all_steps.sh \
    results/brrm_qwen3_8b \
    latest \
    "rewardbench,rmbench,rmb"
```

#### View Results

Print formatted evaluation results:
```bash
uv run python print_evaluation_results.py results/brrm_qwen3_8b/outputs
```

#### Model Outputs on Three Datasets

Example performance of BR-RM-Qwen-14B:

**RewardBench**:
```
Overall Accuracy: 92.1%
├── Chat: 97.0%
├── Chat-Hard: 82.4%
├── Safety: 90.0%
└── Reasoning: 98.8%
```

**RM-Bench**:
```
Overall Accuracy: 85.9%
├── Chat: 77.3%
├── Math: 92.6%
├── Code: 79.8%
├── Safety: 93.7%
├── Easy: 92.0%
├── Normal: 88.1%
└── Hard: 77.6%
```

**RMB**:
```
Overall Accuracy: 74.7%
├── Helpfulness (BoN): 67.0%
├── Helpfulness (Pairwise): 81.0%
├── Harmlessness (BoN): 69.3%
└── Harmlessness (Pairwise): 81.6%
```

## Core Code

### Key Files and Their Purposes

#### Training Components

- **`examples/run_grpo_brrm.py`**: Main training script that orchestrates the GRPO training loop with the two-stage BR-RM environment. Sets up data, model, and training infrastructure.

- **`nemo_rl/environments/brrm_environment.py`**: Core BR-RM implementation containing:
  - `BRRMEnvironment`: Ray remote actor that manages the two-turn evaluation process
  - `format_unified_analysis_prompt()`: Turn 1 prompt formatting for adaptive branching
  - `format_scoring_stage_prompt()`: Turn 2 prompt formatting for scoring
  - `parse_unified_analysis_response()`: Parser for Turn 1 quality assessment output
  - `parse_scoring_response()`: Parser for Turn 2 final preference judgment
  - Reward calculation logic with format penalties and outcome rewards

- **`examples/configs/grpo_brrm.yaml`**: Training configuration including:
  - GRPO hyperparameters (KL penalty, clipping, etc.)
  - Model configuration (architecture, parallelism)
  - Data paths and preprocessing settings
  - Environment-specific settings (format penalty, reward weights)

#### Data Processing

- **`dataset/preprocess_data.py`**: Processes raw datasets into the BR-RM training format:
  - `process_hs3_dataset()`: HelpSteer3 processing
  - `process_skywork_dataset()`: Skywork-80K processing
  - `process_codepref_dataset()`: Code preference processing
  - `process_math10k_dataset()`: Math reasoning processing
  - Converts all to unified JSONL format with context, response1, response2, and preference

#### Evaluation Components

- **`examples/run_eval_brrm.py`**: Evaluation script that:
  - Loads datasets (RewardBench, RM-Bench, RMB)
  - Runs two-stage evaluation using vLLM for generation
  - Parses outputs and calculates accuracy metrics
  - Supports batch evaluation across multiple GPUs

- **`print_evaluation_results.py`**: Post-processing script that:
  - Aggregates results across checkpoints
  - Computes domain-specific accuracies
  - Handles RM-Bench's 3x3 comparison matrix
  - Pretty-prints formatted results

#### Utilities

- **`convert_dcp_to_hf.sh`**: Batch conversion of PyTorch DCP checkpoints to HuggingFace format for evaluation and deployment

- **`train_brrm_multinode_*.sh`**: SLURM job submission scripts for distributed training with configurable:
  - Node count and GPU allocation
  - Model size (8B, 14B)
  - Hyperparameters (learning rate, temperature, KL)
  - Parallelism strategies (FSDP2, tensor parallel, sequence parallel)

### Architecture Overview
```
Training Flow:
1. Load preference data (context, response1, response2, preference)
2. GRPO samples multiple rollouts per prompt
3. For each rollout:
   Turn 1: Model selects dimensions → generates quality analysis
   Turn 2: Model receives Turn 1 output → generates final preference
4. Calculate binary reward: 0 if correct preference, -1 if incorrect, -100 if format error
5. Update policy using GRPO objective with group-relative advantages
6. Repeat until convergence

Evaluation Flow:
1. Load test samples
2. For each sample:
   Turn 1: Generate quality assessment with dimension selection
   Turn 2: Generate final preference conditioned on Turn 1
3. Parse and validate outputs
4. Calculate accuracy metrics by domain and difficulty
```

## Reference

If you use BR-RM in your research, please cite:
```bibtex
@article{jiao2025branchrethinkreward,
  title={Think Twice: Branch-and-Rethink Reasoning Reward Model},
  author={Jiao, Yizhu and Zeng, Jiaqi and Vialard, Julien Veron and Kuchaiev, Oleksii and Han, Jiawei and Delalleau, Olivier},
  journal={arXiv preprint arXiv:2510.23596},
  year={2025}
}
```

For the NeMo-RL framework:
```bibtex
@misc{nemo-rl,
  title = {NeMo RL: A Scalable and Efficient Post-Training Library},
  howpublished = {\url{https://github.com/NVIDIA-NeMo/RL}},
  year = {2025},
  note = {GitHub repository},
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


## Acknowledgments

This work builds upon [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) and uses preference data from:
- [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) (NVIDIA)
- [Skywork-Reward-Preference-80K](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) (Skywork AI)
- [Code-Preference-Pairs](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs) (Vezora)
- [Math-Step-DPO-10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K) (Xin Lai)
