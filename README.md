# SemDistill: Bootstrapping a Low-Latency, Low-Cost Semantic Table Annotator from Noisy LLM

## Abstract

Relational tables widely exist in enterprise data lakes and repositories, yet they often lack explicit semantic context such as column types and relationships, creating a bottleneck for automated data management. Semantic Table Annotation, comprising tasks like Column Type Annotation (CTA) and Column Property Annotation (CPA), is essential to restoring missing semantic context. While Large Language Models (LLMs) excel at these tasks, their scalable deployment is hindered by prohibitive costs, high latency, and privacy information leakage. Distilling LLM capabilities into compact local models offers a viable solution; however, these models inevitably overfit to the structured noise in the teacher’s predictions. In this paper, we formalize the setting of label-efficient distillation from noisy LLMs. Our empirical analysis reveals two error patterns: (1) Systematic Ontological Drift, a global class-conditional bias where LLMs tend to over-generalize specific concepts; and (2) High-Confident Hallucinations, stubborn high-confidence errors on ambiguous data. To address these, we propose SemDistill, a framework that bootstraps high-performance local annotators from noisy LLM outputs using only scarce trusted anchors. SemDistill employs a divide-and-conquer strategy: it rectifies global drift via anchor-guided statistical correction and identifies residual hallucinations through trajectory-aware learning dynamics. Extensive experiments demonstrate that SemDistill effectively addresses the cost-quality trade-off. It outperforms the LLM teacher (GPT-5-mini) and the distillation baseline while reducing inference costs by 3800× and latency by 2600×.

## Quick Start

### Complete Training Pipeline on SATO Dataset

The complete SemDistill training pipeline consists of three stages: initial training, AUM-based data cleaning, and retraining with corrected data.

#### Step 1: Initial Training

```bash
# Set environment variables
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

EXP_NAME=sato
EXP_DIR=outputs/$EXP_NAME
CONFIG_FILE=configs/config_sato.yaml

# Run initial training
python scripts/quick_start.py \
  --config $CONFIG_FILE \
  --experiment_name $EXP_NAME \
  --output_dir $EXP_DIR \
  --logging_dir $EXP_DIR \
  --trainer_mode acd \
  --use_aum \
  --correction_type glc \
  --anchor_budget 20 \
  --trusted_weight 1.0
```

#### Step 2: AUM-based Correction

Use AUM scores to identify and filter noisy samples:

```bash
# Set AUM cutoff threshold (adjust based on your dataset)
aum_cutoff=0.3

# Sample selection based on AUM scores
python scripts/hitl_pipeline.py --config $CONFIG_FILE --step sample --base_dir $EXP_DIR --aum "$aum_cutoff"

# (Optional) Human-in-the-loop correction
python scripts/hitl_pipeline.py --config $CONFIG_FILE --step correct --base_dir $EXP_DIR --aum "$aum_cutoff"
```

#### Step 3: Retraining with Cleaned Data

Retrain the model with the corrected/filtered dataset:

```bash
python scripts/quick_start.py \
  --config $CONFIG_FILE \
  --experiment_name $EXP_NAME \
  --output_dir $EXP_DIR \
  --logging_dir $EXP_DIR \
  --trainer_mode acd \
  --correction_type glc \
  --anchor_budget 20 \
  --trusted_weight 1.0
```

### Quick Training (without AUM cleaning)

For faster training without the data cleaning step:

```bash
python scripts/quick_start.py \
  --config configs/config_sato.yaml \
  --experiment_name sato_quick \
  --output_dir outputs/sato_quick \
  --trainer_mode acd \
  --correction_type glc
```

### Create a Configuration File

Create a new YAML config file in `configs/`:

```yaml
# config_mydata.yaml
dataset:
  dataset_name: "my_dataset"
  data_dir: "data/my_dataset"
  num_classes: 50  # Number of semantic types in your dataset
  max_length: 32

model:
  model_name: "bert-base-uncased"
  use_lora: true
  lora_r: 16
  lora_alpha: 32

training:
  num_epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  trainer_mode: "acd"
  correction_type: "glc"
```

### Run Training

```bash
python quick_start.py \
  --config configs/config_mydata.yaml \
  --experiment_name my_experiment \
  --output_dir outputs/my_experiment
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

