# Build an Aspect-Based Preference Scoring + Evidence Extraction System using Qwen2.5-3B

You are an expert Machine Learning Engineer and NLP Researcher.

Your task is to implement a complete, production-quality Python project (organized as Jupyter notebooks) for fine-tuning **Qwen2.5-3B-Instruct** using **QLoRA**.

The project should be clean, modular, reproducible, and follow modern deep learning best practices.

---

# Project Goal

The model receives a Persian user description about purchasing a laptop.

The output should contain **five predefined aspects**.

For each aspect the model predicts:

1. A continuous relevance score between **0 and 1**
2. An evidence span copied directly from the original input text

The evidence MUST always be an exact span from the input text.

The model must never hallucinate explanations.

---

# Fixed Aspects

The aspects are fixed and always remain the same.

They are:

1. Performance
2. Portability
3. Design & Appearance
4. Durability
5. Cost-effectiveness

---

# Example

Input:

من دانشجوی مهندسی کامپیوتر هستم و می خواهم مدل‌های هوش مصنوعی آموزش بدهم و تحلیل داده انجام بدهم. من هر روز با اتوبوس به دانشگاه می‌روم و لپ‌تاپم را با خودم حمل می‌کنم.

Desired Output:

```json
{
  "performance": {
    "score": 0.95,
    "evidence": "می خواهم مدل‌های هوش مصنوعی آموزش بدهم و تحلیل داده انجام بدهم"
  },
  "portability": {
    "score": 0.93,
    "evidence": "هر روز با اتوبوس به دانشگاه می‌روم و لپ‌تاپم را با خودم حمل می‌کنم"
  },
  "design": {
    "score": 0.10,
    "evidence": ""
  },
  "durability": {
    "score": 0.15,
    "evidence": ""
  },
  "cost_effectiveness": {
    "score": 0.20,
    "evidence": ""
  }
}
```

---

# Model

Use

Qwen2.5-3B-Instruct

Train using

* HuggingFace Transformers
* PEFT
* QLoRA
* bitsandbytes (if CUDA is available)
* Apple Silicon compatible training (MLX or PyTorch MPS fallback should be considered where appropriate)
* Accelerate

Do NOT perform full fine-tuning.

Only use LoRA / QLoRA.

---

# Dataset

The dataset will already exist as JSON.

The code must read it automatically.

Design the loader so that changing the dataset requires no code modification.

Example format:

```json
{
  "text": "...",

  "labels":{

      "performance":{
          "score":0.92,
          "evidence":"..."
      },

      "portability":{
          "score":0.85,
          "evidence":"..."
      },

      "design":{
          "score":0.20,
          "evidence":""
      },

      "durability":{
          "score":0.15,
          "evidence":""
      },

      "cost_effectiveness":{
          "score":0.30,
          "evidence":""
      }

  }

}
```

---

# Important

Evidence is NOT free text.

Evidence MUST be a substring of the input text.

The preprocessing pipeline should automatically verify this.

If an evidence string is not found inside the input text, the sample should either:

* raise an exception
* or be discarded with a warning.

Never silently accept invalid evidence.

---

# Span Representation

During preprocessing convert every evidence into

(start_char, end_char)

Then convert those to

(start_token, end_token)

using the tokenizer offsets.

This token span should later be used during evaluation.

---

# Output Format During Training

The training target should remain structured JSON.

Example:

```json
{
 "performance":{
   "score":0.95,
   "evidence":"..."
 },

 ...
}
```

The prompt template should be designed consistently for instruction tuning.

---

# Fine-tuning Strategy

Use Instruction Fine-tuning.

The model should learn

Input

↓

Structured JSON output

No chain-of-thought should ever be trained.

Do not ask the model to explain its reasoning.

Only predict

* scores
* evidence spans

---

# Data Split

Implement automatic

80% Training

10% Validation

10% Test

Use a fixed random seed.

---

# Evaluation

The evaluation should be separated into two independent tasks.

## Score Evaluation

Compute

* MAE
* RMSE
* Pearson Correlation
* Spearman Correlation

for every aspect individually

and

overall average.

---

## Evidence Evaluation

Evidence evaluation is extremely important.

Implement

Token-level F1

similar to the SQuAD evaluation.

Steps:

Convert

ground truth evidence

↓

token span

Convert

predicted evidence

↓

token span

Compute

Precision

Recall

F1

using token overlap.

Also compute

Exact Match

for comparison.

Generate a final report.

---

# Faithfulness

Faithfulness is mandatory.

The explanation must faithfully support the predicted score.

Implement a multi-objective loss.

The loss should include

Total Loss

=

Score Loss

*

Evidence Loss

*

Faithfulness Regularization

Where

Score Loss

can be MSE or SmoothL1.

Evidence Loss

should encourage accurate span prediction.

Faithfulness Regularization should penalize inconsistent outputs.

The implementation does not have to reproduce a published paper exactly, but it should include a principled consistency objective.

Examples of acceptable strategies include:

* penalizing non-empty evidence for very low predicted scores;
* encouraging high-overlap evidence when the score is high;
* masking evidence loss when no evidence exists;
* weighting evidence loss by the ground-truth score;
* adding consistency constraints between score prediction confidence and evidence prediction.

Document and justify the chosen formulation in the notebook.

---

# Notebook Organization

Do NOT put everything into one notebook.

Organize the project professionally.

Example:

01_environment.ipynb

02_dataset_analysis.ipynb

03_preprocessing.ipynb

04_training.ipynb

05_evaluation.ipynb

06_inference.ipynb

07_error_analysis.ipynb

---

# Code Organization

Create reusable Python modules.

Example

project/

config.py

dataset.py

preprocessing.py

trainer.py

model.py

metrics.py

losses.py

evaluation.py

utils.py

prompts.py

inference.py

visualization.py

Do not place all logic inside notebooks.

The notebooks should only orchestrate experiments.

---

# Configuration

All hyperparameters should be stored in a configuration file.

No hard-coded paths.

---

# Training Features

Implement

Mixed Precision

Gradient Accumulation

Gradient Checkpointing

Early Stopping

Model Checkpointing

Resume Training

TensorBoard logging

Seed fixing

Automatic validation

Learning rate scheduling

Weight decay

---

# Inference

Create an inference notebook.

Input

↓

Persian text

↓

Predicted JSON

Pretty-print the output.

---

# Error Analysis

Create visualizations including

Score distribution

Prediction vs Ground Truth

MAE per aspect

Token F1 histogram

Confusion examples

Worst predictions

Best predictions

---

# Documentation

Every notebook should contain explanations.

Every function should include docstrings.

Use type hints.

Follow PEP8.

Write readable and maintainable code.

---

# Dependencies

Use

Python 3.11+

PyTorch

Transformers

PEFT

Accelerate

Datasets

Evaluate

Scikit-learn

Pandas

NumPy

Matplotlib

Jupyter

Avoid unnecessary dependencies.

---

# Engineering Requirements

The implementation should be production-quality.

The code should be modular.

Reusable.

Easy to extend.

Well documented.

No duplicated code.

No notebook should exceed a reasonable size.

All preprocessing should be deterministic.

The pipeline should be reproducible from start to finish.

The final project should look like something that could be published as an open-source GitHub repository.

Whenever there is a design decision, prefer correctness, maintainability, and research-grade implementation over shortcuts.
