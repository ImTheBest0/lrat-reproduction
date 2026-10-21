# LRAT Reproduction

A 20-question InfoSeek-Eval pilot comparing base Qwen3-Embedding-0.6B with LRAT, including saved runs and an offline evaluation audit.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/ImTheBest0/lrat-reproduction.git
cd lrat-reproduction
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/01_lrat_pilot_audit.ipynb
```

Run the audit directly:

```bash
python analysis/pilot_audit.py
```

## Results

Recorded evaluator scores: **Qwen 10/20**, **LRAT 15/20**. Some credited outputs contain no final answer, so these scores are not validated final-answer accuracy. See the [notebook](notebooks/01_lrat_pilot_audit.ipynb) for details.

The audit runs locally without a GPU. Model weights, corpus, and index binaries are not included.

## References

- [LRAT paper](https://arxiv.org/abs/2604.04949v1)
- [Official repository](https://github.com/Yuqi-Zhou/LRAT)
