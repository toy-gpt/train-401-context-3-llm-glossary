# Toy-GPT: train-401-context-3-llm-glossary

[![Docs](https://img.shields.io/badge/docs-live-blue)](https://toy-gpt.github.io/train-401-context-3-llm-glossary/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![CI](https://github.com/toy-gpt/train-401-context-3-llm-glossary/actions/workflows/ci-shared.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-401-context-3-llm-glossary/actions/workflows/ci-shared.yml)
[![Deploy-Docs](https://github.com/toy-gpt/train-401-context-3-llm-glossary/actions/workflows/deploy-mkdocs-shared.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-401-context-3-llm-glossary/actions/workflows/deploy-mkdocs-shared.yml)
[![Check Links](https://github.com/toy-gpt/train-401-context-3-llm-glossary/actions/workflows/links.yml/badge.svg)](https://github.com/toy-gpt/train-401-context-3-llm-glossary/actions/workflows/links.yml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/toy-gpt/train-401-context-3-llm-glossary/security)

> Demonstrates, at very small scale, how a language model is trained.

This repository is part of a series of toy training repositories plus a companion client repository:

- [**Training repositories**](https://github.com/toy-gpt) produce pretrained artifacts (vocabulary, weights, metadata).
- A [**web app**](https://toy-gpt.github.io/toy-gpt-chat/) loads the artifacts and provides an interactive prompt.

## Contents

- a small, declared text corpus
- a tokenizer and vocabulary builder
- a simple next-token prediction model
- a repeatable training loop
- committed, inspectable artifacts for downstream use

### ⚠️ Large artifacts are excluded from this repository

This repository uses the `llm_glossary` corpus,
which has a vocabulary of approximately 200 unique tokens.
The context-3 model stores one weight row per unique 3-token context,
so the weight matrix grows as vocab³:

| Corpus | Vocab size | Model | Weight matrix rows | Approx. size |
|--------|-----------|-------|--------------------|--------------|
| cat/dog | ~20 tokens | context-3 | 20³ = 8,000 | ~3 MB |
| llm_glossary | ~119 tokens | context-2 | 119² = 14,161 | ~10 MB |
| llm_glossary | ~119 tokens | context-3 | 119³ = 1,685,159 | **~428 MB** |

The weight matrix file (`artifacts/02_model_weights.csv`) is 428 MB.

The token embeddings file (`artifacts/03_token_embeddings.csv`) is 55 MB.

The weight matrix file exceeds GitHub's 100 MB per-file hard limit.

Both files are excluded from version control via `.gitignore`.

### Large, but Sparse (Mostly zeros / empty)

The 428 MB weight matrix contains approximately 8 million rows:
one for every possible 3-token context in a vocabulary
of ~119 unique tokens (119³ = 1,685,159 combinations,
growing further with padding).

Yet the llm_glossary corpus contains only a few hundred sentences.
The vast majority of those 8 million rows are all zeros because
**most 3-token combinations never appear in the training text**.

For example, the context `a|a|*` produces 119 rows:
one for each possible next token.
But the sequence "a a" never occurs in the corpus,
so every one of those rows is zero.
This pattern repeats across nearly the entire matrix.

This is called **sparsity**: a matrix where most entries are zero.
It is not wrong; it's a consequence of how explicit lookup-table models work.

Real large language models solve this in two ways:

- **Embeddings**: represent each token as a dense numeric vector so that similar
  tokens have similar representations, rather than storing one row per combination
- **Attention**: compute context dynamically from embeddings rather than looking
  up a pre-stored row for every possible context

This repository makes the sparsity problem visible and measurable.
The 428 MB file can be generated and inspected,
but it's not committed because storing 8 million
mostly-zero rows in a git repository serves no practical purpose.
Running the training script locally generates it in seconds
and makes the point quite well.

### Combinatorial Explosion

At vocabulary size V and context window W, the weight matrix has V^W rows.
**This combinatorial explosion of context-window models is exactly why real language models use embeddings and attention instead of explicit lookup tables.**

The smaller corpora (cat/dog, animals) are used for committed artifacts
precisely because their vocabularies are tiny.

## Scope

This is:

- an intentionally inspectable training pipeline
- a next-token predictor trained on an explicit corpus
- a demonstration of why naive context-window scaling fails at non-trivial vocabulary sizes

This is not:

- a production system
- a full Transformer implementation
- a chat interface
- a claim of semantic understanding

## Outputs

Training runs successfully and produces all artifacts locally.
Only `artifacts/00_meta.json` and `artifacts/01_vocabulary.csv` are committed,
as they are small and sufficient to inspect vocabulary and model metadata.

Training logs and evidence are written under `outputs/` (for example, `outputs/train_log.csv`).

## Quick Start

See `SETUP.md` for full setup and workflow instructions.

Run the full training script:

```shell
uv run python src/toy_gpt_train/d_train.py
```

Run individually:

- a/b/c are demos (can be run alone if desired)
- d_train produces artifacts
- e_infer consumes artifacts

```shell
uv run python src/toy_gpt_train/a_tokenizer.py
uv run python src/toy_gpt_train/b_vocab.py
uv run python src/toy_gpt_train/c_model.py
uv run python src/toy_gpt_train/d_train.py
uv run python src/toy_gpt_train/e_infer.py
```

## Command Reference

The commands below are used in the workflow guide above.
They are provided here for convenience.

Follow the guide for the **full instructions**.

<details>
<summary>Show command reference</summary>

### In a machine terminal (open in your `Repos` folder)

After you get a copy of this repo in your own GitHub account,
open a machine terminal in your `Repos` folder:

```shell
# Replace username with YOUR GitHub username.
git clone https://github.com/username/train-401-context-3-llm-glossary

cd train-401-context-3-llm-glossary
code .
```

### In a VS Code terminal

```shell
uv self update
uv python pin 3.14
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
git add -A
uvx pre-commit run --all-files

uv run python -m cintel.case_drift_detector

uv run ruff format .
uv run ruff check . --fix
uv run zensical build

git add -A
git commit -m "update"
git push -u origin main
```

</details>

## Provenance and Purpose

The primary corpus used for training is declared in `SE_MANIFEST.toml`.

This repository commits pretrained artifacts so the client can run
without retraining.

## Annotations

[ANNOTATIONS.md](./ANNOTATIONS.md) - REQ/WHY/OBS annotations used

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)

## SE Manifest

[SE_MANIFEST.toml](./SE_MANIFEST.toml) - project intent, scope, and role
