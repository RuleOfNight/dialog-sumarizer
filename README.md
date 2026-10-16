
# Graph-Augmented Dialogue Summarization

A notebook-based experiment for English dialogue summarization using **BART**, **syntactic dependency graphs**, and **speaker and turn embeddings**. The project explores how dialogue structure and grammatical relationships can be incorporated into a pretrained sequence-to-sequence model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RuleOfNight/dialog-sumarizer/blob/main/SummaryPaper_%282%29_%281%29_%282%29_%282%29.ipynb)

## Overview

- **Backbone:** `facebook/bart-base`.
- **Dataset:** `knkarthick/samsum`, using its training and validation splits.
- **Graph construction:** spaCy dependency parsing aligned to BART subword positions.
- **Dialogue structure:** learned speaker and turn embeddings added to token embeddings.
- **Graph integration:** two custom Transformer layers after the BART encoder, with learned dependency-relation biases added to attention scores.
- **Evaluation:** ROUGE F1 scores and qualitative comparisons with reference summaries.

This repository contains an experimental training notebook with saved outputs. It does not include a deployed application or a downloadable trained checkpoint.

## Architecture

```text
Dialogue
  ├── spaCy parsing → dependency relation matrix → per-head attention bias
  └── BART tokenization + speaker IDs + turn IDs
                         ↓
       Token + speaker + turn embeddings
                         ↓
                    BART encoder
                         ↓
       Two graph-biased Transformer layers
                         ↓
                    BART decoder
                         ↓
                      Summary
```

Dependency edges connect the first subword of each parsed word to the first subword of its syntactic head. The graph layers use dense attention with learned relation biases; they do not restrict attention to graph neighbors.

## Recorded Validation Results

Scores below are copied from the notebook's saved outputs, not independently reproduced. They are mean per-example F1 scores on a **0–1 scale**, computed with stemming enabled.

| Training stage | Validation loss | ROUGE-1 | ROUGE-L |
| --- | ---: | ---: | ---: |
| Frozen-backbone warm-up | 2.0229 | 0.2750 | 0.2428 |
| Joint training — epoch 1 | 1.7370 | 0.4911 | 0.4051 |
| Joint training — epoch 2 | 1.6705 | 0.4893 | 0.4087 |
| Joint training — epoch 3 | 1.6723 | 0.5070 | 0.4190 |
| Joint training — epoch 4 | 1.6535 | 0.5082 | 0.4209 |
| Low-learning-rate fine-tuning — epoch 1 | 1.6250 | 0.5119 | **0.4254** |
| Low-learning-rate fine-tuning — epoch 2 | 1.6202 | **0.5129** | 0.4249 |
| Low-learning-rate fine-tuning — epoch 3 | 1.6231 | 0.5126 | 0.4249 |

Checkpoint selection uses validation ROUGE-L. The best recorded ROUGE-L is **0.4254**, paired with ROUGE-1 **0.5119**. The evaluation function also computes ROUGE-2, but the displayed training logs do not report it.

No held-out test-set evaluation, plain BART baseline, or component ablation is included. These results therefore do not establish an improvement over standard BART. The notebook's “Baseline” evaluation reloads the trained graph model; it is not an unmodified BART baseline.

## Training Configuration

| Setting | Value |
| --- | --- |
| Random seed | 42 |
| Batch size | 16 |
| Maximum input / target length | 512 / 128 tokens |
| Graph layers | 2 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Warm-up | 1 epoch; BART frozen; learning rate `2e-4` |
| Joint training | 4 epochs total; BART `2e-5`, added components `1e-4` |
| Additional fine-tuning | 3 epochs; BART `5e-7`, added components `2e-6` |
| Validation decoding | 4 beams, maximum length 128, no-repeat trigram blocking, repetition penalty 1.2 |

The notebook metadata records a Google Colab T4 GPU runtime. GPU memory requirements and runtime have not been independently measured.

## Running in Google Colab

1. Open the notebook using the badge above and save a working copy.
2. Select a GPU runtime. The code falls back to CPU when CUDA is unavailable, but training will be slower.
3. Run the dependency installation and spaCy model download cells. Internet access is needed to download the pretrained model and dataset.
4. Before training, review the compatibility notes below. Then run the data preparation, model definition and training cells in order.
5. Run the continuation section for joint epochs 2–4. It saves improved checkpoints as `best_graph_bart_dialogsum.pt` in the runtime's working directory.
6. Run checkpoint inference and, optionally, the additional fine-tuning section. Copy checkpoints to persistent storage before the Colab runtime ends.

The notebook installs `transformers`, `peft`, `pandas`, `spacy`, `datasets` and `rouge_score`, then downloads `en_core_web_sm`. It also imports PyTorch, NumPy and tqdm from the runtime. Although `peft` is installed, the implementation does **not** use LoRA or PEFT adapters.

## Reproduction Notes

- **Tokenizer/model vocabulary:** the tokenizer adds `#Person1#` and `#Person2#`, but the initial training model is not resized. For a fresh run, call `model.bart.resize_token_embeddings(len(tokenizer))` after constructing the model and before creating optimizers.
- **Checkpoint compatibility:** a checkpoint saved after resizing has a different embedding shape from the original BART vocabulary. Resize the destination model to the saved vocabulary size **before** loading such a checkpoint. The notebook's existing loading sequence assumes an original-size checkpoint and resizes afterward; it needs adjustment when loading a resized checkpoint, including after fine-tuning overwrites the same file. Preserve the tokenizer and dependency-label mapping alongside weights.
- **Environment:** package versions are not pinned. A fresh runtime may behave differently from the saved run; record working versions when reproducing the experiment.
- **Data limits:** preprocessing parses at most 5,000 input characters, then truncates model input to 512 tokens. With the current embedding sizes and IDs starting at 1, dialogues support up to 19 speakers and 99 turns before an embedding index exceeds its bounds.
- **Summary quality:** saved examples include incorrect attribution and factual details. ROUGE measures text overlap and does not guarantee factual consistency.

Despite names such as `GraphDialogSumDataset` and `best_graph_bart_dialogsum.pt`, the dataset loaded by the notebook is **SAMSum**, not DialogSum.

## Repository Contents

```text
dialog-sumarizer/
├── README.md
└── SummaryPaper_(2)_(1)_(2)_(2).ipynb
```

## Resources

- [BART backbone](https://huggingface.co/facebook/bart-base)
- [SAMSum dataset used in the notebook](https://huggingface.co/datasets/knkarthick/samsum)
- [spaCy English models](https://spacy.io/models/en)
- [ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge)
