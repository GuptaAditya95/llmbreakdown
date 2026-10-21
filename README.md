# llmbreakdown

**Attention by Hand** is an interactive, do-it-yourself walk through one full forward pass of the Transformer from
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
from token embeddings all the way through the encoder and decoder to the output probabilities.

**Live site:** https://guptaaditya95.github.io/llmbreakdown/

## Who this is for

Anyone who wants to understand architectures from scratch and enjoys learning them by working
through the math themselves. The vision is simple: once you go through the interactive site and its
hands-on homework, you should be able to hold a mental image of how the Transformer works. That
picture makes every later variant much more intuitive.

## What inspired this

I was learning about attention, and *Attention Is All You Need* is the foundational paper of that
whole ecosystem. Reading it, I wanted more than the equations. I wanted to see every matrix, know its
shape at every step, and do the multiplications myself until the forward pass clicked.

The part that fascinated me most was positional encoding. The researchers brought sine and cosine
into the model to give every position its own signature, so positions come to life without relying
only on learned vectors, which are far less flexible. That idea is what pushed me to build this.

I hope it helps you build your own picture of the Transformer, and that you go on to make even more
vivid versions of how you imagine this whole process.

## What's inside

| Path | What it is |
| --- | --- |
| [`index.html`](index.html) | The interactive site: step by step through the forward pass, with a practice panel beside every step and a tracker of matrix dimensions after each step |
| [`notes/encoder-walkthrough.md`](notes/encoder-walkthrough.md) | Plain-language walkthrough of the encoder |
| [`notes/decoder-walkthrough.md`](notes/decoder-walkthrough.md) | Plain-language walkthrough of the decoder and its connections |
| [`notes/positional-encoding-calculations.md`](notes/positional-encoding-calculations.md) | Sinusoidal positional encoding worked by hand, with the 6×512 table in [`positional-encoding-6x512.csv`](notes/positional-encoding-6x512.csv) |
| [`notes/encoder-layer-calculations.md`](notes/encoder-layer-calculations.md) | Every calculation in one encoder layer, with [`encoder_layer_example.py`](notes/encoder_layer_example.py) to check them |
| [`notes/encoder-practice.ipynb`](notes/encoder-practice.ipynb) | Notebook to fill in every encoder matrix for a small example |
| [`notes/transformer-practice.ipynb`](notes/transformer-practice.ipynb) | Notebook covering the full encoder and decoder pass |
| [`worksheets/sample-encoder-worksheet.pdf`](worksheets/sample-encoder-worksheet.pdf) | A printable worksheet for practicing on paper |

The `make_*_practice.py` scripts in `notes/` regenerate the notebooks.

## Run it locally

It is a single self-contained HTML file. Clone the repo and open `index.html` in a browser.

## Like it?

If this helped you understand the Transformer, please **give the repo a star ⭐**. It helps other
learners find it, and it tells me to keep building more of these for other architectures.

Issues and pull requests with corrections or ideas are very welcome.
