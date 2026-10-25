# Let's Break It Down

Hands-on, by-hand breakdowns of the architectures behind modern AI. Each one walks through a full
forward pass, every matrix, every shape, every sum, with a practice panel beside each step so you do
the math yourself.

**Live site:** https://guptaaditya95.github.io/llmbreakdown/

## Breakdowns

| # | Breakdown | Paper | Links |
| --- | --- | --- | --- |
| 01 | **Attention, by hand** | [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | [Site](https://guptaaditya95.github.io/llmbreakdown/attention/) · [Folder](attention/) |

More are on the way.

## Who this is for

Anyone who wants to understand architectures from scratch and enjoys learning them by working
through the math themselves. The vision is simple: once you go through a breakdown and its hands-on
homework, you should be able to hold a mental image of how that architecture works. That picture
makes every later variant much more intuitive.

## What inspired this

I was learning about attention, and *Attention Is All You Need* is the foundational paper of that
whole ecosystem. Reading it, I wanted more than the equations. I wanted to see every matrix, know its
shape at every step, and do the multiplications myself until the forward pass clicked.

The part that fascinated me most was positional encoding. The researchers brought sine and cosine
into the model to give every position its own signature, so positions come to life without relying
only on learned vectors, which are far less flexible. That idea is what pushed me to build the first
breakdown, and I enjoyed it so much that I want to keep doing this for more architectures.

I hope these help you build your own picture of each model, and that you go on to make even more
vivid versions of how you imagine these processes.

## Repo layout

```
index.html          the hub page listing every breakdown
PLAYBOOK.md         how to build the next breakdown
attention/          Attention, by hand
  index.html        the interactive site
  notes/            walkthroughs, worked calculations and practice notebooks
  worksheets/       printable practice worksheet
```

### Adding a breakdown

The full recipe (design system, page structure, flow, exercises, notebooks, worksheet, publishing) is in
[PLAYBOOK.md](PLAYBOOK.md). The short version:

1. Create a folder for it (for example `bert/`) with its own `index.html` and any notes.
2. In the root `index.html`, copy an existing `<li class="card">` block and point its link at the new folder.
3. Add a row to the Breakdowns table above.

## Run it locally

Everything is plain, self-contained HTML. Clone the repo and open `index.html` in a browser.

## Like it?

If these help you understand how the models work, please **give the repo a star ⭐**. It helps other
learners find it, and it tells me to keep breaking down more architectures.

Issues and pull requests with corrections or ideas are very welcome.
