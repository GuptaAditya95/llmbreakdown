# BERT's encoder, step by step with shapes

This follows one sentence pair from raw text to BERT's final hidden vectors, using the BERT-base sizes from the
paper. Section, table and figure numbers refer to the arXiv version,
[arXiv:1810.04805v2](https://arxiv.org/abs/1810.04805).

The headline is short: **BERT's encoder is the Transformer's encoder** (§3, "almost identical to the original").
What changes is what goes in (three embeddings instead of one embedding plus sinusoids), one activation (GELU
instead of ReLU), the sizes, and the job it is trained for. Every step below says whether it is the same as
[Attention, by hand](../../attention/) or changed.

The paper leaves the low-level details to Vaswani et al. (2017) and to the released code. Where a detail comes
from the code ([github.com/google-research/bert](https://github.com/google-research/bert), `modeling.py`,
`create_pretraining_data.py`, `run_pretraining.py`) rather than the paper, it is marked **[code]**.

| Symbol | Meaning | BERT-base (paper) |
|---|---|---|
| n | tokens in the packed sequence, including [CLS] and the [SEP]s | 11 in this example, at most 512 |
| V | WordPiece vocabulary size | 30,000 (§3); the released `vocab.txt` has 30,522 **[code]** |
| H | hidden size: width of every token vector (the Transformer's d_model) | 768 |
| L | number of stacked encoder layers (Transformer blocks) | 12 |
| A | number of attention heads | 12 |
| d_k = d_v | width of each head's queries/keys and values, H / A | 64 |
| d_ff | inner width of the feed-forward network, 4H (§3, footnote 3) | 3072 |
| max positions | rows in the learned position table | 512 (Appendix A.2) |

BERT-large is L = 24, H = 1024, A = 16, d_ff = 4096, 340M parameters (§3).

Shapes are written `(rows × columns)`, one row per token. Real training adds a batch dimension in front, so
`(n × 768)` becomes `(batch × n × 768)`; pre-training used batches of 256 sequences (Appendix A.2).

---

## Step 0: Two "sentences" in, one sequence out (§3)

BERT's input can hold one sentence or a pair. A "sentence" is any contiguous span of text, not necessarily a
linguistic sentence (§3). Our pair is the one in Figure 2:

```
A: my dog is cute
B: he likes playing
```

**vs. the Transformer:** the Transformer had a source sentence for the encoder and a target sentence for the
decoder. BERT has no decoder, so both sentences go into the same encoder, side by side.

## Step 1: WordPiece tokens, plus [CLS] and [SEP] (§3)

WordPiece (Wu et al., 2016) splits rare words into known pieces. A piece that continues a word starts with `##`:
"playing" → `play ##ing`. Then three special tokens are added:

- `[CLS]` at the very front. Its final vector, called **C**, is the "aggregate sequence representation" used
  for classification (§3).
- `[SEP]` after sentence A and after sentence B.

```
tokens = [CLS] my dog is cute [SEP] he likes play ##ing [SEP]        n = 11
ids    = 101  …                102  …                       102     shape: (11,)
```

The special IDs (`[CLS]` = 101, `[SEP]` = 102, `[MASK]` = 103) are the ones in the released vocabulary
**[code]**; the other IDs in these notes are illustrative.

## Step 2: Masking, pre-training only (§3.1, Appendix A.1)

To pre-train, BERT hides some tokens and learns to predict them (the masked language model, "MLM"). The paper
chooses 15% of the token positions at random. Each chosen position is replaced:

| with probability | replacement | example from Appendix A.1 |
|---|---|---|
| 80% | the `[MASK]` token | my dog is hairy → my dog is `[MASK]` |
| 10% | a random token | my dog is hairy → my dog is apple |
| 10% | the original token, unchanged | my dog is hairy → my dog is hairy |

The model still has to predict the original token at all three kinds of position, so it can never assume that
an unmasked-looking token is safe to copy (Appendix A.1). Table 8 compares other mixes.

From the code **[code]**: `[CLS]` and `[SEP]` are never chosen, and the number of positions is
`max(1, round(0.15 × n))`, capped by `max_predictions_per_seq` (20 for 128-token sequences in the released instructions). For n = 11 that is 2. In our example:

```
position 4: cute  → [MASK]      (the 80% case)
position 7: likes → apple       (the 10% random case)
input  = [CLS] my dog is [MASK] [SEP] he apple play ##ing [SEP]
```

Masking changes the tokens, not the shape: still `(11,)`. At inference time there is no random masking at all
(see the heads walkthrough).

## Step 3: Segment IDs (§3, Figure 2)

Each token gets a 0 if it belongs to sentence A and a 1 if it belongs to sentence B. `[CLS]` and the first
`[SEP]` count as A; the last `[SEP]` counts as B.

```
segment = [0 0 0 0 0 0 1 1 1 1 1]                 shape: (11,)
```

**New in BERT.** The Transformer never packed two sentences into one sequence, so it had nothing like this.

## Step 4: Three embedding lookups (§3, Figure 2)

BERT owns three learned tables, each with H columns:

| table | shape (paper) | row picked by |
|---|---|---|
| token embeddings E_tok | V × H = 30,000 × 768 | the token's ID |
| segment embeddings E_seg | 2 × H = 2 × 768 (rows E_A and E_B) | the segment ID |
| position embeddings E_pos | 512 × H = 512 × 768 | the position 0, 1, 2, … |

Each lookup copies rows, so each gives `(n × H)` = `(11 × 768)`.

**vs. the Transformer, changed in two ways.**

1. Position comes from a **learned table**, not the sine and cosine formula. Vaswani et al. had already found
   learned and sinusoidal positions to work about equally well (their Table 3, row E). The price is a hard
   limit: the table has 512 rows, so BERT cannot read a 513th token. Appendix A.2 adds that BERT trained on
   length-128 sequences for 90% of the steps and on length 512 for the last 10% "to learn the positional
   embeddings".
2. The embeddings are **not multiplied by √d_model** **[code]**. The paper doesn't mention it either way; the
   released `embedding_lookup` simply returns the rows.

## Step 5: Add them, then normalize (§3; LayerNorm and dropout [code])

```
X_sum = E_tok[ids] + E_seg[segment] + E_pos[0..n-1]      (11 × 768)   ← element-wise, §3
X0    = LayerNorm(X_sum)                                  (11 × 768)   [code]
X0    = dropout(X0, p = 0.1)                              training only (Appendix A.2: "on all layers")
```

The paper only says the input representation is the sum of the three (§3). The released code then applies a
LayerNorm and dropout to the sum before the first layer (`embedding_postprocessor`). LayerNorm keeps the shape:
each row is shifted to mean 0 and scaled to standard deviation 1, then multiplied by a learned γ and shifted by
a learned β.

## Step 6: One encoder layer (§3: "almost identical to the original")

This is the Transformer encoder layer you already know, with one swap.

**Multi-head self-attention (same as the Transformer).** For each of the A heads:

```
Q = X0 · W_Q,  K = X0 · W_K,  V = X0 · W_V        (11 × 768) · (768 × 64) = (11 × 64)
S = Q · Kᵀ                                        (11 × 64) · (64 × 11)   = (11 × 11)
A = softmax(S / √64) = softmax(S / 8)             (11 × 11), each row adds up to 1
head = A · V                                      (11 × 11) · (11 × 64)   = (11 × 64)
MHA = Concat(head_1 … head_12) · W_O              (11 × 768) · (768 × 768) = (11 × 768)
```

There is **no mask**: every token attends to every token in both sentences, to its left and to its right.
That is exactly what the Transformer's encoder already did. The contrast the paper draws is with OpenAI GPT,
which uses the masked, left-to-right self-attention of a Transformer *decoder* (§3, footnote 4). Because
sentence A and sentence B sit in one sequence, self-attention also does the "bidirectional cross attention
between two sentences" for free (§3.2). The code adds a bias to each projection **[code]**; these notes leave
the biases at 0.

**Add & Norm (same, post-norm).** `X1 = LayerNorm(X0 + MHA)`, shape `(11 × 768)`.

**Feed-forward (changed: GELU).**

```
P  = X1 · W1 + b1          (11 × 768) · (768 × 3072) = (11 × 3072)
G  = GELU(P)               (11 × 3072)                 ← GELU, not ReLU (Appendix A.2)
F  = G · W2 + b2           (11 × 3072) · (3072 × 768) = (11 × 768)
out = LayerNorm(X1 + F)    (11 × 768)
```

GELU (Hendrycks and Gimpel, 2016) is `GELU(x) = x · Φ(x)`, where Φ is the standard normal cumulative
distribution. It behaves like ReLU for large |x| but is smooth around 0 and lets small negative values through
slightly: GELU(−1) ≈ −0.16, GELU(0) = 0, GELU(1) ≈ 0.84. The paper says BERT uses it "rather than the
standard relu, following OpenAI GPT" (Appendix A.2). The released code uses the tanh approximation
`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` **[code]**.

## Step 7: Stack L = 12 layers (§3)

Each layer has its own weights and gives back exactly the shape it was given, `(11 × 768)`. After layer 12 the
paper names the outputs (§3, Figure 1):

```
T = final hidden states          (11 × 768)
C = T[0]                         the [CLS] vector, C ∈ R^768
T_i = T[i]                       the vector of token i, T_i ∈ R^768
```

These vectors are what Part 2 reads. During pre-training C feeds the next sentence prediction head and the
T_i at the masked positions feed the masked LM head. After fine-tuning, C or the T_i feed a small task layer.

---

## Kept, changed, new: the Transformer encoder vs. BERT's encoder

| | Transformer encoder (Vaswani et al., 2017) | BERT encoder |
|---|---|---|
| decoder | yes, a second stack | **removed** |
| input | one sentence | **one or two sentences, packed with [CLS] and [SEP]** |
| token embedding | learned, × √d_model | learned, **no scaling [code]** |
| position | fixed sinusoids | **learned table, 512 rows** |
| segment embedding | none | **new: E_A / E_B** |
| LayerNorm on the input sum | no | **yes [code]** |
| self-attention | multi-head, no mask, ÷ √d_k | same |
| Add & Norm | post-norm, LayerNorm(x + Sublayer(x)) | same |
| feed-forward | d_ff = 4 × d_model, ReLU | d_ff = 4H, **GELU** |
| sizes (base) | N = 6, d_model = 512, h = 8 | **L = 12, H = 768, A = 12** |
| what sits on top | linear + softmax over the next word | **MLM head and NSP head during pre-training; a task layer after fine-tuning** |

## Learned parameters, BERT-base

| Part | Shapes | Count |
|---|---|---|
| Token embeddings | 30,522 × 768 (released vocab **[code]**) | 23,440,896 |
| Position embeddings | 512 × 768 | 393,216 |
| Segment embeddings | 2 × 768 | 1,536 |
| Embedding LayerNorm **[code]** | γ, β of size 768 | 1,536 |
| One layer: Q, K, V, O projections | 4 × (768 × 768 + 768) | 2,362,368 |
| One layer: FFN | (768 × 3072 + 3072) + (3072 × 768 + 768) | 4,722,432 |
| One layer: two LayerNorms | 2 × (γ, β of size 768) | 3,072 |
| **One layer** | | **7,087,872** |
| **Twelve layers** | | **85,054,464** |
| Pooler (used by NSP, Part 2) **[code]** | 768 × 768 + 768 | 590,592 |
| **Total** | | **109,482,240 ≈ 110M** (§3) |

`bert_example.py` prints this count. With the paper's round 30,000-piece vocabulary the total is 109,081,344,
still "110M". The token embedding table alone is about a fifth of the model.

## Self-check

1. Why does BERT need segment embeddings when the Transformer's encoder didn't?
2. What shape is X_sum for a 128-token sequence in BERT-base? And for BERT-large?
3. A token at position 600: what goes wrong, and why wouldn't it go wrong with sinusoids?
4. Which sublayer of the encoder layer changed from the Transformer, and how?
5. Does BERT's self-attention use a mask during pre-training? Which famous model does?
6. Which output vector does BERT use to summarize the whole input, and where does it come from?

Answers: (1) two sentences share one sequence, so each token must say which one it came from; (2) 128 × 768
and 128 × 1024; (3) the learned table has no row 600, while the formula works for any position; (4) the
feed-forward network uses GELU instead of ReLU; (5) no (apart from hiding padding **[code]**); OpenAI GPT
masks the future; (6) C, the final hidden vector of [CLS].
