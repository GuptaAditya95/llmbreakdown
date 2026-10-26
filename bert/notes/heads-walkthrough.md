# BERT's heads: two tasks, one summed loss, and how BERT is used

This continues from [encoder-walkthrough.md](encoder-walkthrough.md). The encoder has turned the masked
sequence into T, one 768-wide vector per token, with C = T[0] for [CLS]. This part is what BERT adds on top,
and it is the part that is genuinely new compared with the Transformer.

Section numbers refer to [arXiv:1810.04805v2](https://arxiv.org/abs/1810.04805). Details marked **[code]** come
from the released implementation ([github.com/google-research/bert](https://github.com/google-research/bert),
`run_pretraining.py` and `modeling.py`), not from the paper. The paper describes the heads in two sentences
(§3.1); the code fills in the layers.

| Symbol | Meaning | BERT-base |
|---|---|---|
| k | number of masked positions in this sequence | 2 here; about 15% of n |
| V | vocabulary size | 30,000 (§3); 30,522 in `vocab.txt` **[code]** |
| H | hidden size | 768 |
| T_i | final hidden vector of token i | ∈ R^768 |
| C | final hidden vector of [CLS] | ∈ R^768 |

```
                    total loss  (one number, pre-training only)
                         ▲
               ┌─────────┴─────────┐
           MLM loss  +         NSP loss
               ▲                   ▲
   softmax over V per masked   softmax over {IsNext, NotNext}
               ▲                   ▲
   · E_tokᵀ + bias  (tied)     · W_nspᵀ + b   (2 × H)
               ▲                   ▲
   dense + GELU + LayerNorm    pooler: dense + tanh
               ▲                   ▲
   T at the masked positions   C = T[0]
               └─────────┬─────────┘
                    T  (n × H), from the encoder
```

---

## Task 1: the masked LM head (§3.1)

**The paper:** "the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the
vocabulary, as in a standard LM", and "T_i will be used to predict the original token with cross entropy loss"
(§3.1). Unlike a denoising autoencoder, only the chosen positions are predicted, not the whole input.

**Step 1: keep only the masked rows.** Gather the k chosen positions out of T **[code: `gather_indexes`]**.

```
T_m = T[[4, 7]]                         (k × H) = (2 × 768)
```

All three kinds of chosen position are included: the ones turned into [MASK], the randomly replaced ones and
the ones left unchanged.

**Step 2: transform.** One dense layer H → H, GELU, LayerNorm **[code: `cls/predictions/transform`]**.

```
U  = GELU(T_m · W_t + b_t)              (2 × 768) · (768 × 768) = (2 × 768)
U′ = LayerNorm(U)                       (2 × 768)
```

**Step 3: score every vocabulary piece, with the embedding table.** The output weights are the token
embedding table itself, transposed (weight tying), plus one learned bias per vocabulary entry **[code]**.

```
logits = U′ · E_tokᵀ + b_vocab          (2 × 768) · (768 × 30,000) = (2 × 30,000)
```

Each logit is a dot product between the transformed vector and one token's input embedding: "how much does
what I think belongs here look like this piece?" (The Transformer tied its output layer to the embedding table
in the same way, §3.4 of Vaswani et al.)

**Step 4: softmax and the MLM loss.**

```
P = softmax(logits)                     (2 × 30,000), each row adds up to 1
loss_i = −ln P[i, original token i]
MLM loss = mean over the k positions    (one number)
```

With the toy weights in `bert_example.py`: P(cute) = 0.025 and P(likes) = 0.025, so MLM loss ≈ 3.68. With 16
pieces in the toy vocabulary, a uniform guess would give ln 16 ≈ 2.77; random weights do worse than that.

## Task 2: the next sentence prediction head (§3.1)

**The paper:** for each pre-training example, 50% of the time B really follows A (label IsNext) and 50% of the
time B is a random sentence from the corpus (NotNext). "C is used for next sentence prediction" (§3.1,
Figure 1). The final model reaches 97–98% NSP accuracy (footnote 5), and footnote 6 warns that C "is not a
meaningful sentence representation without fine-tuning".

**Step 5: pool [CLS].** The code first runs C through a dense layer with tanh, the "pooler" **[code:
`get_pooled_output`]**.

```
pooled = tanh(C · W_p + b_p)            (1 × 768) · (768 × 768) = (1 × 768)
```

**Step 6: a two-way classifier.**

```
logits = pooled · W_nspᵀ + b_nsp        (1 × 768) · (768 × 2) = (1 × 2)     [code: W_nsp is 2 × H]
P = softmax(logits)                     (IsNext, NotNext)
NSP loss = −ln P[true label]            (one number)
```

Toy numbers: P(IsNext) = 0.28 for a pair that really is IsNext, so NSP loss ≈ 1.26.

Table 5 shows NSP mattered for this model: removing it hurt QNLI, MNLI and SQuAD (§5.1). Later work questioned
how much it helps (RoBERTa, Liu et al., 2019, dropped it), but that is outside this paper.

## Adding them: the total pre-training loss (Appendix A.2)

> "The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction
> likelihood." (Appendix A.2; "likelihood" here means the negative log-likelihood that is minimized.)

```
total loss = MLM loss + NSP loss = 3.68 + 1.26 = 4.94          (one number)
```

What is added are **two numbers that measure how wrong each head was**. The heads' outputs are never added:
one is a probability over 30,000 pieces for each masked position (k × V), the other is a probability for
IsNext and NotNext (1 × 2). They don't even have the same shape.

The sum exists so that a single backward pass improves both tasks at once. Both losses flow back into the
**same** encoder and embedding tables, so the shared weights are pulled toward representations that are good at
filling blanks *and* at relating two sentences. The MLM loss also reaches the token embedding table twice,
through the input lookup and through the tied output layer.

The sum is used only during pre-training. It is never computed when BERT is used.

---

## Using BERT: inference

Inference means running the model once on new text and reading off an answer. No loss, no backward pass, no
dropout and no random masking. What sits on top of the encoder depends on what you want.

### Way 1: fill in the blank, with the MLM head alone

Put `[MASK]` exactly where you want a guess (no 80/10/10, that was a training trick), run one forward pass,
and read the softmax at that position. The NSP head and pooler aren't used and nothing is added.

```
input   [CLS] my dog is [MASK] [SEP] he likes play ##ing [SEP]
T       (11 × H)     one pass, every position at once
head    T[4] → transform → · E_tokᵀ → softmax → top guesses for position 4
```

Two things to notice, compared with the Transformer's decoder from Attention, by hand:

- **One pass, not a loop.** The decoder wrote a translation one word at a time and fed each word back in. BERT
  sees the whole input at once and predicts every [MASK] in the same pass.
- **Several blanks are guessed independently.** Two [MASK]s in one input each get their own distribution, and
  neither sees the other's guess. That is one reason BERT is not used to generate long text.

Toy result with random weights: the top guess for position 4 is "apple". A trained BERT would put most of the
probability on words like "cute".

### Way 2: fine-tune, then run the fine-tuned model (§3.2, §4, Figure 1 right, Figure 4)

For a task such as entailment, sentiment, question answering or named-entity tagging, both pre-training heads
are thrown away. One small task layer is put on top of the same encoder, and *all* the parameters are trained a
little further on labelled data (§3.2: "fine-tune all the parameters end-to-end"; 3 epochs for GLUE, §4.1).
Then inference is one forward pass through the fine-tuned model.

| task | input | read from | new layer | output |
|---|---|---|---|---|
| sentence-pair classification (MNLI) | premise as A, hypothesis as B | C | W ∈ R^{K×H} (§4.1) | softmax(C Wᵀ): K labels |
| single-sentence classification (SST-2) | sentence as A, B empty | C | W ∈ R^{K×H} | softmax(C Wᵀ) |
| question answering (SQuAD v1.1) | question as A, paragraph as B | every T_i in B | start S ∈ R^H, end E ∈ R^H (§4.2) | P_i = e^{S·T_i} / Σ_j e^{S·T_j}; best span i ≤ j maximizing S·T_i + E·T_j |
| tagging (CoNLL NER) | sentence | every T_i (first sub-token of each word) | a token classifier (§5.3) | one label per word |

§4.1 writes the classification head on C directly. The released `run_classifier.py` feeds it the pooled output
(the same dense + tanh as NSP) **[code]**.

### Way 3: features (§5.3)

You can also freeze BERT and read hidden vectors out of it as features for another model. For NER,
concatenating the top four layers' T_i gets 96.1 dev F1, 0.3 behind fine-tuning the whole model (Table 7).

### Training vs. using, side by side

| | pre-training | fill-mask | fine-tuned task |
|---|---|---|---|
| random masking (15%, 80/10/10) | yes | no: you place [MASK] | no |
| MLM head | yes | yes | removed |
| NSP head + pooler | yes | unused | removed (the pooler may be reused **[code]**) |
| task layer | none | none | new, small |
| loss | MLM + NSP, summed | none | task loss while fine-tuning; none when used |
| dropout | 0.1 | off | 0.1 while fine-tuning; off when used |
| forward passes per input | one | one | one |

## Learned parameters on top of the encoder (BERT-base)

| Part | Shapes | Count |
|---|---|---|
| MLM transform **[code]** | 768 × 768 + 768, LayerNorm 2 × 768 | 592,128 |
| MLM output | the token embedding table (shared), plus a bias per piece | 30,522 new |
| Pooler **[code]** | 768 × 768 + 768 | 590,592 (already in the 110M) |
| NSP classifier **[code]** | 2 × 768 + 2 | 1,538 |
| A GLUE head (§4.1) | K × 768 (+ K) | 2,307 for K = 3 |

## Self-check

1. What shape are the MLM logits for a 128-token sequence with 19 masked positions?
2. Which vector does the NSP head read, and what layer does the released code put in front of the classifier?
3. The total loss is 4.94. Which two numbers were added, and what does each one measure?
4. Why can't you add the MLM head's output to the NSP head's output?
5. You want BERT's best guess for a blank. Which heads run, and how many forward passes?
6. After fine-tuning BERT for sentiment, what happens to the MLM and NSP heads?

Answers: (1) 19 × 30,000 (19 × 30,522 in the code); (2) C, the [CLS] vector, through a dense + tanh pooler;
(3) the mean MLM loss (how surprised the model was by the hidden tokens) and the NSP loss (how surprised it was
by the IsNext/NotNext label); (4) one is a distribution over the vocabulary per masked position, the other a
distribution over two labels; they mean different things and have different shapes, so only their losses,
which are single numbers, are summed; (5) the encoder and the MLM head only, one pass; (6) they are dropped and
replaced by a small classifier on C.
