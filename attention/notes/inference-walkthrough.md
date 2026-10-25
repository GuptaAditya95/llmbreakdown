# Inference walkthrough: how the trained Transformer translates

Part 3 of *Attention, by hand*. Parts 1 and 2 ran one **training** pass: the English sentence went
through the encoder, the correct German sentence (shifted right) went through the decoder, and every
position was graded at once. This note follows what happens **after training**, when the model is given
an English sentence it has never seen and has to write the German itself.

Section numbers refer to the arXiv version of the paper,
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Shapes are `(rows × columns)`, one row per token;
real implementations add a batch dimension in front.

## Symbols

| symbol | meaning | this site (recommended) | paper |
|---|---|---|---|
| n | English (source) tokens | your sentence, up to 8 | any |
| t | decoder tokens read at the current turn (⟨s⟩ + words written so far) | 1, 2, 3, … | 1, 2, 3, … |
| T | number of turns until ⟨/s⟩ or the length limit | ≤ 8 | ≤ n + 50 |
| d_model | width of every token vector | 8 | 512 |
| h, d_k | heads, width per head | 2, 4 | 8, 64 |
| N | layers per stack | 6 | 6 |
| V | vocabulary size | your translation's words + ⟨/s⟩ | ≈ 37,000 |
| k | beam size | 2 | 4 |
| α | length-penalty exponent | 0.6 | 0.6 |

The weights are random stand-ins for learned values, so the words the toy model picks are nonsense.
The loop is the one a trained model runs; only the numbers inside differ.

## 1. Training vs. inference

| | training (Part 2) | inference (Part 3) |
|---|---|---|
| decoder input | the true translation, shifted right | only what the model has written so far |
| passes | 1 for the whole sentence | 1 per output token |
| output rows used | all of them (one loss each) | only the last |
| next input word | from the data (teacher forcing) | the model's own pick |
| stops | at the end of the sentence | when it picks ⟨/s⟩, or at a length limit |

The paper calls the decoder **auto-regressive**: "at each step the model is auto-regressive, consuming
the previously generated symbols as additional input when generating the next" (§3).

The mask (§3.2.3) is what makes this work. It trained every position to predict its next word while
seeing only the words to its left, which is exactly the situation at inference time.

## 2. Encode once

The English sentence is fully known, so the encoder runs **once** and produces

    Z: (n × d_model)                       paper (n × 512)

Every decoder layer's cross-attention turns Z into keys and values, K = Z·W_K and V = Z·W_V, one pair per
head: `(n × d_k)`, paper `(n × 64)`. They don't depend on the German words, so they're computed once and
reused at every turn. In the paper that's 6 layers × 8 heads × 2 = 96 matrices.

## 3. Start with ⟨s⟩

The decoder's input at turn 1 is one token, the start token ⟨s⟩:

    Y0 = E[⟨s⟩] × √d_model + PE[0]         (1 × d_model)       paper (1 × 512)

At position 0 every angle is 0, so PE[0] = (0, 1, 0, 1, …) (§3.5).

## 4. One turn of the loop (turn t)

1. **Run the decoder** (all N layers) on the t tokens so far. Masked self-attention scores are `(t × t)`,
   cross-attention scores `(t × n)`, the output `(t × d_model)`.
2. **Keep only the last row.** Row r predicts the token after position r, so every row but the last
   predicts a word already written. `(1 × d_model)`.
3. **Logits**: last row · W_embᵀ (the embedding table, tied with the output layer, §3.4).
   `(1 × V)`, paper `(1 × 37000)`.
4. **Softmax** → a probability for every vocabulary word, adding up to 1. `(1 × V)`.
5. **Choose a word.** Greedy decoding takes the argmax. Alternatives: beam search (§6, below) or
   sampling (draw a word in proportion to its probability).
6. **Append it** at position t and go again. The decoder input grows from `(t,)` to `(t + 1,)` tokens.

At step 6 the model reads its own pick, right or wrong. During training it always read the true word,
so it never practised recovering from its own mistakes. This mismatch is commonly called **exposure
bias** (an interpretation; the paper doesn't discuss it).

### Sampling and temperature (not in the paper)

Chat assistants usually sample instead of taking the argmax. A **temperature** τ divides the logits
before the softmax, `softmax(logits / τ)`: τ < 1 sharpens the distribution towards greedy, τ > 1
flattens it. Top-k and top-p ("nucleus", Holtzman et al., 2020) sampling only draw among the most likely
words.

## 5. Stopping

The loop stops when the chosen word is the end token ⟨/s⟩: every training target ended with it, so the
model learns when a translation is finished. Because a model may never pick it, there is a limit: the
paper sets the maximum output length to **input length + 50** (§6.1). The site caps the toy run at 8.

## 6. The KV cache

Because of the mask, row r depends only on rows 0 … r. Appending a token cannot change any earlier
row, in any layer. So recomputing them each turn is wasted work.

A **KV cache** stores each layer's self-attention keys and values for tokens already processed. Each
turn, only the new token's row goes through the stack:

| per head, per layer, at turn t | without cache | with cache |
|---|---|---|
| queries computed | t × d_k | 1 × d_k |
| new keys/values | t × d_k each | 1 × d_k each (appended) |
| scores | t × t (masked) | 1 × t (no mask needed) |
| output rows | t × d_model | 1 × d_model |

Over T turns that's T rows per layer instead of 1 + 2 + … + T = T(T + 1)/2.

At inference the mask has nothing left to do with a cache: the new token is the latest, and everything
in the cache is in its past.

Cache size in the paper's sizes: 6 layers × 2 (K and V) × t tokens × 512 numbers (8 heads × 64). After
30 tokens that's 184,320 numbers per sentence. Cross-attention K and V (section 2) are cached too, but
their size is fixed by n.

## 7. Beam search (§6.1)

Greedy commits to one word per turn. Beam search keeps the **k best partial translations**:

1. Extend each of the k beams by every vocabulary word: k × V candidates (paper: 4 × 37,000).
2. Score each by the **sum of the natural logs** of its words' probabilities.
3. Keep the top k. A candidate ending in ⟨/s⟩ is set aside as finished.
4. Repeat until k translations have finished (or the limit is reached).
5. Rank finished translations by score ÷ lp, with the **length penalty**
   `lp = ((5 + length) / 6)^α`, α = 0.6. Without it, every extra word adds a negative log and longer
   translations always lose.

The paper uses k = 4 and α = 0.6, chosen on the development set (§6.1). The length penalty comes from
Google's neural machine translation system (Wu et al., 2016, §7). The site uses k = 2 so every candidate
fits on the card, and ranks unfinished beams by their raw sum of logs, which is a simplification.

## 8. The whole inference

| step | name | this run | paper |
|---|---|---|---|
| encode the English, once | Z | n × 8 | n × 512 |
| cross K, V per layer and head, once | K, V | n × 4 | n × 64 |
| start: ⟨s⟩ + PE[0] | Y0 | 1 × 8 | 1 × 512 |
| turn t: decoder on t tokens | Y | t × 8 | t × 512 |
| … with a KV cache | new row | 1 × 8 | 1 × 512 |
| last row · W_embᵀ | logits | 1 × V | 1 × 37000 |
| softmax | P | 1 × V | 1 × 37000 |
| greedy / beam / sample | token | 1 | 1 |
| append, repeat until ⟨/s⟩ | output | T tokens | ≤ n + 50 |

## Self-check

1. How many times does the encoder run to translate one sentence? The decoder?
2. Why is only the last row of the decoder output used at each turn?
3. At turn 5 with a 6-word source, what shapes are one head's self- and cross-attention scores?
4. Why can a KV cache reuse old rows, and why is the mask unnecessary once you use one?
5. With k = 2, how many candidates are scored per turn for V = 37,000? Why divide by a length penalty?

Answers: (1) once; once per output token. (2) The others predict words already written. (3) 5 × 5 and
5 × 6. (4) The mask means old rows never see new tokens; the new token's past is the whole cache.
(5) 74,000; sums of negative logs always favour shorter outputs.

## Homework

`inference-practice.ipynb` (built by `make_inference_practice.py`) uses the tiny model from
`transformer-practice.ipynb`: greedy decoding of "the cat sat" by hand, one self-attention row with a KV
cache checked against the full recompute, and two turns of beam search with the length penalty.

## References

- Vaswani et al. (2017). Attention is all you need. arXiv:1706.03762.
- Wu et al. (2016). Google's neural machine translation system. arXiv:1609.08144.
- Holtzman et al. (2020). The curious case of neural text degeneration. arXiv:1904.09751.
