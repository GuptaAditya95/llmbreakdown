# Transformer decoder, step by step with shapes

This picks up where [encoder-walkthrough.md](encoder-walkthrough.md) ended. The encoder has turned **"the cat sat on the mat"** into Z, one 512-wide vector per English token. Now the decoder has to produce the German translation: **"die Katze saß auf der Matte"**.

Same sizes as before (base model, Table 3): d_model = 512, h = 8 heads, d_k = d_v = 64, d_ff = 2048, N = 6 layers.

| New symbol | Meaning | Value here |
|---|---|---|
| n | number of source (English) tokens | 6 |
| m | number of target (German) positions fed to the decoder | 7 |
| Z | encoder output | (6 × 512) |
| Y | decoder's running matrix | (7 × 512) |

---

## The big idea in one paragraph

The encoder reads the whole source sentence at once and every token can look at every other token. The decoder is a **next-word predictor**: at every position it answers "given the source sentence and the German words so far, what is the next German word?" Everything special about the decoder follows from that one job:

1. It must **not peek at future German words** → *masked* self-attention.
2. It must **look at the English sentence** → *cross-attention* to Z.
3. It must **turn a vector into a word** → final linear + softmax over the vocabulary.
4. It must have something to condition on before any word exists → a start token, and inputs **shifted right** by one.

---

## The decoder layer in one picture

The encoder layer had 2 sublayers. A decoder layer has **3**. The new one sits in the middle.

```
                        Y (7×512)  ← German so far (shifted right)
                         │
                         ├───────────────────────────┐
                         ▼                           │ residual
              Masked multi-head SELF-attention       │   Q, K, V all from Y
                         ▼                           │   (can only look left)
                         +  ◄────────────────────────┘
                         ▼
                     LayerNorm  →  Y1 (7×512)
                         │
                         ├───────────────────────────┐
 Z (6×512) ─────┐        ▼                           │ residual
 from encoder   └──► Multi-head CROSS-attention      │   Q from Y1
 (K and V)                ▼                          │   K, V from Z
                         +  ◄────────────────────────┘
                         ▼
                     LayerNorm  →  Y2 (7×512)
                         │
                         ├───────────────────────────┐
                         ▼                           │ residual
                  Feed-forward network               │   same as encoder
                         ▼                           │
                         +  ◄────────────────────────┘
                         ▼
                     LayerNorm  →  Y_out (7×512)  → next decoder layer
```

Every sublayer still has the form **LayerNorm(x + Sublayer(x))** (§3.1), so residuals and layer norm work exactly as in the encoder. Each of the 6 decoder layers has its own weights and receives the **same** Z.

---

## Step 0: Shift the target right

Add a start token `<s>` and an end token `</s>` to the German sentence. Then make two copies offset by one (§3.1, "output embeddings are offset by one position"):

```
position:        0     1      2     3    4    5      6
decoder input:  <s>   die   Katze  saß  auf  der   Matte      ← what the decoder sees
target:         die  Katze   saß   auf  der  Matte  </s>      ← what it must predict
```

Read it column by column. At position 2 the decoder has seen `<s> die Katze` and has to predict `saß`. Position *i* is always trying to predict the word that sits at input position *i + 1*. That is the whole meaning of "shifted right".

`<s>` exists so that position 0 has something to stand on: "nothing yet, predict the first word".

## Step 1: Embed and add positions

Exactly the encoder's Steps 1 and 2, applied to the 7 German input tokens:

```
Y0 = W_emb[ids] * sqrt(512) + PE[0:7]          (7 × 512)
Y0 = dropout(Y0, p=0.1)
```

The paper uses one shared embedding matrix for English input, German input and the output layer (§3.4), which works because the BPE vocabulary is shared between the two languages.

---

## Step 2: One decoder layer

### 2a. Masked self-attention (the German words look at each other)

Q, K and V all come from Y, just like encoder self-attention:

```
Q_i = Y · W_i^Q    K_i = Y · W_i^K    V_i = Y · W_i^V     each (7 × 64)
S_i = Q_i · K_iᵀ / 8                                      (7 × 7)
```

**The one new thing is the mask.** Before the softmax, every entry above the diagonal is set to −∞ (§3.2.3):

```
                 <s>  die  Katze saß  auf  der  Matte     ← keys (what can be looked at)
        <s>     [  ·   −∞   −∞   −∞   −∞   −∞   −∞ ]
        die     [  ·    ·   −∞   −∞   −∞   −∞   −∞ ]
        Katze   [  ·    ·    ·   −∞   −∞   −∞   −∞ ]
        saß     [  ·    ·    ·    ·   −∞   −∞   −∞ ]
        auf     [  ·    ·    ·    ·    ·   −∞   −∞ ]
        der     [  ·    ·    ·    ·    ·    ·   −∞ ]
        Matte   [  ·    ·    ·    ·    ·    ·    · ]
     ↑ queries
```

e^(−∞) = 0, so after the softmax those positions get **exactly zero weight**. Row "Katze" can only mix in `<s>`, `die` and itself.

**Why it's needed:** position 2 is supposed to *predict* `saß`. During training, `saß` is sitting right there at input position 3. Without the mask, the model would just copy it and learn nothing. The mask makes training honest: each position sees only what it would see at inference time.

**Tiny numeric example** (4 positions, one head, made-up scaled scores):

```
scaled scores S:                      softmax WITHOUT mask:
         <s>  die Katze saß                    <s>  die  Katze  saß
<s>    [2.0, 1.0, 0.5, 3.0]           <s>    [0.23, 0.09, 0.05, 0.63]   ← mostly looks at the FUTURE "saß"
die    [1.0, 2.5, 0.0, 1.5]           die    [0.13, 0.60, 0.05, 0.22]
Katze  [0.5, 1.0, 2.0, 0.0]           Katze  [0.13, 0.21, 0.58, 0.08]
saß    [1.5, 0.5, 1.0, 2.0]           saß    [0.28, 0.10, 0.17, 0.46]

add mask (−∞ above diagonal), then softmax:
         <s>   die  Katze  saß
<s>    [1.00, 0.00, 0.00, 0.00]   ← can only see itself
die    [0.18, 0.82, 0.00, 0.00]
Katze  [0.14, 0.23, 0.63, 0.00]
saß    [0.28, 0.10, 0.17, 0.46]   ← last row is unchanged: nothing to hide from it
```

Each row is renormalized over only the positions it's allowed to see. The rest is identical to encoder attention: weighted sum of V, concat 8 heads, multiply by W^O.

```
Y1 = LayerNorm(Y + dropout(MaskedMHA(Y)))           (7 × 512)
```

### 2b. Cross-attention (the German words look at the English sentence)

This is the bridge between encoder and decoder, and the paper calls it "encoder-decoder attention" (§3.2.3). The only difference from self-attention is **where Q, K and V come from**:

```
Q_i = Y1 · W_i^Q        (7 × 512)·(512 × 64) = (7 × 64)    ← from the DECODER
K_i = Z  · W_i^K        (6 × 512)·(512 × 64) = (6 × 64)    ← from the ENCODER
V_i = Z  · W_i^V        (6 × 512)·(512 × 64) = (6 × 64)    ← from the ENCODER
```

Intuition: the **question** comes from the German side ("I'm the position about to produce the word after *die*; which English words matter to me?"). The **things being searched** and **the information handed back** come from the English side.

```
S_i = Q_i · K_iᵀ / 8      (7 × 64)·(64 × 6) = (7 × 6)      ← not square any more!
A_i = softmax(S_i) per row                  (7 × 6)
head_i = A_i · V_i        (7 × 6)·(6 × 64)  = (7 × 64)
```

Row r of A_i is German position r's attention over the **6 English words**. Then concat 8 heads → (7 × 512), multiply by W^O → (7 × 512). The output has one row per *German* position, because the queries decide the number of rows.

**No mask here.** The whole English sentence is known from the start, so every German position may look at every English word.

**Shape rule worth memorizing:** output rows = number of queries, the softmax runs over the number of keys. Self-attention: 7 queries, 7 keys. Cross-attention: 7 queries, 6 keys.

**Tiny numeric example** (2 German queries, 3 English words, d_k = 2, made-up vectors):

```
Q (from decoder)       K (from encoder)          V (from encoder)
die   [1, 0]           the  [2, 0]               the  [1.0, 0.0]
Katze [0, 1]           cat  [0, 2]               cat  [0.0, 1.0]
                       mat  [1, 1]               mat  [0.5, 0.5]

Q·Kᵀ:           the   cat   mat         ÷ √2, softmax:     the   cat   mat
die          [  2,    0,    1 ]         die             [ 0.58, 0.14, 0.28 ]
Katze        [  0,    2,    1 ]         Katze           [ 0.14, 0.58, 0.28 ]

A·V:   die   → [0.72, 0.28]      (mostly "the"'s information)
       Katze → [0.28, 0.72]      (mostly "cat"'s information)
```

In a trained model this is the alignment you'd hope for: the position predicting the word after `die` pulls in what it needs from `cat`. (In this toy it's by construction.)

```
Y2 = LayerNorm(Y1 + dropout(CrossMHA(Q from Y1, K/V from Z)))     (7 × 512)
```

### 2c. Feed-forward network

Identical to the encoder: expand to 2048, ReLU, compress to 512, each row on its own.

```
Y_out = LayerNorm(Y2 + dropout(FFN(Y2)))              (7 × 512)
```

---

## Step 3: Stack 6 decoder layers

```
             Z (6×512) ──┬──────────┬──────────┬─── … ───┐
                         ▼          ▼          ▼         ▼
Y0 (7×512) → DecLayer1 → DecLayer2 → DecLayer3 → … → DecLayer6 → Y_6 (7×512)
```

Two things people often get wrong:

- Every decoder layer gets the **final** encoder output Z (from encoder layer 6), not the matching encoder layer's output.
- Each decoder layer has its **own** W^K and W^V for cross-attention, so each layer reads Z in its own way, even though Z itself is the same.

## Step 4: Linear + softmax → word probabilities (§3.4)

```
logits = Y_6 · W_embᵀ          (7 × 512)·(512 × 37000) = (7 × 37000)
P      = softmax(logits) per row                         (7 × 37000)
```

Row r is a probability distribution over the whole vocabulary for "the word at target position r". W_embᵀ is the same embedding matrix used at the input, transposed (weight tying, §3.4). Intuitively each logit is a dot product between the decoder's vector and a word's embedding: "which word's embedding is most like what I'm saying here?"

---

## Step 5: How decoding proceeds. Training vs inference

This is the part that makes the decoder feel different from the encoder.

### Training: all positions at once (teacher forcing)

We already know the correct German sentence, so we feed the whole shifted input in one pass:

```
input:   <s>  die  Katze  saß  auf  der  Matte
              ─── one forward pass, all 7 positions in parallel ───
P:       7 rows of probabilities
target:  die Katze  saß   auf  der  Matte  </s>

loss = average over positions of  −log P[r, target_r]     (cross-entropy, label smoothing 0.1, §5.4)
```

The decoder input is always the **true** previous words, even if the model would have predicted something else ("teacher forcing"). The mask is what makes this legal: 7 predictions are computed at once, but each one only saw the words to its left. So one forward pass gives 7 training examples.

The encoder also runs once on the English sentence, and Z is shared by all 7 positions.

### Inference: one word at a time (autoregressive)

Now there is no German sentence to feed in. The model builds it from its own predictions:

```
Encoder runs ONCE:  "the cat sat on the mat" → Z (6×512)

step 1: input [<s>]                               → last row of P → pick "die"
step 2: input [<s>, die]                          → last row of P → pick "Katze"
step 3: input [<s>, die, Katze]                   → last row of P → pick "saß"
step 4: input [<s>, die, Katze, saß]              → pick "auf"
step 5: input [<s>, die, Katze, saß, auf]         → pick "der"
step 6: input [<s>, die, Katze, saß, auf, der]    → pick "Matte"
step 7: input [<s>, die, Katze, saß, auf, der, Matte] → pick "</s>"  → stop
```

At step t, the decoder input is t tokens, so its matrices are (t × 512), cross-attention scores are (t × 6), and only the **last row** of P is used, because that's the prediction for the next word. Z doesn't change between steps.

The mask still matters at inference: it keeps every position computing exactly what it computed during training, so the model behaves the way it was trained.

Two practical details:

- **Choosing the word.** "Pick" above is greedy (take the highest probability). The paper actually uses **beam search** with beam size 4 and length penalty α = 0.6, keeping the 4 best partial translations at each step, and caps output at source length + 50 (§6.1).
- **Not redoing work.** Because of the mask, earlier positions' outputs never change when a new word is appended. Real implementations cache each layer's K and V for past positions (a "KV cache") and only compute the new row each step. That's an implementation trick, not in the paper, but it's why the mask makes inference efficient as well as correct.

---

## Side-by-side: encoder layer vs decoder layer

| | Encoder layer | Decoder layer |
|---|---|---|
| Sublayers | 2 | 3 |
| Self-attention | sees all tokens | **masked**: sees only itself and earlier tokens |
| Cross-attention | none | Q from decoder, K and V from Z |
| FFN | same | same |
| Residual + LayerNorm | after each sublayer | after each sublayer |
| Input | source tokens + PE | target tokens **shifted right** + PE |
| Runs at inference | once | once per generated word |

## Learned parameters per decoder layer

| Part | Count |
|---|---|
| Masked self-attention (Q, K, V, O + biases) | 1,050,624 |
| Cross-attention (Q, K, V, O + biases) | 1,050,624 |
| FFN | 2,099,712 |
| Three LayerNorms | 3,072 |
| **One layer** | **4,204,032** |
| **Six layers** | **≈ 25.2 M** |

Encoder (≈ 18.9 M) + decoder (≈ 25.2 M) + shared embedding (≈ 18.9 M) ≈ 63 M, which matches the paper's 65 M for the base model (Table 3) once you account for the exact vocabulary size.

---

## Self-check questions

1. In cross-attention, why are there 7 output rows and not 6?
2. What would go wrong during training if you removed the mask but kept everything else?
3. At inference step 4, what are the shapes of the decoder's self-attention scores and its cross-attention scores?
4. Why does decoder input position 0 hold `<s>` instead of `die`?
5. Does decoder layer 3 attend to encoder layer 3's output? What does it attend to?

<details><summary>Answers</summary>

1. Rows come from the queries, and the queries come from the 7 German positions. The 6 English words are only the keys and values.
2. Each position could attend to the next input token, which *is* its target. It would learn to copy, get near-zero training loss, and fail completely at inference where that token doesn't exist yet.
3. Input is 4 tokens: self-attention scores (4 × 4) with the mask, cross-attention scores (4 × 6).
4. Position 0 must predict the first word, `die`. If `die` were its input, it would be predicting something it already sees. `<s>` means "nothing yet".
5. No. Every decoder layer attends to Z, the output of the last (6th) encoder layer, through its own W^K and W^V.

</details>
