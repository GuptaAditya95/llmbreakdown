# Transformer encoder, step by step with shapes

This follows one sentence from raw text to the encoder output, using the base model sizes from the paper (Table 3, §3).

| Symbol | Meaning | Value in the paper |
|---|---|---|
| n | number of tokens in the sentence | 6 in this example |
| V | vocabulary size | ~37,000 (shared BPE vocabulary, EN-DE, §5.1) |
| d_model | width of every vector moving through the model | 512 |
| h | number of attention heads | 8 |
| d_k = d_v | width of each head's queries/keys and values | 512 / 8 = 64 |
| d_ff | inner width of the feed-forward network | 2048 |
| N | number of stacked encoder layers | 6 |

Shapes are written as `(rows × columns)`. One row is one token. Real training also adds a batch dimension in front, so `(n × 512)` becomes `(batch × n × 512)`. Every step below works the same way for each sentence in the batch.

---

## Step 0: Tokenize

Sentence: **"the cat sat on the mat"**

The tokenizer turns it into vocabulary IDs. The numbers here are made up for illustration:

```
tokens = [the, cat, sat, on, the, mat]
ids    = [  5, 912, 4031, 27,   5, 2210]       shape: (6,)
```

The two "the" tokens get the same ID. Only their position will tell them apart.

## Step 1: Embedding lookup (§3.4)

The embedding matrix `W_emb` has shape **(V × d_model) = (37000 × 512)**, one learned row per vocabulary entry.

The lookup just picks out the rows for our IDs. It's the same as multiplying a one-hot matrix `(6 × 37000)` by `W_emb`, but no real multiplication happens.

```
E = W_emb[ids]                                 (6 × 512)
E = E * sqrt(512)  ≈ E * 22.6                  (6 × 512)   ← scaling from §3.4
```

Both "the" rows in E are still identical.

## Step 2: Add positional encoding (§3.5)

PE is a fixed `(n × 512)` matrix, so nothing in it is learned. For position `pos` and dimension pair `i`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/512))
PE[pos, 2i+1] = cos(pos / 10000^(2i/512))
```

Each row is a different pattern of sine waves, and dimensions further along the row oscillate more slowly. The paper picked this form because PE[pos+k] is a linear function of PE[pos], which makes relative offsets easy for the model to learn.

```
X0 = E + PE                                    (6 × 512)   ← element-wise addition
X0 = dropout(X0, p=0.1)                        (§5.4)
```

The two "the" rows are now different, because one sits at position 0 and the other at position 4.

**X0 is the input to encoder layer 1.** Q, K and V are only created after this, inside the layer.

---

## Step 3: One encoder layer

Each of the N = 6 layers takes a `(6 × 512)` matrix and returns a `(6 × 512)` matrix. Call the input X.

### 3a. Make Q, K, V for each head (§3.2.2)

Each head i has its own three learned matrices:

```
W_i^Q : (512 × 64)    W_i^K : (512 × 64)    W_i^V : (512 × 64)
```

All three are applied to the **same** X, which is what makes this *self*-attention:

```
Q_i = X · W_i^Q      (6 × 512)·(512 × 64) = (6 × 64)
K_i = X · W_i^K      (6 × 64)
V_i = X · W_i^V      (6 × 64)
```

Each token now has a 64-wide query ("what am I looking for"), key ("what do I offer to be matched on") and value ("what I pass along if chosen"), separately in each head.

### 3b. Scaled dot-product attention in each head (§3.2.1, Eq. 1)

```
S_i = Q_i · K_iᵀ          (6 × 64)·(64 × 6) = (6 × 6)
```

`S_i[r, c]` is the dot product of token r's query with token c's key, which measures how much token r should look at token c. This compares every token with every token, all at once.

```
S_i = S_i / sqrt(64) = S_i / 8           (6 × 6)   ← divide every entry by 8
A_i = softmax(S_i)  applied to each row    (6 × 6)   ← each row now adds up to 1
```

Row r of `A_i` holds token r's attention weights over all 6 tokens. For example, in some head the row for "sat" might put most of its weight on "cat".

```
head_i = A_i · V_i        (6 × 6)·(6 × 64) = (6 × 64)
```

Each output row is a **weighted average of the value vectors**, using that token's attention weights.

### 3c. Combine the heads (§3.2.2)

```
Concat(head_1, …, head_8)       8 × (6 × 64)  →  (6 × 512)   ← placed side by side
MHA = Concat · W^O              (6 × 512)·(512 × 512) = (6 × 512)
```

In practice the eight `(512 × 64)` matrices for Q are stored as one `(512 × 512)` matrix, and likewise for K and V, so a single matrix multiply does all heads. The math is identical.

### 3d. Residual connection and layer norm (§3.1)

```
X1 = LayerNorm(X + dropout(MHA))        (6 × 512)
```

- **Residual:** add the layer's input X back in, element by element. The attention result is an *update* on top of what each token already was.
- **LayerNorm:** for **each row (token) separately**, subtract that row's mean over its 512 numbers, divide by its standard deviation, then apply a learned scale γ and shift β (each of size 512). This is the one place in the model where mean/std normalization actually happens. Attention itself only divides by √d_k.

### 3e. Position-wise feed-forward network (§3.3, Eq. 2)

```
W1 : (512 × 2048), b1 : (2048)     W2 : (2048 × 512), b2 : (512)

H   = ReLU(X1 · W1 + b1)          (6 × 512)·(512 × 2048) = (6 × 2048)   ← expand
F   = H · W2 + b2                 (6 × 2048)·(2048 × 512) = (6 × 512)   ← compress back
```

The same W1 and W2 are applied to every row on its own. Token 3's output depends only on token 3's row, so there is **no mixing between tokens here**. All mixing happened in attention.

### 3f. Residual connection and layer norm again

```
X_out = LayerNorm(X1 + dropout(F))      (6 × 512)
```

That's one layer: **(6 × 512) in, (6 × 512) out.**

---

## Step 4: Stack 6 layers (§3.1)

```
X0 → Layer1 → X_1 → Layer2 → X_2 → … → Layer6 → X_6
     each (6 × 512)
```

Every layer has the same structure but **its own weights**. Lower layers tend to capture local or syntactic relations, and higher layers more abstract ones. That's a general observation, not a claim made in the paper.

## Step 5: The encoder output

```
Z = X_6                 (6 × 512)
```

Z holds one 512-number vector per input token, and each vector now reflects that token *in the context of the whole sentence*. Z isn't turned into words. It's handed to every decoder layer, where it is projected into the keys and values for the decoder's encoder-decoder attention. That's where the decoder walkthrough will pick up.

---

## A tiny worked attention example with real numbers

The same steps at toy size: **3 tokens, d_model = 4, a single head with d_k = 2.** I computed these with numpy and rounded them to 2 decimals.

**Embeddings E (3 × 4)** and **positional encodings PE (3 × 4)** (using the sin/cos formula with d = 4):

```
E  = [[1, 0, 1, 0],        PE = [[0.00,  1.00, 0.00, 1.00],
      [0, 1, 0, 1],              [0.84,  0.54, 0.01, 1.00],
      [1, 1, 0, 0]]              [0.91, -0.42, 0.02, 1.00]]

X = E + PE = [[1.00, 1.00, 1.00, 1.00],
              [0.84, 1.54, 0.01, 2.00],
              [1.91, 0.58, 0.02, 1.00]]
```

**Weights (4 × 2)**, made up for illustration (in a real model they're learned):

```
W^Q = [[1,0],[0,1],[1,0],[0,1]]   W^K = [[0,1],[1,0],[0,1],[1,0]]   W^V = [[1,0],[0,0],[0,1],[1,1]]
```

**Projections (3 × 2):**

```
Q = X·W^Q = [[2.00, 2.00],   K = X·W^K = [[2.00, 2.00],   V = X·W^V = [[2.00, 2.00],
             [0.85, 3.54],                [3.54, 0.85],                [2.84, 2.01],
             [1.93, 1.58]]                [1.58, 1.93]]                [2.91, 1.02]]
```

Example entry: Q[0,0] = 1·1 + 1·0 + 1·1 + 1·0 = 2.

**Scores Q·Kᵀ (3 × 3):**

```
[[8.00, 8.78, 7.03],
 [8.78, 6.03, 8.18],
 [7.03, 8.18, 6.11]]
```

Example entry: row 0 · key 1 = 2·3.54 + 2·0.85 = 8.78.

**Divide by √2 ≈ 1.414:**

```
[[5.66, 6.21, 4.97],
 [6.21, 4.26, 5.78],
 [4.97, 5.78, 4.32]]
```

**Softmax over each row (each row adds up to 1):**

```
A = [[0.31, 0.54, 0.15],     ← token 0 looks mostly at token 1
     [0.56, 0.08, 0.36],     ← token 1 barely looks at itself here
     [0.26, 0.60, 0.14]]
```

**Output A·V (3 × 2):**

```
[[2.59, 1.85],
 [2.40, 1.64],
 [2.63, 1.87]]
```

Example entry: row 0 = 0.31·[2.00, 2.00] + 0.54·[2.84, 2.01] + 0.15·[2.91, 1.02] ≈ [2.59, 1.85].

In the real model this happens in 8 heads at once with 64-wide vectors. The 8 outputs are concatenated, multiplied by W^O, and fed into the residual connection, layer norm and FFN described above.

---

## Learned parameters per encoder layer

| Part | Shapes | Count |
|---|---|---|
| Q, K, V, O projections | 4 × (512 × 512) + biases | 1,050,624 |
| FFN | (512 × 2048) + (2048 × 512) + biases | 2,099,712 |
| Two LayerNorms | 2 × (γ, β of size 512) | 2,048 |
| **One layer** | | **3,152,384** |
| **Six layers** | | **≈ 18.9 M** |

The embedding matrix (37000 × 512 ≈ 18.9 M) is roughly as large as the whole encoder stack. The paper shares it between the encoder input, the decoder input and the output layer (§3.4).
