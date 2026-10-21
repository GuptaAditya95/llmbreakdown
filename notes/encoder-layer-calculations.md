# One encoder layer, every calculation

This continues from [positional-encoding-calculations.md](positional-encoding-calculations.md). We take its output X, the input for "the cat sat on the mat", and push it through **one complete encoder layer** with real numbers.

## Sizes

| | Toy version here | Paper (base model) |
|---|---|---|
| tokens n | 6 | any |
| d_model | 8 | 512 |
| heads h | 2 | 8 |
| d_k = d_v = d_model / h | 4 | 64 |
| d_ff (FFN inner width) | 32 (4 × d_model) | 2048 (4 × d_model) |

The weights below are random numbers rounded to one decimal, because a real model *learns* them during training. So the attention patterns you'll see are arbitrary and don't mean anything linguistic yet. The point is to see every operation and shape. I left out biases and dropout to keep it readable, and LayerNorm uses its starting values γ = 1, β = 0.

Everything was computed with [encoder_layer_example.py](encoder_layer_example.py), which you can rerun. It prints every matrix in full, including the large FFN weights not shown here.

## The layer in one picture

```
X (6×8)
 │
 ├──────────────────────────────┐
 ▼                              │ residual (skip connection)
Multi-head self-attention       │
 ▼                              │
 +  ◄───────────────────────────┘
 ▼
LayerNorm  →  X1 (6×8)
 │
 ├──────────────────────────────┐
 ▼                              │ residual
Feed-forward network            │
 ▼                              │
 +  ◄───────────────────────────┘
 ▼
LayerNorm  →  X_out (6×8)   → into layer 2
```

The paper writes each block as **LayerNorm(x + Sublayer(x))** (§3.1).

---

## Input X (6 × 8)

```
         d0     d1     d2     d3     d4     d5     d6     d7
the  [ 0.10,  1.20, -0.30,  1.40,  0.00,  1.10, -0.20,  1.30]
cat  [ 1.14,  0.44,  0.60,  1.20, -0.39,  1.10,  0.60,  0.80]
sat  [ 0.71, -0.02,  0.30,  0.68,  0.22,  1.50, -0.10,  1.00]
on   [ 0.14, -1.29,  0.50,  1.06,  0.33,  0.80,  0.40,  1.10]
the  [-0.66, -0.45,  0.09,  1.32,  0.04,  1.10, -0.20,  1.30]
mat  [-0.56,  0.28,  0.38,  1.18, -0.15,  1.20,  0.10,  0.60]
```

---

# Sublayer 1: Multi-head self-attention

**What it's for.** This is the only place where tokens exchange information. Each token asks "which other tokens are relevant to me?" and pulls in a weighted mix of them.

## Step 1: Project X into Q, K and V (head 1)

Each head has three weight matrices of shape **(d_model × d_k) = (8 × 4)**:

```
W_Q1 (8×4)               W_K1 (8×4)               W_V1 (8×4)
[ 0.1 -0.1  0.3  0.1]    [ 0.2 -0.1  0.8  0.7]    [-0.3 -0.5 -0.5  0.1]
[-0.3  0.2  0.7  0.5]    [ 0.3 -1.1  0.0  0.3]    [ 0.2  0.7  0.0  0.5]
[-0.4 -0.6 -0.3  0.0]    [ 0.5 -0.3  0.9 -0.7]    [ 0.7  0.6 -1.2  0.6]
[-1.2 -0.1 -0.6 -0.4]    [-0.3  0.5  0.0  1.0]    [ 0.2  0.2  0.2  0.2]
[-0.3 -0.2  0.2  0.5]    [ 0.1 -0.3 -0.2 -0.5]    [ 0.2 -0.2 -1.0 -0.1]
[-0.1  0.7 -0.3  0.2]    [-0.6  0.3  0.3  0.6]    [-0.4  0.5 -0.1  0.0]
[ 0.5  0.0 -0.4 -0.5]    [-0.4  0.8 -0.1  0.8]    [-0.4 -0.3  0.0 -0.7]
[-0.2  0.1 -0.5 -0.1]    [-0.2 -0.4  0.1  0.5]    [ 0.2 -0.1 -0.6 -1.2]
```

Multiply: **(6 × 8) · (8 × 4) = (6 × 4)**.

```
Q1 = X·W_Q1                 K1 = X·W_K1                 V1 = X·W_V1
the [-2.38  1.17 -0.78  0.24]  [-1.03 -0.89  0.29  3.19]  [ 0.18  1.37 -0.30 -0.71]
cat [-1.55  0.42 -1.30 -0.50]  [-0.80  0.43  1.88  3.44]  [-0.19  0.71 -1.25 -0.41]
sat [-1.32  0.78 -1.17  0.15]  [-0.96  0.10  1.35  2.17]  [-0.19  0.58 -1.55 -0.77]
on  [-1.27 -0.07 -2.53 -1.04]  [-1.25  1.81  0.81  1.61]  [ 0.07 -0.36 -1.53 -1.75]
the [-2.03  0.68 -2.22 -0.61]  [-1.45  0.83  0.03  1.79]  [ 0.34  0.81 -0.44 -1.40]
mat [-1.85  0.70 -1.52 -0.33]  [-1.09  0.47  0.33  1.78]  [ 0.30  1.48 -0.27 -0.23]
```

**One entry by hand.** Q1[the, 0] = row "the" of X · column 0 of W_Q1:

```
0.10·0.1 + 1.20·(−0.3) + (−0.30)·(−0.4) + 1.40·(−1.2) + 0.00·(−0.3) + 1.10·(−0.1) + (−0.20)·0.5 + 1.30·(−0.2)
= 0.01 − 0.36 + 0.12 − 1.68 + 0 − 0.11 − 0.10 − 0.26 = −2.38
```

The roles:
- **Q (query):** what this token is looking for.
- **K (key):** what this token can be matched on.
- **V (value):** what this token hands over if it gets picked.

They're all made from the same X, just with different learned weights.

## Step 2: Scores, Q · Kᵀ

**(6 × 4) · (4 × 6) = (6 × 6)**. Entry [r, c] is how well token r's query matches token c's key.

```
S1 = Q1·K1ᵀ      the    cat    sat     on    the    mat     ← keys
query the   [  1.95   1.76   1.86   4.85   4.84   3.30]
query cat   [ -0.75  -2.75  -1.32   0.86   1.68   0.56]
query sat   [  0.81  -0.28   0.09   2.38   2.82   1.68]
query on    [ -2.68  -7.36  -4.48  -2.25  -0.14  -1.35]
query the   [ -1.10  -4.36  -2.32   1.01   2.38   0.70]
query mat   [ -0.22  -2.23  -0.94   1.82   2.64   1.24]
```

By hand, S1[the, on] = Q1[the] · K1[on] = (−2.38)(−1.25) + (1.17)(1.81) + (−0.78)(0.81) + (0.24)(1.61) = 2.98 + 2.12 − 0.63 + 0.39 = **4.85**

## Step 3: Scale by √d_k

Divide every entry by √4 = 2 (in the paper it's √64 = 8):

```
S1 / 2         the    cat    sat     on    the    mat
the         [ 0.97   0.88   0.93   2.43   2.42   1.65]
cat         [-0.38  -1.37  -0.66   0.43   0.84   0.28]
sat         [ 0.40  -0.14   0.05   1.19   1.41   0.84]
on          [-1.34  -3.68  -2.24  -1.13  -0.07  -0.68]
the         [-0.55  -2.18  -1.16   0.51   1.19   0.35]
mat         [-0.11  -1.12  -0.47   0.91   1.32   0.62]
```

**Why scale?** Dot products of 64-number vectors can get large. A softmax over large numbers puts almost all its weight on one entry, and its gradients become tiny, so training stalls (§3.2.1). Dividing by √d_k keeps the numbers in a reasonable range.

## Step 4: Softmax on each row

For each row, take e^x of every entry, then divide by the row's total. The result is positive and each row adds up to 1.

Row "the", by hand:

```
e^0.97, e^0.88, e^0.93, e^2.43, e^2.42, e^1.65
=  2.64,  2.41,  2.53, 11.36, 11.25,  5.21      total = 35.40
÷ 35.40 → 0.07, 0.07, 0.07, 0.32, 0.32, 0.15
```

Full attention weights:

```
A1             the    cat    sat     on    the    mat     (each row adds up to 1)
the         [ 0.07   0.07   0.07   0.32   0.32   0.15]
cat         [ 0.10   0.04   0.08   0.23   0.35   0.20]
sat         [ 0.11   0.07   0.08   0.25   0.31   0.18]
on          [ 0.12   0.01   0.05   0.15   0.43   0.24]
the         [ 0.08   0.02   0.04   0.23   0.45   0.19]
mat         [ 0.09   0.03   0.06   0.25   0.38   0.19]
```

Read row "the" as: the first "the" takes 32% of its new content from "on", 32% from the second "the", and so on. In a trained model these patterns become meaningful, for example "sat" attending strongly to "cat".

## Step 5: Weighted sum of the values

**(6 × 6) · (6 × 4) = (6 × 4)**. Each output row is that token's weights applied to the V rows.

```
head1 = A1·V1
the [ 0.16  0.55 -0.89 -1.17]
cat [ 0.19  0.71 -0.76 -1.09]
sat [ 0.17  0.67 -0.81 -1.08]
on  [ 0.24  0.85 -0.61 -1.05]
the [ 0.23  0.71 -0.70 -1.16]
mat [ 0.20  0.67 -0.77 -1.13]
```

By hand, head1[the, 0] = 0.07·0.18 + 0.07·(−0.19) + 0.07·(−0.19) + 0.32·0.07 + 0.32·0.34 + 0.15·0.30 ≈ **0.16**

Steps 2 to 5 together are Equation 1 of the paper: **Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V**

## Step 6: Head 2 does the same with its own weights

Head 2 has separate W_Q2, W_K2 and W_V2 (shown in the script output), so it learns to look for different relationships. Its attention weights:

```
A2             the    cat    sat     on    the    mat
the         [ 0.03   0.01   0.08   0.19   0.47   0.22]
cat         [ 0.07   0.02   0.06   0.12   0.47   0.25]
sat         [ 0.13   0.06   0.12   0.14   0.29   0.26]
on          [ 0.16   0.12   0.17   0.13   0.17   0.24]
the         [ 0.07   0.05   0.14   0.21   0.30   0.23]
mat         [ 0.08   0.04   0.10   0.17   0.38   0.23]

head2 = A2·V2  (6×4)
the [ 0.44 -0.34 -0.23 -0.72]
cat [ 0.56 -0.38 -0.18 -0.82]
sat [ 0.73 -0.39 -0.20 -0.87]
on  [ 0.89 -0.40 -0.21 -0.92]
the [ 0.58 -0.35 -0.25 -0.74]
mat [ 0.58 -0.36 -0.22 -0.79]
```

The paper runs 8 heads in parallel. One might track subject-verb links, another the neighbouring word, and so on.

## Step 7: Concatenate the heads and project with W_O

Place the heads side by side: 2 × (6 × 4) gives **(6 × 8)**.

```
Concat      ←──── head1 ────→       ←──── head2 ────→
the [ 0.16  0.55 -0.89 -1.17 | 0.44 -0.34 -0.23 -0.72]
cat [ 0.19  0.71 -0.76 -1.09 | 0.56 -0.38 -0.18 -0.82]
...
```

Then multiply by W_O, which is **(h·d_v × d_model) = (8 × 8)** here and (512 × 512) in the paper. This mixes what the heads found back into one d_model-wide vector per token.

```
MHA = Concat·W_O  (6×8)
the [-0.24  0.08  0.19 -2.27  0.14  1.04  0.64 -1.59]
cat [-0.31  0.20  0.13 -1.90  0.13  0.96  0.58 -1.66]
sat [-0.36  0.32  0.23 -1.99  0.08  0.90  0.53 -1.82]
on  [-0.47  0.53  0.17 -1.58 -0.03  0.82  0.46 -1.97]
the [-0.34  0.27  0.13 -1.91  0.03  0.96  0.57 -1.73]
mat [-0.32  0.24  0.16 -1.98  0.07  0.97  0.58 -1.71]
```

The attention sublayer took (6 × 8) in and gives (6 × 8) out.

## Step 8: Add the residual, then LayerNorm

**Residual connection:** add the original X back in, element by element.

```
X + MHA  (6×8)
the [-0.14  1.28 -0.11 -0.87  0.14  2.14  0.44 -0.29]
cat [ 0.83  0.64  0.73 -0.70 -0.26  2.06  1.18 -0.86]
sat [ 0.35  0.30  0.53 -1.31  0.30  2.40  0.43 -0.82]
on  [-0.33 -0.76  0.67 -0.52  0.30  1.62  0.86 -0.87]
the [-1.00 -0.18  0.22 -0.59  0.07  2.06  0.37 -0.43]
mat [-0.88  0.52  0.54 -0.80 -0.08  2.17  0.68 -1.11]
```

**Why add X back?** Attention produces an *update* to each token, not a replacement, so the token keeps its own identity and position information. Residual connections also let gradients flow straight through 6 or more stacked layers, which makes deep models trainable.

**LayerNorm:** for **each row separately**, subtract that row's mean and divide by its standard deviation, across its 8 numbers (512 in the paper).

Row "the", by hand:

```
mean = (−0.14 + 1.28 − 0.11 − 0.87 + 0.14 + 2.14 + 0.44 − 0.29) / 8 = 0.32
std  = 0.90
first entry: (−0.14 − 0.32) / 0.90 = −0.51
```

```
X1 = LayerNorm(X + MHA)  (6×8)       ← each row now has mean 0, std 1
the [-0.51  1.06 -0.48 -1.33 -0.21  2.02  0.13 -0.68]
cat [ 0.41  0.20  0.30 -1.24 -0.77  1.73  0.78 -1.41]
sat [ 0.08  0.03  0.25 -1.56  0.02  2.08  0.16 -1.07]
on  [-0.54 -1.06  0.66 -0.78  0.22  1.81  0.90 -1.19]
the [-1.23 -0.28  0.18 -0.76  0.00  2.31  0.36 -0.57]
mat [-0.99  0.38  0.40 -0.92 -0.21  2.01  0.54 -1.22]
```

**Why normalize?** It keeps every token's numbers on the same scale as they pass through layer after layer, which keeps training stable. In a real model, γ and β (learned, size d_model) then rescale and shift each dimension.

---

# Sublayer 2: Position-wise feed-forward network

**What it's for.** Attention mixed information *between* tokens. The FFN now processes *each token on its own*, transforming what it just gathered. It's the same small two-layer network applied to every row (§3.3, Eq. 2):

**FFN(x) = max(0, x·W1 + b1)·W2 + b2**

## Step 9: Expand, from 8 to 32 dimensions

W1 is **(d_model × d_ff) = (8 × 32)**, and (512 × 2048) in the paper.

**(6 × 8) · (8 × 32) = (6 × 32)**

Row "the" of X1·W1 (first 12 of 32 values):

```
[-0.30  0.79  0.04 -0.78 -0.27  1.19  0.08 -0.23  1.35 -0.85 -1.28  0.67 ...]
```

## Step 10: ReLU

Replace every negative number with 0:

```
[ 0.00  0.79  0.04  0.00  0.00  1.19  0.08  0.00  1.35  0.00  0.00  0.67 ...]
```

This is the non-linearity. Without it, two matrix multiplications in a row would collapse into one, and the network couldn't learn anything beyond a linear map.

## Step 11: Compress, from 32 back to 8

W2 is **(d_ff × d_model) = (32 × 8)**.

**(6 × 32) · (32 × 8) = (6 × 8)**

```
F = H·W2  (6×8)
the [-1.21 -1.33 -1.46 -1.68  0.32 -1.54  0.81 -0.59]
cat [-0.28 -0.86 -1.46 -2.09  0.33 -1.09  1.31 -0.53]
sat [-0.96 -0.98 -1.50 -2.24 -0.12 -1.19  0.86 -0.88]
on  [-0.03 -0.46 -1.09 -1.18  0.25 -0.18  2.31 -1.24]
the [-1.16 -0.82 -1.64 -2.18 -0.20 -0.47  2.03 -1.61]
mat [-0.92 -0.79 -1.80 -1.98 -0.33 -0.93  1.91 -1.74]
```

Row "cat" of F depends **only** on row "cat" of X1. No information crosses between tokens here.

## Step 12: Add the residual, then LayerNorm again

```
X1 + F  (6×8)
the [-1.73 -0.26 -1.94 -3.01  0.11  0.48  0.95 -1.27]
cat [ 0.13 -0.66 -1.17 -3.33 -0.44  0.64  2.10 -1.93]
sat [-0.88 -0.96 -1.25 -3.80 -0.10  0.89  1.01 -1.94]
on  [-0.57 -1.52 -0.43 -1.96  0.46  1.63  3.20 -2.43]
the [-2.39 -1.11 -1.47 -2.94 -0.20  1.85  2.38 -2.18]
mat [-1.91 -0.41 -1.40 -2.90 -0.54  1.08  2.45 -2.96]

X_out = LayerNorm(X1 + F)  (6×8)   ← output of encoder layer 1
the [-0.70  0.45 -0.87 -1.70  0.74  1.03  1.39 -0.34]
cat [ 0.46 -0.05 -0.38 -1.79  0.09  0.80  1.74 -0.88]
sat [ 0.00 -0.05 -0.25 -2.00  0.53  1.21  1.29 -0.73]
on  [-0.21 -0.74 -0.13 -0.99  0.37  1.03  1.91 -1.25]
the [-0.89 -0.19 -0.39 -1.19  0.30  1.42  1.71 -0.78]
mat [-0.62  0.23 -0.32 -1.17  0.16  1.08  1.85 -1.21]
```

That's one full encoder layer: **(6 × 8) in, (6 × 8) out.** In the paper it's (6 × 512) in and (6 × 512) out.

---

## Then repeat 5 more times

X_out becomes the input to layer 2, which has the same steps but **its own** W_Q, W_K, W_V, W_O, W1, W2, γ and β. After layer 6 you have the encoder output, still (6 × 512), with one context-aware vector per token, ready for the decoder.

## Summary of shapes (paper sizes, n = 6)

| Step | Operation | Shape |
|---|---|---|
| Input | X | 6 × 512 |
| Per head (× 8) | Q, K, V = X · W (512 × 64) | 6 × 64 each |
| | Scores Q·Kᵀ, ÷ 8, softmax by row | 6 × 6 |
| | Weights · V | 6 × 64 |
| Combine | Concat 8 heads | 6 × 512 |
| | · W_O (512 × 512) | 6 × 512 |
| Add & Norm | LayerNorm(X + MHA) | 6 × 512 |
| FFN | · W1 (512 × 2048), ReLU | 6 × 2048 |
| | · W2 (2048 × 512) | 6 × 512 |
| Add & Norm | LayerNorm(X1 + F) | 6 × 512 |
| × 6 layers | | 6 × 512 |

## What each piece contributes

| Piece | Job |
|---|---|
| Q, K, V projections | Give each token a "what I want / what I offer / what I'll pass along" view of itself |
| Q·Kᵀ / √d_k, softmax | Decide how much each token should listen to every other token |
| Weights · V | Actually pull in that information |
| Multiple heads | Look for several kinds of relationship at once |
| W_O | Merge the heads' findings into one vector |
| Residual + LayerNorm | Keep each token's identity, keep the numbers stable, make deep stacking trainable |
| FFN | Per-token processing of what was gathered, which is most of the layer's parameters |
