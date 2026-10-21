# Positional encoding: every calculation

Sentence: **"the cat sat on the mat"**, which is 6 tokens at positions 0 to 5.

The formula from §3.5 of the paper:

```
PE[pos, 2i]   = sin( pos / 10000^(2i / d_model) )
PE[pos, 2i+1] = cos( pos / 10000^(2i / d_model) )
```

- `pos` is the token's position: 0, 1, 2, …
- `i` is the pair number: 0, 1, 2, …, d_model/2 − 1
- Dimension `2i` (even) gets the sine and dimension `2i+1` (odd) gets the cosine, **of the same angle**.

The word itself never appears in the formula.

I do the whole calculation in an 8-dimension version first so every number fits on screen, and then show how it scales to the paper's 512. All numbers were computed in Python and rounded to 4 decimals.

---

## Part 1: The full calculation with d_model = 8

### Step 1: Work out each pair's divisor

There are 8 / 2 = 4 pairs. For each pair, compute `10000^(2i/8)`:

| Pair i | Dimensions | Exponent 2i/8 | Divisor 10000^(2i/8) | So the angle is |
|---|---|---|---|---|
| 0 | 0 (sin), 1 (cos) | 0/8 = 0 | 10000^0 = **1** | pos ÷ 1 |
| 1 | 2 (sin), 3 (cos) | 2/8 = 0.25 | 10000^0.25 = **10** | pos ÷ 10 |
| 2 | 4 (sin), 5 (cos) | 4/8 = 0.5 | 10000^0.5 = **100** | pos ÷ 100 |
| 3 | 6 (sin), 7 (cos) | 6/8 = 0.75 | 10000^0.75 = **1000** | pos ÷ 1000 |

These divisors depend only on d_model, so they're the same for every sentence.

### Step 2: Work out the angle for each position and pair

angle = pos ÷ divisor. All angles are in radians.

| pos | token | pair 0 (÷1) | pair 1 (÷10) | pair 2 (÷100) | pair 3 (÷1000) |
|---|---|---|---|---|---|
| 0 | the | 0 | 0.0 | 0.00 | 0.000 |
| 1 | cat | 1 | 0.1 | 0.01 | 0.001 |
| 2 | sat | 2 | 0.2 | 0.02 | 0.002 |
| 3 | on | 3 | 0.3 | 0.03 | 0.003 |
| 4 | the | 4 | 0.4 | 0.04 | 0.004 |
| 5 | mat | 5 | 0.5 | 0.05 | 0.005 |

### Step 3: Take the sine and cosine of each angle

Here is every entry written out for one position, **pos = 3 ("on")**:

| Dim | Pair | Function | Angle | Value |
|---|---|---|---|---|
| 0 | 0 | sin | 3 | sin(3) = **0.1411** |
| 1 | 0 | cos | 3 | cos(3) = **−0.9900** |
| 2 | 1 | sin | 0.3 | sin(0.3) = **0.2955** |
| 3 | 1 | cos | 0.3 | cos(0.3) = **0.9553** |
| 4 | 2 | sin | 0.03 | sin(0.03) = **0.0300** |
| 5 | 2 | cos | 0.03 | cos(0.03) = **0.9996** |
| 6 | 3 | sin | 0.003 | sin(0.003) = **0.0030** |
| 7 | 3 | cos | 0.003 | cos(0.003) = **1.0000** |

Doing the same for every position gives the full PE matrix, **(6 × 8)**:

| pos | token | d0 sin | d1 cos | d2 sin | d3 cos | d4 sin | d5 cos | d6 sin | d7 cos |
|---|---|---|---|---|---|---|---|---|---|
| 0 | the | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | cat | 0.8415 | 0.5403 | 0.0998 | 0.9950 | 0.0100 | 1.0000 | 0.0010 | 1.0000 |
| 2 | sat | 0.9093 | −0.4161 | 0.1987 | 0.9801 | 0.0200 | 0.9998 | 0.0020 | 1.0000 |
| 3 | on | 0.1411 | −0.9900 | 0.2955 | 0.9553 | 0.0300 | 0.9996 | 0.0030 | 1.0000 |
| 4 | the | −0.7568 | −0.6536 | 0.3894 | 0.9211 | 0.0400 | 0.9992 | 0.0040 | 1.0000 |
| 5 | mat | −0.9589 | 0.2837 | 0.4794 | 0.8776 | 0.0500 | 0.9988 | 0.0050 | 1.0000 |

What to notice:

- **Pair 0 (d0, d1)** swings around a lot from one word to the next. It's the "second hand", and it tells neighbouring words apart.
- **Pair 1 (d2, d3)** moves steadily but slowly.
- **Pairs 2 and 3** have barely moved over 6 words. They only matter for telling apart positions that are hundreds or thousands of tokens apart.
- The two "the" rows (pos 0 and pos 4) are completely different. That's the whole point.

### Step 4: Add PE to the embeddings

The embeddings below are **made up** for illustration (real ones are learned). Both "the" rows are identical, because it's the same word.

```
Embeddings E (6 × 8), made up
the: [ 0.10,  0.20, -0.30,  0.40,  0.00,  0.10, -0.20,  0.30]
cat: [ 0.30, -0.10,  0.50,  0.20, -0.40,  0.10,  0.60, -0.20]
sat: [-0.20,  0.40,  0.10, -0.30,  0.20,  0.50, -0.10,  0.00]
on:  [ 0.00, -0.30,  0.20,  0.10,  0.30, -0.20,  0.40,  0.10]
the: [ 0.10,  0.20, -0.30,  0.40,  0.00,  0.10, -0.20,  0.30]
mat: [ 0.40,  0.00, -0.10,  0.30, -0.20,  0.20,  0.10, -0.40]
```

X = E + PE, adding element by element (same row, same column):

```
X (6 × 8)
pos0 the: [ 0.10,  1.20, -0.30,  1.40,  0.00,  1.10, -0.20,  1.30]
pos1 cat: [ 1.14,  0.44,  0.60,  1.20, -0.39,  1.10,  0.60,  0.80]
pos2 sat: [ 0.71, -0.02,  0.30,  0.68,  0.22,  1.50, -0.10,  1.00]
pos3 on:  [ 0.14, -1.29,  0.50,  1.06,  0.33,  0.80,  0.40,  1.10]
pos4 the: [-0.66, -0.45,  0.09,  1.32,  0.04,  1.10, -0.20,  1.30]
pos5 mat: [-0.56,  0.28,  0.38,  1.18, -0.15,  1.20,  0.10,  0.60]
```

Worked example, pos 1 dim 0: 0.30 (cat) + 0.8415 (PE) = 1.14.

The two "the" rows started identical in E but are now different in X. X is what goes into encoder layer 1, where it gets projected into Q, K and V.

---

## Part 2: The same thing at the paper's size, d_model = 512

Nothing changes except that there are now **256 pairs** instead of 4, so the divisors are spaced more finely.

### Divisors for some of the 256 pairs

Divisor = 10000^(2i/512). The **wavelength** is how many positions one full sine cycle takes, which works out to 2π × divisor.

| Pair i | Dims | Divisor | Wavelength (positions per cycle) |
|---|---|---|---|
| 0 | 0, 1 | 1.0000 | 6.3 |
| 1 | 2, 3 | 1.0366 | 6.5 |
| 2 | 4, 5 | 1.0746 | 6.8 |
| … | … | … | … |
| 64 | 128, 129 | 10.0000 | 62.8 |
| 128 | 256, 257 | 100.0000 | 628.3 |
| 192 | 384, 385 | 1000.0000 | 6,283.2 |
| 255 | 510, 511 | 9646.6 | 60,611.5 |

Each pair's divisor is about 3.66% larger than the one before it (10000^(2/512) ≈ 1.0366). This is the "geometric progression from 2π to 10000·2π" that §3.5 describes.

### PE values for our sentence, some pairs (sin, cos)

| pos | token | pair 0 (dims 0, 1) | pair 1 (dims 2, 3) | pair 64 (dims 128, 129) | pair 255 (dims 510, 511) |
|---|---|---|---|---|---|
| 0 | the | 0.0000, 1.0000 | 0.0000, 1.0000 | 0.0000, 1.0000 | 0.0000, 1.0000 |
| 1 | cat | 0.8415, 0.5403 | 0.8219, 0.5697 | 0.0998, 0.9950 | 0.0001, 1.0000 |
| 2 | sat | 0.9093, −0.4161 | 0.9364, −0.3509 | 0.1987, 0.9801 | 0.0002, 1.0000 |
| 3 | on | 0.1411, −0.9900 | 0.2451, −0.9695 | 0.2955, 0.9553 | 0.0003, 1.0000 |
| 4 | the | −0.7568, −0.6536 | −0.6572, −0.7537 | 0.3894, 0.9211 | 0.0004, 1.0000 |
| 5 | mat | −0.9589, 0.2837 | −0.9939, 0.1107 | 0.4794, 0.8776 | 0.0005, 1.0000 |

Pair 0 matches the 8-dim version exactly, because its divisor is 1 in both. Pair 64 has divisor 10, so it matches pair 1 of the 8-dim version. The 512-dim encoding is just the same idea with many more speeds packed in between.

The full **6 × 512** matrix, all 3,072 numbers, is in [positional-encoding-6x512.csv](positional-encoding-6x512.csv). Each row is one position, and the columns are dim0 to dim511.

### What happens next at full size

```
E   = embedding lookup × √512          (6 × 512)
PE  = the matrix above                 (6 × 512)
X0  = E + PE                           (6 × 512)   element-wise addition
X0  = dropout(X0, p = 0.1)             (§5.4)
```

X0 goes into encoder layer 1.

---

## One more check: why a shift is a rotation

Take pair 0 (divisor 1) and move from pos 1 to pos 3, a shift of k = 2:

```
sin(3) = sin(1)·cos(2) + cos(1)·sin(2) = 0.8415·(−0.4161) + 0.5403·0.9093 = −0.3501 + 0.4913 = 0.1411  ✓
cos(3) = cos(1)·cos(2) − sin(1)·sin(2) = 0.5403·(−0.4161) − 0.8415·0.9093 = −0.2248 − 0.7652 = −0.9900  ✓
```

So the new pair (sin 3, cos 3) is just the old pair (sin 1, cos 1) multiplied by a fixed 2×2 matrix that depends only on k = 2, not on where you started. That's why the model can learn "look 2 words back" with a linear projection (§3.5).
