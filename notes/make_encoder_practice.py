"""Builds encoder-practice.ipynb: a fill-in-every-matrix walk through one encoder layer.

Usage: python3 make_nb.py OUT.ipynb
Each "fill" cell holds nan placeholders; each "check" cell grades it against values
computed from the learner's own (already checked) earlier matrices, so rounding doesn't snowball.
"""
import json, sys

HELPERS = r'''# Run this cell first. It sets up the example, the weights and the checker.
# You don't need to read it, and please don't peek at REF: that's the answer key.
import numpy as np
np.set_printoptions(precision=4, suppress=True)
nan = np.nan

# ---- settings for this run (paper values in comments) ----
sentence = "the cat sat"
n       = 3    # tokens            (any length)
d_model = 4    # vector width      (paper: 512)
h       = 2    # heads             (paper: 8)
d_k     = d_model // h   # = 2     (paper: 64)
d_ff    = 8    # FFN inner width   (paper: 2048)

# ---- a tiny vocabulary and its (made-up) learned embedding table, V x d_model ----
vocab = {"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5}
W_emb = np.array([
    [ 0.0,  0.0,  0.0,  0.0],   # 0 <pad>
    [ 0.1,  0.2, -0.1,  0.3],   # 1 the
    [ 0.4, -0.2,  0.3,  0.1],   # 2 cat
    [-0.3,  0.5,  0.2, -0.1],   # 3 sat
    [ 0.2,  0.1, -0.4,  0.2],   # 4 on
    [ 0.3, -0.1,  0.1,  0.4],   # 5 mat
])

# ---- attention weights (made up; a trained model learns these), each d_model x d_k ----
W_Q1 = np.array([[ 1.0, 0.0], [0.0,  1.0], [ 0.5, 0.0], [0.0, -0.5]])
W_K1 = np.array([[ 0.0, 1.0], [1.0,  0.0], [ 0.0, 0.5], [-0.5, 0.0]])
W_V1 = np.array([[ 1.0, 0.0], [0.0,  0.0], [ 0.0, 1.0], [ 0.5, 0.5]])
W_Q2 = np.array([[ 0.0, 0.5], [1.0,  0.0], [ 0.0, 1.0], [ 0.5, 0.0]])
W_K2 = np.array([[ 0.5, 0.0], [0.0,  1.0], [ 1.0, 0.0], [ 0.0, 0.5]])
W_V2 = np.array([[ 0.0, 1.0], [0.5,  0.0], [ 1.0, 0.0], [ 0.0, -0.5]])
W_O  = np.array([[ 0.5, 0.0, 0.0, 0.5],
                 [ 0.0, 0.5, 0.5, 0.0],
                 [ 0.0, 0.5, -0.5, 0.0],
                 [ 0.5, 0.0, 0.0, -0.5]])

# ---- feed-forward weights: W1 is d_model x d_ff, W2 is d_ff x d_model ----
W1 = np.array([[ 0.5, -0.5,  0.0,  1.0,  0.0, -1.0,  0.5,  0.0],
               [ 0.0,  0.5,  0.5,  0.0, -0.5,  0.0,  0.0,  1.0],
               [ 1.0,  0.0, -0.5,  0.0,  0.5,  0.5,  0.0, -0.5],
               [ 0.0,  0.0,  0.5, -0.5,  0.0,  0.0,  1.0,  0.0]])
b1 = np.array([0.0, 0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.1])
W2 = np.array([[ 0.5,  0.0,  0.0,  0.0],
               [ 0.0,  0.5,  0.0,  0.0],
               [ 0.0,  0.0,  0.5,  0.0],
               [ 0.0,  0.0,  0.0,  0.5],
               [-0.5,  0.0,  0.5,  0.0],
               [ 0.0, -0.5,  0.0,  0.5],
               [ 0.5,  0.5,  0.0,  0.0],
               [ 0.0,  0.0, -0.5, -0.5]])
b2 = np.array([0.1, 0.0, -0.1, 0.0])
eps = 1e-5   # LayerNorm's tiny safety constant; it won't change your 2-decimal answers

# =============== the checker (no need to read below) ===============
def _sm(S):
    e = np.exp(S - S.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
def _ln(R):
    mu = R.mean(axis=1); sd = np.sqrt(R.var(axis=1) + eps)
    return mu, sd, (R - mu[:, None]) / sd[:, None]

_ok = {}
def got(k): return _ok[k] if k in _ok else REF[k]

_RULES = {   # how each matrix is computed from the matrices before it
  "ids":  lambda: np.array([vocab[t] for t in sentence.split()], dtype=float),
  "E":    lambda: W_emb[got("ids").astype(int)],
  "Es":   lambda: got("E") * np.sqrt(d_model),
  "div":  lambda: np.array([10000 ** (2 * i / d_model) for i in range(d_model // 2)]),
  "PE":   lambda: np.array([[f(p / got("div")[c // 2]) for c, f in
                             zip(range(d_model), [np.sin, np.cos] * (d_model // 2))] for p in range(n)]),
  "X":    lambda: got("Es") + got("PE"),
  "Q1":   lambda: got("X") @ W_Q1, "K1": lambda: got("X") @ W_K1, "V1": lambda: got("X") @ W_V1,
  "Q2":   lambda: got("X") @ W_Q2, "K2": lambda: got("X") @ W_K2, "V2": lambda: got("X") @ W_V2,
  "S1":   lambda: got("Q1") @ got("K1").T, "S2": lambda: got("Q2") @ got("K2").T,
  "S1s":  lambda: got("S1") / np.sqrt(d_k), "S2s": lambda: got("S2") / np.sqrt(d_k),
  "A1":   lambda: _sm(got("S1s")), "A2": lambda: _sm(got("S2s")),
  "H1":   lambda: got("A1") @ got("V1"), "H2": lambda: got("A2") @ got("V2"),
  "C":    lambda: np.concatenate([got("H1"), got("H2")], axis=1),
  "MHA":  lambda: got("C") @ W_O,
  "R1":   lambda: got("X") + got("MHA"),
  "mu1":  lambda: _ln(got("R1"))[0], "sd1": lambda: _ln(got("R1"))[1],
  "X1":   lambda: (got("R1") - got("mu1")[:, None]) / got("sd1")[:, None],
  "P":    lambda: got("X1") @ W1 + b1,
  "Hf":   lambda: np.maximum(0, got("P")),
  "F":    lambda: got("Hf") @ W2 + b2,
  "R2":   lambda: got("X1") + got("F"),
  "mu2":  lambda: _ln(got("R2"))[0], "sd2": lambda: _ln(got("R2"))[1],
  "Z":    lambda: (got("R2") - got("mu2")[:, None]) / got("sd2")[:, None],
}
REF = {}
for _k, _f in _RULES.items():   # the full answer key, computed in order
    REF[_k] = _f()

_HINTS = {
  "ids": "Look each word up in the vocab dictionary.",
  "E": "Row r is the row of W_emb whose number is ids[r]. No multiplication, just copy the row.",
  "Es": "Multiply every entry of E by sqrt(d_model) = sqrt(4) = 2.",
  "div": "Pair i uses 10000 ** (2*i / d_model). With d_model = 4 that's 10000**0 and 10000**0.5.",
  "PE": "Even column 2i: sin(pos / div[i]). Odd column 2i+1: cos(pos / div[i]). Angles are in radians.",
  "X": "Add Es and PE entry by entry (same row, same column).",
  "S1s": "Divide every score by sqrt(d_k) = sqrt(2) ≈ 1.4142.",
  "S2s": "Divide every score by sqrt(d_k) = sqrt(2) ≈ 1.4142.",
  "A1": "For each row: take e**x of every entry, then divide each by the row's total. Each row must sum to 1.",
  "A2": "For each row: take e**x of every entry, then divide each by the row's total. Each row must sum to 1.",
  "C": "Put H1 and H2 side by side: the first d_k columns are H1, the next d_k are H2.",
  "R1": "Residual: add the layer's input X to MHA, entry by entry.",
  "mu1": "The mean of each row of R1 (sum the 4 numbers, divide by 4).",
  "sd1": "For each row: average the squared distances from the row's mean, then take the square root.",
  "X1": "(R1[r, c] - mu1[r]) / sd1[r]. Every row ends up with mean 0 and spread 1.",
  "P": "X1 @ W1, then add b1[c] to every entry in column c.",
  "Hf": "ReLU: keep positive numbers, replace negatives with 0.",
  "F": "Hf @ W2, then add b2[c] to every entry in column c.",
  "R2": "Residual again: X1 + F, entry by entry.",
  "mu2": "The mean of each row of R2.",
  "sd2": "The standard deviation of each row of R2 (same recipe as sd1).",
  "Z": "(R2[r, c] - mu2[r]) / sd2[r].",
}
_MATMUL = {"Q1": ("X", W_Q1, "W_Q1"), "K1": ("X", W_K1, "W_K1"), "V1": ("X", W_V1, "W_V1"),
           "Q2": ("X", W_Q2, "W_Q2"), "K2": ("X", W_K2, "W_K2"), "V2": ("X", W_V2, "W_V2"),
           "S1": ("Q1", None, "K1ᵀ"), "S2": ("Q2", None, "K2ᵀ"),
           "H1": ("A1", None, "V1"), "H2": ("A2", None, "V2"), "MHA": ("C", W_O, "W_O")}

def _hint(name, r, c):
    if name in _MATMUL:
        a, B, bname = _MATMUL[name]
        A = got(a)
        if B is None:
            B = {"S1": lambda: got("K1").T, "S2": lambda: got("K2").T,
                 "H1": lambda: got("V1"), "H2": lambda: got("V2")}[name]()
        terms = " + ".join(f"({A[r, j]:.2f})×({B[j, c]:.2f})" for j in range(A.shape[1]))
        return f"{name}[{r},{c}] = row {r} of {a} · column {c} of {bname} = {terms}"
    return _HINTS.get(name, "")

def check(name, yours):
    """Grade your matrix. Wrong or blank cells are listed with a hint for the first one."""
    exp = _RULES[name]()
    y = np.asarray(yours, dtype=float)
    if y.shape != exp.shape:
        print(f"✗ {name}: shape {y.shape}, but it should be {exp.shape}."); return
    if np.isnan(y).all():
        print(f"… {name} is still blank. Replace every nan with your number."); return
    tol = 0.02 + 0.01 * np.abs(exp)
    bad = np.argwhere(np.isnan(y) | (np.abs(y - exp) > tol))
    if len(bad) == 0:
        _ok[name] = y
        print(f"✓ {name} {exp.shape} is correct. Nice work!"); return
    print(f"✗ {name}: {len(bad)} of {exp.size} entries need another look:")
    for idx in bad[:6]:
        where = f"[{', '.join(map(str, idx))}]"
        print(f"   {name}{where}: you have {y[tuple(idx)]:.4g}")
    idx = bad[0]
    r, c = (int(idx[0]), int(idx[1])) if y.ndim == 2 else (int(idx[0]), 0)
    print("   Hint:", _hint(name, r, c))
    print(f"   Stuck? reveal('{name}') shows the answer and lets you carry on.")

def reveal(name):
    """Show the answer for one matrix and use it from here on."""
    exp = _RULES[name](); _ok[name] = exp
    print(f"{name} {exp.shape} =\n{np.round(exp, 4)}")

def check_shapes(answers):
    """Grade a dictionary of shapes for the paper's sizes."""
    PAPER = {"E": (6, 512), "W_Q1": (512, 64), "Q1": (6, 64), "S1": (6, 6), "A1": (6, 6),
             "H1": (6, 64), "C": (6, 512), "W_O": (512, 512), "P": (6, 2048), "Z": (6, 512)}
    right = 0
    for k, v in PAPER.items():
        mine = tuple(answers.get(k, ()))
        ok = mine == v; right += ok
        print(f"{'✓' if ok else '✗'} {k:5s} you said {mine or '(blank)'}" + ("" if ok else f", the paper has {v}"))
    print(f"\n{right} of {len(PAPER)} right.")

print("Ready. Sentence:", sentence.split(), "| n =", n, "| d_model =", d_model, "| h =", h, "| d_k =", d_k, "| d_ff =", d_ff)
'''

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
def code(s, tag=None):
    c = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}
    if tag: c["metadata"]["tags"] = [tag]
    return c

def blank(name, rows, cols=None, comment=""):
    if cols is None:
        body = f"{name} = np.array([" + ", ".join(["nan"] * rows) + "])"
    else:
        row = "[" + ", ".join(["nan"] * cols) + "]"
        body = f"{name} = np.array([\n" + ",\n".join("    " + row for _ in range(rows)) + "\n])"
    return (f"# {comment}\n" if comment else "") + body

def fill(name, rows, cols=None, comment=""):
    return code(blank(name, rows, cols, comment) + f"\ncheck('{name}', {name})", tag=f"fill:{name}")

cells = [
md("""
# The encoder by hand: final homework

This is the last exercise for the encoder part of **Attention by Hand**. You'll push one short sentence through a whole encoder layer and **write down every matrix yourself**, from the token IDs to the encoder output Z.

**How it works**

1. Run the first code cell. It sets up the sentence, a tiny vocabulary and all the weights.
2. Each step shows a code cell full of `nan`. Replace every `nan` with your number, then run the cell. It checks your answer straight away.
3. If something is off, the checker says which entries and gives a hint for the first one. Still stuck? Call `reveal('name')` to see the answer and carry on.

Work on paper or use Python as a calculator for single numbers (`np.sin(1)`, `np.exp(0.5)`, `0.3*2 + 1.1*0.5`). Try not to use `@` to multiply whole matrices: doing the dot products yourself is the point. Two decimals is enough. Each check uses *your* previous checked matrices, so small rounding differences won't snowball.

**The sizes we use** (the same idea as the site, just small enough to do by hand)

| setting | here | paper |
|---|---|---|
| sentence length n | 3 | any |
| d_model | 4 | 512 |
| heads h | 2 | 8 |
| d_k = d_model / h | 2 | 64 |
| d_ff | 8 | 2048 |
| layers N | 1 | 6 |

Dropout is skipped, as it would be when the model is used after training.
"""),
code(HELPERS, tag="helpers"),

md("""
## Step 1: Tokenize

The sentence is `"the cat sat"`. Look each word up in `vocab` to get its ID.

**Shape:** (n,) = (3,)
"""),
fill("ids", 3, comment="one ID per token"),

md("""
## Step 2: Embedding lookup

`W_emb` has one row per vocabulary entry, so it's (6 × 4) here and (≈37,000 × 512) in the paper. The lookup copies out the row for each ID.

**Shape:** (n × d_model) = (3 × 4)
"""),
code("print(W_emb)"),
fill("E", 3, 4, "row r = W_emb[ids[r]]"),

md("""
## Step 3: Scale by √d_model

The paper multiplies the embeddings by √d_model (§3.4). Here that's √4 = 2. In the paper it's √512 ≈ 22.63.

**Shape:** (3 × 4), unchanged
"""),
fill("Es", 3, 4, "every entry of E times 2"),

md("""
## Step 4: Positional encoding

First the divisor for each pair, `10000 ** (2i / d_model)`. With d_model = 4 there are 2 pairs.
"""),
fill("div", 2, comment="div[0] for pair 0, div[1] for pair 1"),
md("""
Now the PE matrix. For position `pos` and pair `i`:

- column 2i: `sin(pos / div[i])`
- column 2i+1: `cos(pos / div[i])`

Positions are 0, 1, 2. Use radians, and give 4 decimals for this one.

**Shape:** (n × d_model) = (3 × 4)
"""),
fill("PE", 3, 4, "row = position 0, 1, 2; columns: sin, cos (pair 0), sin, cos (pair 1)"),

md("""
## Step 5: Add them: X = Es + PE

This X is what enters the encoder layer.

**Shape:** (3 × 4)
"""),
fill("X", 3, 4, "entry by entry"),

md("""
## Step 6: Queries, keys and values for each head

Each head has its own W_Q, W_K and W_V, each (d_model × d_k) = (4 × 2). Multiply X by each one:

(3 × 4) · (4 × 2) = (3 × 2)

Entry [r, c] is row r of X dotted with column c of the weight matrix.
"""),
code("print('W_Q1 =\\n', W_Q1, '\\nW_K1 =\\n', W_K1, '\\nW_V1 =\\n', W_V1)"),
md("### Head 1"),
fill("Q1", 3, 2, "X · W_Q1"),
fill("K1", 3, 2, "X · W_K1"),
fill("V1", 3, 2, "X · W_V1"),
md("### Head 2"),
code("print('W_Q2 =\\n', W_Q2, '\\nW_K2 =\\n', W_K2, '\\nW_V2 =\\n', W_V2)"),
fill("Q2", 3, 2, "X · W_Q2"),
fill("K2", 3, 2, "X · W_K2"),
fill("V2", 3, 2, "X · W_V2"),

md("""
## Step 7: Attention scores S = Q · Kᵀ

Every token's query against every token's key: (3 × 2) · (2 × 3) = (3 × 3). Entry [r, c] says how well token r's query matches token c's key.
"""),
fill("S1", 3, 3, "head 1: S1[r, c] = Q1[r] · K1[c]"),
fill("S2", 3, 3, "head 2"),

md("""
## Step 8: Scale by √d_k

Divide every score by √d_k = √2 ≈ 1.4142 (in the paper, √64 = 8). This keeps the softmax from getting too peaky.
"""),
fill("S1s", 3, 3, "S1 / sqrt(2)"),
fill("S2s", 3, 3, "S2 / sqrt(2)"),

md("""
## Step 9: Softmax, row by row

For each row: `e**x` for every entry, then divide by the row's total. Every row must add up to 1. Row r is how token r spreads its attention over the 3 tokens.

**Shape:** (3 × 3)
"""),
fill("A1", 3, 3, "head 1 attention weights"),
fill("A2", 3, 3, "head 2 attention weights"),

md("""
## Step 10: Weighted sum of values, head = A · V

(3 × 3) · (3 × 2) = (3 × 2). Each output row is a blend of the value rows, mixed using that token's attention weights.
"""),
fill("H1", 3, 2, "A1 · V1"),
fill("H2", 3, 2, "A2 · V2"),

md("""
## Step 11: Concatenate the heads and mix with W_O

Put the heads side by side: h × (3 × 2) gives (3 × 4). Then multiply by W_O (4 × 4).
"""),
fill("C", 3, 4, "[H1 | H2]"),
code("print(W_O)"),
fill("MHA", 3, 4, "C · W_O"),

md("""
## Step 12: Add & Norm (1)

**Residual:** add the input back in, `R1 = X + MHA`.

**LayerNorm:** for each row on its own, subtract the row's mean and divide by its standard deviation. Here γ = 1 and β = 0, so there's no extra scale or shift.
"""),
fill("R1", 3, 4, "X + MHA"),
fill("mu1", 3, comment="mean of each row of R1"),
fill("sd1", 3, comment="standard deviation of each row of R1 (divide by 4, not 3)"),
fill("X1", 3, 4, "(R1[r, c] - mu1[r]) / sd1[r]"),

md("""
## Step 13: Feed-forward network

Expand with W1 (4 × 8) plus b1, apply ReLU, then compress with W2 (8 × 4) plus b2. Each token goes through on its own, so there's no mixing between rows here.

(3 × 4) · (4 × 8) = (3 × 8), then (3 × 8) · (8 × 4) = (3 × 4)
"""),
code("print('W1 =\\n', W1, '\\nb1 =', b1)"),
fill("P", 3, 8, "X1 · W1 + b1"),
fill("Hf", 3, 8, "ReLU(P)"),
code("print('W2 =\\n', W2, '\\nb2 =', b2)"),
fill("F", 3, 4, "Hf · W2 + b2"),

md("""
## Step 14: Add & Norm (2), the encoder output

Same recipe as step 12, with X1 as the residual.
"""),
fill("R2", 3, 4, "X1 + F"),
fill("mu2", 3, comment="mean of each row of R2"),
fill("sd2", 3, comment="standard deviation of each row of R2"),
fill("Z", 3, 4, "(R2[r, c] - mu2[r]) / sd2[r]"),

md("""
## You did it

Z is the output of one encoder layer: one 4-number vector per token, now mixed with context from the whole sentence. In the paper you'd feed Z into layer 2 and repeat until layer 6, each layer with its own weights.

Run the cell below to see your whole forward pass, step by step.
"""),
code("""
for k in ["ids", "E", "Es", "PE", "X", "Q1", "K1", "V1", "S1", "A1", "H1", "C", "MHA", "X1", "P", "F", "Z"]:
    print(f"{k:4s} {str(got(k).shape):8s} {'✓ yours' if k in _ok else '· answer key'}")
print(f"\\n{len(_ok)} of {len(_RULES)} matrices done by you.")
"""),

md("""
## Bonus: the paper's sizes, by hand

No numbers this time, just shapes. Imagine a 6-token sentence and the paper's sizes (d_model 512, h 8, d_k 64, d_ff 2048). Fill in each shape as a tuple, like `(6, 512)`.
"""),
code("""
paper_shapes = {
    "E":    (),   # embeddings after the lookup
    "W_Q1": (),   # one head's query weights
    "Q1":   (),   # one head's queries
    "S1":   (),   # one head's scores
    "A1":   (),   # one head's attention weights
    "H1":   (),   # one head's output
    "C":    (),   # all heads concatenated
    "W_O":  (),   # output projection
    "P":    (),   # FFN hidden layer
    "Z":    (),   # encoder output
}
check_shapes(paper_shapes)
""", tag="fill:paper_shapes"),

md("""
## Want more?

- **Change the settings.** In the first cell, try a different sentence from the vocabulary, such as `"the cat sat on the mat"`, and set `n` to match. Then redo the pass. Both "the" rows start the same and only positional encoding tells them apart.
- **Make the weights your own.** Edit any weight matrix and rerun. The checker recomputes the answer key for you.
- **Compare with the site.** Every step here matches a step in Attention by Hand, which also animates each multiplication.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("cells:", len(cells))
