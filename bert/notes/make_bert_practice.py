"""Builds the two BERT practice notebooks: fill in every matrix, from token IDs to the loss.

Usage: python3 make_bert_practice.py enc  OUT.ipynb   # encoder only: ids -> T
       python3 make_bert_practice.py full OUT.ipynb   # encoder + MLM and NSP heads + the summed loss
Each "fill" cell holds nan placeholders; each check grades it against values computed from the
learner's own (already checked) earlier matrices, so rounding doesn't snowball.
The made-up weights come from seed 19371, picked so the tiny model's fill-mask guess comes out right
and every check passes when each answer is rounded to 2 decimals.
"""
import json, sys
import numpy as np

PART, OUT = sys.argv[1], sys.argv[2]
FULL = PART == "full"

# ---- the example: "[CLS] [MASK] [SEP] barks [SEP]", the hidden word is "dog", the label is IsNext
n, d, h, dk, dff, V = 5, 4, 2, 2, 8, 8
r = np.random.default_rng(19371)
w = lambda *s: r.choice([-1, -0.5, 0, 0, 0.5, 1], size=s)
e = lambda *s: np.round(r.choice(np.arange(-5, 6)) * 0 + r.integers(-5, 6, size=s) / 10, 1)
W_tok = e(V, d); W_tok[0] = 0; W_seg = e(2, d); W_pos = e(n, d)
WQ = [w(d, dk) for _ in range(h)]; WK = [w(d, dk) for _ in range(h)]; WV = [w(d, dk) for _ in range(h)]
W_O = w(d, d); W1 = w(d, dff); b1 = r.choice([-0.1, 0, 0, 0.1], dff); W2 = w(dff, d); b2 = r.choice([-0.1, 0, 0, 0.1], d)
W_t = w(d, d); b_t = r.choice([-0.1, 0, 0.1], d)
W_p = w(d, d); b_p = r.choice([-0.1, 0, 0.1], d); W_nsp = w(2, d)

def lit(name, A, note=""):
    A = np.asarray(A, dtype=float) + 0.0
    if A.ndim == 1:
        return f"{name} = np.array([{', '.join(f'{v:4.1f}' for v in A)}]){'   # ' + note if note else ''}"
    rows = [f"[{', '.join(f'{v:4.1f}' for v in row)}]" for row in A]
    pad = " " * (len(name) + 13)
    return f"{name} = np.array([{rows[0]},\n" + "".join(f"{pad}{row},\n" for row in rows[1:-1]) + f"{pad}{rows[-1]}]){'   # ' + note if note else ''}"

TOKS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "dog", "cat", "barks", "sleeps"]
tok_table = "\n".join(f"    [{', '.join(f'{v:4.1f}' for v in row + 0.0)}],   # {i} {t}" for i, (row, t) in enumerate(zip(W_tok, TOKS)))
pos_table = "\n".join(f"    [{', '.join(f'{v:4.1f}' for v in row + 0.0)}],   # position {i}" for i, row in enumerate(W_pos))

HEAD_WEIGHTS = f'''
# ---- the two heads (Part 2). MLM: transform W_t, b_t, then the token table again (tied), bias 0.
{lit("W_t", W_t)}
{lit("b_t", b_t)}
b_out = np.zeros(V)   # the released code learns a bias per vocabulary piece; 0 here
# ---- NSP: pooler W_p, b_p (dense + tanh on C), then a 2-way classifier, one row per class.
{lit("W_p", W_p)}
{lit("b_p", b_p)}
{lit("W_nsp", W_nsp, "row 0: IsNext, row 1: NotNext")}
b_nsp = np.zeros(2)
''' if FULL else ""

HEAD_RULES = '''
  "t":        lambda: got("T")[mask_pos],
  "pre_t":    lambda: got("t") @ W_t + b_t,
  "g":        lambda: _gelu(got("pre_t")),
  "mu_t":     lambda: _ln(got("g")[None])[0][0], "sd_t": lambda: _ln(got("g")[None])[1][0],
  "u":        lambda: (got("g") - got("mu_t")) / got("sd_t"),
  "logits":   lambda: got("u") @ W_tok.T + b_out,
  "P_mlm":    lambda: _sm(got("logits")[None])[0],
  "loss_mlm": lambda: -np.log(got("P_mlm")[vocab[target]]),
  "pred_id":  lambda: np.array(float(np.argmax(got("P_mlm")))),
  "c":        lambda: got("T")[0],
  "pre_p":    lambda: got("c") @ W_p + b_p,
  "pooled":   lambda: np.tanh(got("pre_p")),
  "nsp_logits": lambda: got("pooled") @ W_nsp.T + b_nsp,
  "P_nsp":    lambda: _sm(got("nsp_logits")[None])[0],
  "loss_nsp": lambda: -np.log(got("P_nsp")[label]),
  "total":    lambda: got("loss_mlm") + got("loss_nsp"),
''' if FULL else ""

HEAD_HINTS = '''
  "t": "Copy row mask_pos (= 1) of T. Only the masked position goes into the MLM head.",
  "g": "GELU of each entry of pre_t, same formula as for I.",
  "mu_t": "The mean of the 4 numbers in g.", "sd_t": "The standard deviation of g (divide by 4).",
  "u": "(g[c] - mu_t) / sd_t.",
  "logits": "logits[v] = u · row v of W_tok (the input token table, reused) + b_out[v], which is 0.",
  "P_mlm": "Softmax over all 8 pieces: e**x of each logit, divided by their total.",
  "loss_mlm": "-ln of the probability of the true word, P_mlm[vocab['dog']] = P_mlm[4].",
  "pred_id": "The index of the biggest probability in P_mlm. That is the fill-mask answer.",
  "c": "Copy row 0 of T: the [CLS] vector.",
  "pooled": "tanh of each entry of pre_p (np.tanh is fine).",
  "nsp_logits": "nsp_logits[k] = pooled · row k of W_nsp + b_nsp[k], which is 0.",
  "P_nsp": "Softmax over the two numbers.",
  "loss_nsp": "-ln of the probability of the true label, IsNext = index 0.",
  "total": "Add the two losses. This one sum is what pre-training makes smaller.",
''' if FULL else ""

HELPERS = r'''# Run this cell first. It sets up the example, the weights and the checker.
# You don't need to read it, and please don't peek at REF: that's the answer key.
import numpy as np
np.set_printoptions(precision=4, suppress=True)
nan = np.nan

# ---- settings for this run (BERT-base values in comments) ----
sentence_A = "dog"      # the word "dog" is hidden behind [MASK]
sentence_B = "barks"
tokens   = ["[CLS]", "[MASK]", "[SEP]", "barks", "[SEP]"]
mask_pos = 1            # position of the hidden word
target   = "dog"        # what the MLM head must predict there
label    = 0            # 0 = IsNext (B really follows A), 1 = NotNext
n    = 5    # tokens              (BERT: up to 512)
H    = 4    # hidden size         (BERT-base: 768)
A    = 2    # attention heads     (BERT-base: 12)
d_k  = H // A   # = 2             (BERT-base: 64)
d_ff = 8    # FFN inner width     (BERT-base: 3072 = 4H)
V    = 8    # vocabulary size     (BERT: 30,000 WordPiece pieces)

# ---- a tiny WordPiece vocabulary and its (made-up) learned token table, V x H ----
vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "dog": 4, "cat": 5, "barks": 6, "sleeps": 7}
W_tok = np.array([
''' + tok_table + r'''
])
# ---- segment table (row 0: sentence A, row 1: sentence B) and LEARNED position table, n x H ----
''' + lit("W_seg", W_seg) + r'''
W_pos = np.array([
''' + pos_table + r'''
])
eps = 1e-12   # LayerNorm's tiny safety constant in the released code

# ---- attention weights (made up; a trained model learns these), each H x d_k ----
''' + "\n".join(lit(f"W_{m}{k+1}", M[k]) for k in range(h) for m, M in (("Q", WQ), ("K", WK), ("V", WV))) + r'''
''' + lit("W_O", W_O) + r'''

# ---- feed-forward weights: W1 is H x d_ff, W2 is d_ff x H ----
''' + lit("W1", W1) + "\n" + lit("b1", b1) + "\n" + lit("W2", W2) + "\n" + lit("b2", b2) + r'''
''' + HEAD_WEIGHTS + r'''
# =============== the checker (no need to read below) ===============
def _sm(S):
    e = np.exp(S - S.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
def _ln(R):
    mu = R.mean(axis=1); sd = np.sqrt(R.var(axis=1) + eps)
    return mu, sd, (R - mu[:, None]) / sd[:, None]
def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

_ok = {}
def got(k): return _ok[k] if k in _ok else REF[k]

_RULES = {   # how each matrix is computed from the matrices before it
  "ids":  lambda: np.array([vocab[t] for t in tokens], dtype=float),
  "seg":  lambda: np.array([0, 0, 0, 1, 1], dtype=float),
  "E_tok": lambda: W_tok[got("ids").astype(int)],
  "E_seg": lambda: W_seg[got("seg").astype(int)],
  "E_pos": lambda: W_pos[:n],
  "X_sum": lambda: got("E_tok") + got("E_seg") + got("E_pos"),
  "mu0":  lambda: _ln(got("X_sum"))[0], "sd0": lambda: _ln(got("X_sum"))[1],
  "X0":   lambda: (got("X_sum") - got("mu0")[:, None]) / got("sd0")[:, None],
  "Q1":   lambda: got("X0") @ W_Q1, "K1": lambda: got("X0") @ W_K1, "V1": lambda: got("X0") @ W_V1,
  "Q2":   lambda: got("X0") @ W_Q2, "K2": lambda: got("X0") @ W_K2, "V2": lambda: got("X0") @ W_V2,
  "S1":   lambda: got("Q1") @ got("K1").T, "S2": lambda: got("Q2") @ got("K2").T,
  "S1s":  lambda: got("S1") / np.sqrt(d_k), "S2s": lambda: got("S2") / np.sqrt(d_k),
  "A1":   lambda: _sm(got("S1s")), "A2": lambda: _sm(got("S2s")),
  "H1":   lambda: got("A1") @ got("V1"), "H2": lambda: got("A2") @ got("V2"),
  "C":    lambda: np.concatenate([got("H1"), got("H2")], axis=1),
  "MHA":  lambda: got("C") @ W_O,
  "R1":   lambda: got("X0") + got("MHA"),
  "mu1":  lambda: _ln(got("R1"))[0], "sd1": lambda: _ln(got("R1"))[1],
  "X1":   lambda: (got("R1") - got("mu1")[:, None]) / got("sd1")[:, None],
  "P":    lambda: got("X1") @ W1 + b1,
  "I":    lambda: _gelu(got("P")),
  "F":    lambda: got("I") @ W2 + b2,
  "R2":   lambda: got("X1") + got("F"),
  "mu2":  lambda: _ln(got("R2"))[0], "sd2": lambda: _ln(got("R2"))[1],
  "T":    lambda: (got("R2") - got("mu2")[:, None]) / got("sd2")[:, None],''' + HEAD_RULES + r'''
}
REF = {}
for _k, _f in _RULES.items():   # the full answer key, computed in order
    REF[_k] = _f()

_HINTS = {
  "ids": "Look each token up in the vocab dictionary. Position 1 is [MASK], not dog.",
  "seg": "0 from [CLS] up to and including the first [SEP], then 1.",
  "E_tok": "Row r is the row of W_tok whose number is ids[r]. Just copy the row.",
  "E_seg": "Row r is W_seg[seg[r]]: row 0 for sentence A, row 1 for sentence B.",
  "E_pos": "Row r is W_pos[r]. BERT learns these rows; no sines or cosines.",
  "X_sum": "Add the three tables entry by entry. No scaling by sqrt(H) in BERT.",
  "mu0": "The mean of each row of X_sum (sum the 4 numbers, divide by 4).",
  "sd0": "For each row: average the squared distances from the row's mean, then take the square root.",
  "X0": "(X_sum[r, c] - mu0[r]) / sd0[r]. This LayerNorm comes from the released code.",
  "S1s": "Divide every score by sqrt(d_k) = sqrt(2) ≈ 1.4142.",
  "S2s": "Divide every score by sqrt(d_k) = sqrt(2) ≈ 1.4142.",
  "A1": "For each row: take e**x of every entry, then divide each by the row's total. Each row must sum to 1.",
  "A2": "For each row: take e**x of every entry, then divide each by the row's total. Each row must sum to 1.",
  "C": "Put H1 and H2 side by side: the first d_k columns are H1, the next d_k are H2.",
  "R1": "Residual: add the layer's input X0 to MHA, entry by entry.",
  "mu1": "The mean of each row of R1.",
  "sd1": "The standard deviation of each row of R1 (divide by 4, not 3).",
  "X1": "(R1[r, c] - mu1[r]) / sd1[r].",
  "P": "X1 @ W1, then add b1[c] to every entry in column c.",
  "I": "GELU(x) = 0.5*x*(1 + tanh(0.7979*(x + 0.044715*x**3))). Use np.tanh for single numbers.",
  "F": "I @ W2, then add b2[c] to every entry in column c.",
  "R2": "Residual again: X1 + F, entry by entry.",
  "mu2": "The mean of each row of R2.",
  "sd2": "The standard deviation of each row of R2.",
  "T": "(R2[r, c] - mu2[r]) / sd2[r]. Row 0 is C, the [CLS] vector.",''' + HEAD_HINTS + r'''
}
_MATMUL = {"Q1": ("X0", W_Q1, "W_Q1"), "K1": ("X0", W_K1, "W_K1"), "V1": ("X0", W_V1, "W_V1"),
           "Q2": ("X0", W_Q2, "W_Q2"), "K2": ("X0", W_K2, "W_K2"), "V2": ("X0", W_V2, "W_V2"),
           "S1": ("Q1", None, "K1ᵀ"), "S2": ("Q2", None, "K2ᵀ"),
           "H1": ("A1", None, "V1"), "H2": ("A2", None, "V2"), "MHA": ("C", W_O, "W_O")}

def _hint(name, r, c):
    if name in _MATMUL:
        a, B, bname = _MATMUL[name]
        A_ = got(a)
        if B is None:
            B = {"S1": lambda: got("K1").T, "S2": lambda: got("K2").T,
                 "H1": lambda: got("V1"), "H2": lambda: got("V2")}[name]()
        terms = " + ".join(f"({A_[r, j]:.2f})×({B[j, c]:.2f})" for j in range(A_.shape[1]))
        return f"{name}[{r},{c}] = row {r} of {a} · column {c} of {bname} = {terms}"
    return _HINTS.get(name, "")

def check(name, yours):
    """Grade your matrix. Wrong or blank cells are listed with a hint for the first one."""
    exp = np.atleast_1d(np.asarray(_RULES[name](), dtype=float))
    y = np.atleast_1d(np.asarray(yours, dtype=float))
    if y.shape != exp.shape:
        print(f"✗ {name}: shape {y.shape}, but it should be {exp.shape}."); return
    if np.isnan(y).all():
        print(f"… {name} is still blank. Replace every nan with your number."); return
    tol = 0.02 + 0.01 * np.abs(exp)
    bad = np.argwhere(np.isnan(y) | (np.abs(y - exp) > tol))
    if len(bad) == 0:
        _ok[name] = y if np.ndim(yours) else float(y[0])
        print(f"✓ {name}{' ' + str(exp.shape) if np.ndim(yours) else ''} is correct. Nice work!"); return
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
    print(f"{name} {np.shape(exp)} =\n{np.round(exp, 4)}")

def check_shapes(answers):
    """Grade a dictionary of shapes for BERT-base and a 128-token input."""
    PAPER = {"E_tok": (128, 768), "W_Q1": (768, 64), "Q1": (128, 64), "S1": (128, 128), "A1": (128, 128),
             "H1": (128, 64), "C": (128, 768), "W_O": (768, 768), "P": (128, 3072), "T": (128, 768)''' + (''',
             "logits": (30000,), "W_nsp": (2, 768)''' if FULL else "") + r'''}
    right = 0
    for k, v in PAPER.items():
        mine = tuple(answers.get(k, ()))
        ok = mine == v; right += ok
        print(f"{'✓' if ok else '✗'} {k:6s} you said {mine or '(blank)'}" + ("" if ok else f", BERT-base has {v}"))
    print(f"\n{right} of {len(PAPER)} right.")

print("Ready. Tokens:", tokens, "| n =", n, "| H =", H, "| A =", A, "| d_k =", d_k, "| d_ff =", d_ff, "| V =", V)
'''

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
def code(s, tag=None):
    c = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}
    if tag: c["metadata"]["tags"] = [tag]
    return c

def blank(name, rows, cols=None, comment=""):
    if rows == 0:
        body = f"{name} = nan"
    elif cols is None:
        body = f"{name} = np.array([" + ", ".join(["nan"] * rows) + "])"
    else:
        row = "[" + ", ".join(["nan"] * cols) + "]"
        body = f"{name} = np.array([\n" + ",\n".join("    " + row for _ in range(rows)) + "\n])"
    return (f"# {comment}\n" if comment else "") + body

def fill(name, rows, cols=None, comment=""):
    return code(blank(name, rows, cols, comment) + f"\ncheck('{name}', {name})", tag=f"fill:{name}")

title = "BERT by hand: the whole pre-training pass" if FULL else "BERT by hand: the encoder"
cells = [
md(f"""
# {title}

This is the {'final homework for **BERT by Hand**' if FULL else 'homework for Part 1 of **BERT by Hand**'}. You'll push one tiny masked sentence pair through a BERT encoder layer{', then through both pre-training heads to the one summed loss,' if FULL else ''} and **write down every matrix yourself**.

**How it works**

1. Run the first code cell. It sets up the pair, a tiny vocabulary and all the weights.
2. Each step shows a code cell full of `nan`. Replace every `nan` with your number, then run the cell. It checks your answer straight away.
3. If something is off, the checker says which entries and gives a hint for the first one. Still stuck? Call `reveal('name')` to see the answer and carry on.

Work on paper or use Python as a calculator for single numbers (`np.exp(0.5)`, `np.tanh(1)`, `0.3*2 + 1.1*0.5`). Try not to use `@` on whole matrices: doing the dot products yourself is the point. Two decimals is enough. Each check uses *your* previous checked matrices, so small rounding differences won't snowball.

**The example.** Sentence A is "dog", sentence B is "barks". Packed, that's `[CLS] dog [SEP] barks [SEP]`, and "dog" was picked for masking, so the model reads `[CLS] [MASK] [SEP] barks [SEP]`. B really follows A, so the NSP label is IsNext.

**The sizes we use**

| setting | here | BERT-base |
|---|---|---|
| tokens n | 5 | up to 512 |
| hidden size H | 4 | 768 |
| heads A | 2 | 12 |
| d_k = H / A | 2 | 64 |
| d_ff | 8 | 3072 |
| layers L | 1 | 12 |
| vocabulary V | 8 | 30,000 |

Dropout is skipped. A few details (the LayerNorm on the embeddings, the GELU formula{', the MLM transform and the pooler' if FULL else ''}) come from Google's released code rather than the paper; the site marks them with a *code* tag.
"""),
code(HELPERS, tag="helpers"),

md("""
## Step 1: Input IDs and segment IDs

Look each of the 5 tokens up in `vocab`. Then write the segment ID of each position: 0 for `[CLS]`, sentence A and the first `[SEP]`; 1 for sentence B and the last `[SEP]`.

**Shape:** (n,) = (5,) each
"""),
fill("ids", 5, comment="one ID per token, [CLS] [MASK] [SEP] barks [SEP]"),
fill("seg", 5, comment="0 or 1 per token"),

md("""
## Step 2: Three lookups

BERT's input is built from three tables, each with H = 4 columns:

- `E_tok`: row `ids[r]` of the token table `W_tok` (8 × 4 here, 30,000 × 768 in BERT-base)
- `E_seg`: row `seg[r]` of the segment table `W_seg` (2 × 4)
- `E_pos`: row r of the **learned** position table `W_pos` (512 × 768 in BERT-base, no sines)

**Shape:** (n × H) = (5 × 4) each
"""),
code("print('W_tok =\\n', W_tok, '\\nW_seg =\\n', W_seg, '\\nW_pos =\\n', W_pos)"),
fill("E_tok", 5, 4, "row r = W_tok[ids[r]]; row 1 is the [MASK] row"),
fill("E_seg", 5, 4, "row r = W_seg[seg[r]]"),
fill("E_pos", 5, 4, "row r = W_pos[r]"),

md("""
## Step 3: Add them, then LayerNorm

`X_sum = E_tok + E_seg + E_pos`. Unlike the Transformer, there's no × √H scaling. The released code then normalises each row before the first layer: subtract the row's mean, divide by its standard deviation (γ = 1, β = 0 here).
"""),
fill("X_sum", 5, 4, "entry by entry"),
fill("mu0", 5, comment="mean of each row of X_sum"),
fill("sd0", 5, comment="standard deviation of each row (divide by 4)"),
fill("X0", 5, 4, "(X_sum[r, c] - mu0[r]) / sd0[r]"),

md("""
## Step 4: Queries, keys and values for each head

From here on the layer is the Transformer encoder you already know. Each head has its own W_Q, W_K and W_V, each (H × d_k) = (4 × 2):

(5 × 4) · (4 × 2) = (5 × 2)
"""),
md("### Head 1"),
code("print('W_Q1 =\\n', W_Q1, '\\nW_K1 =\\n', W_K1, '\\nW_V1 =\\n', W_V1)"),
fill("Q1", 5, 2, "X0 · W_Q1"),
fill("K1", 5, 2, "X0 · W_K1"),
fill("V1", 5, 2, "X0 · W_V1"),
md("### Head 2"),
code("print('W_Q2 =\\n', W_Q2, '\\nW_K2 =\\n', W_K2, '\\nW_V2 =\\n', W_V2)"),
fill("Q2", 5, 2, "X0 · W_Q2"),
fill("K2", 5, 2, "X0 · W_K2"),
fill("V2", 5, 2, "X0 · W_V2"),

md("""
## Step 5: Scores, scale, softmax

`S = Q · Kᵀ` is (5 × 2) · (2 × 5) = (5 × 5). There is **no mask**: `[MASK]` at position 1 can look at "barks" to its right as well as `[CLS]` to its left. That's the "bidirectional" in BERT's name. Then divide by √d_k = √2 and softmax each row.
"""),
fill("S1", 5, 5, "head 1: S1[r, c] = Q1[r] · K1[c]"),
fill("S2", 5, 5, "head 2"),
fill("S1s", 5, 5, "S1 / sqrt(2)"),
fill("S2s", 5, 5, "S2 / sqrt(2)"),
fill("A1", 5, 5, "head 1 attention weights, rows sum to 1"),
fill("A2", 5, 5, "head 2 attention weights"),

md("""
## Step 6: Blend the values, join the heads

`H = A · V` is (5 × 5) · (5 × 2) = (5 × 2). Then put the heads side by side, (5 × 4), and multiply by W_O (4 × 4).
"""),
fill("H1", 5, 2, "A1 · V1"),
fill("H2", 5, 2, "A2 · V2"),
fill("C", 5, 4, "[H1 | H2]"),
code("print(W_O)"),
fill("MHA", 5, 4, "C · W_O"),

md("""
## Step 7: Add & Norm (1)
"""),
fill("R1", 5, 4, "X0 + MHA"),
fill("mu1", 5, comment="mean of each row of R1"),
fill("sd1", 5, comment="standard deviation of each row of R1"),
fill("X1", 5, 4, "(R1[r, c] - mu1[r]) / sd1[r]"),

md("""
## Step 8: Feed-forward with GELU

The one change inside the layer: BERT uses **GELU** instead of ReLU (paper appendix A.2). The released code uses this formula:

`GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`, with √(2/π) ≈ 0.7979.

Big positive numbers pass almost unchanged, big negative ones become almost 0, and small negatives stay slightly negative.

(5 × 4) · (4 × 8) = (5 × 8), then (5 × 8) · (8 × 4) = (5 × 4)
"""),
code("print('W1 =\\n', W1, '\\nb1 =', b1)"),
fill("P", 5, 8, "X1 · W1 + b1"),
fill("I", 5, 8, "GELU(P), entry by entry"),
code("print('W2 =\\n', W2, '\\nb2 =', b2)"),
fill("F", 5, 4, "I · W2 + b2"),

md("""
## Step 9: Add & Norm (2), the encoder output T
"""),
fill("R2", 5, 4, "X1 + F"),
fill("mu2", 5, comment="mean of each row of R2"),
fill("sd2", 5, comment="standard deviation of each row of R2"),
fill("T", 5, 4, "(R2[r, c] - mu2[r]) / sd2[r]"),
]

if FULL:
    cells += [
md("""
## Step 10: The MLM head, part 1: transform the masked row

The masked LM head only reads the rows of T at masked positions, here row 1. The released code first passes that row through a dense layer, GELU and LayerNorm:

`pre_t = t · W_t + b_t`, `g = GELU(pre_t)`, `u = LayerNorm(g)`
"""),
code("print('W_t =\\n', W_t, '\\nb_t =', b_t)"),
fill("t", 4, comment="T row 1, the [MASK] position"),
fill("pre_t", 4, comment="t · W_t + b_t"),
fill("g", 4, comment="GELU(pre_t)"),
fill("mu_t", 0, comment="one number: the mean of g"),
fill("sd_t", 0, comment="one number: the standard deviation of g"),
fill("u", 4, comment="(g - mu_t) / sd_t"),

md("""
## Step 11: The MLM head, part 2: a probability for every piece

The output layer **reuses the token table**: the score for piece v is `u · W_tok[v]` (plus a bias, which is 0 here). Then a softmax over all 8 pieces, and the loss is −ln of the probability given to the true word, "dog".

(4,) · (4 × 8) = (8,)
"""),
fill("logits", 8, comment="logits[v] = u · W_tok[v]"),
fill("P_mlm", 8, comment="softmax of logits"),
fill("loss_mlm", 0, comment="-ln P_mlm[vocab['dog']]"),

md("""
## Step 12: The NSP head

The NSP head reads only C = T row 0. The released code's pooler applies a dense layer and tanh, then a 2-way classifier gives IsNext and NotNext scores. The loss is −ln of the probability of the true label (IsNext, index 0).
"""),
code("print('W_p =\\n', W_p, '\\nb_p =', b_p, '\\nW_nsp =\\n', W_nsp)"),
fill("c", 4, comment="T row 0, the [CLS] vector"),
fill("pre_p", 4, comment="c · W_p + b_p"),
fill("pooled", 4, comment="tanh(pre_p)"),
fill("nsp_logits", 2, comment="[IsNext, NotNext] = pooled · rows of W_nsp"),
fill("P_nsp", 2, comment="softmax of the two logits"),
fill("loss_nsp", 0, comment="-ln P_nsp[0]"),

md("""
## Step 13: Add the two losses

This is the only thing that's added in BERT: two numbers, the MLM loss and the NSP loss, during pre-training (paper appendix A.2). One backward pass from this sum trains both heads and the shared encoder at once.
"""),
fill("total", 0, comment="loss_mlm + loss_nsp"),

md("""
## Step 14: Using it: fill in the blank

At inference nothing is added and there's no loss: you only read P_mlm. Which piece gets the biggest probability? Write its **vocabulary ID**. (We picked these made-up weights so the guess comes out right; random weights would give a random guess.)

For a real task like MNLI or SQuAD you'd drop both heads and fine-tune a new small layer on C or T instead.
"""),
fill("pred_id", 0, comment="the vocab ID with the biggest P_mlm"),
code("""
inv = {i: t for t, i in vocab.items()}
print("BERT fills [MASK] with:", inv[int(got("pred_id"))], "| the hidden word was:", target)
"""),
    ]

done_keys = '["ids", "E_tok", "X_sum", "X0", "Q1", "S1", "A1", "H1", "C", "MHA", "X1", "P", "I", "F", "T"' + (', "u", "logits", "P_mlm", "loss_mlm", "pooled", "P_nsp", "loss_nsp", "total"' if FULL else "") + ']'
cells += [
md(f"""
## You did it

{'You took a masked sentence pair all the way to the one number pre-training minimises, and read off the fill-mask answer.' if FULL else 'T is the output of one BERT encoder layer: one 4-number vector per token, each mixed with context from both sentences and from both directions. BERT-base repeats this 12 times, each layer with its own weights.'}

Run the cell below to see your whole pass, step by step.
"""),
code(f"""
for k in {done_keys}:
    print(f"{{k:9s}} {{str(np.shape(got(k))):8s}} {{'✓ yours' if k in _ok else '· answer key'}}")
print(f"\\n{{len(_ok)}} of {{len(_RULES)}} matrices done by you.")
"""),

md("""
## Bonus: BERT-base sizes, by hand

No numbers this time, just shapes. Imagine a 128-token input and BERT-base (H 768, A 12, d_k 64, d_ff 3072). Fill in each shape as a tuple, like `(128, 768)`.
"""),
code("""
paper_shapes = {
    "E_tok": (),   # token embeddings after the lookup
    "W_Q1":  (),   # one head's query weights
    "Q1":    (),   # one head's queries
    "S1":    (),   # one head's scores
    "A1":    (),   # one head's attention weights
    "H1":    (),   # one head's output
    "C":     (),   # all heads concatenated
    "W_O":   (),   # output projection
    "P":     (),   # FFN hidden layer
    "T":     (),   # encoder output""" + ("""
    "logits": (),  # MLM scores at one masked position
    "W_nsp": (),   # the NSP classifier""" if FULL else "") + """
}
check_shapes(paper_shapes)
""", tag="fill:paper_shapes"),

md("""
## Want more?

- **Mask a different word.** In the first cell, set `tokens` to `["[CLS]", "dog", "[SEP]", "[MASK]", "[SEP]"]`, `mask_pos = 3` and `target = "barks"`, then redo the pass. The checker recomputes the answer key.
- **Make the weights your own.** Edit any weight matrix and rerun.
- **Compare with the site.** Every step here matches a step in BERT by Hand, which also animates each multiplication.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
print("cells:", len(cells))
