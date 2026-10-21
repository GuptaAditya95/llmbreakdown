"""Builds transformer-practice.ipynb: the encoder homework followed by one full decoder layer.

Usage: python3 make_transformer_practice.py OUT.ipynb
Reuses the encoder cells from make_encoder_practice.py (same folder), then adds the decoder:
shifted-right input, masked self-attention, cross-attention to Z, FFN, and the output softmax.
Every "fill" cell is graded against values computed from the learner's own checked matrices.
"""
import json, os, sys

_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, "make_encoder_practice.py"), encoding="utf-8").read()
ns = {}
exec(_src.split('nb = {"cells"')[0], ns)          # HELPERS, md, code, fill and the encoder cells
md, code, fill, enc_cells = ns["md"], ns["code"], ns["fill"], ns["cells"]

DEC_HELPERS = r'''# Run this cell before the decoder part. It adds the German words, the decoder's
# weights and the decoder's answer key to the checker. Again, no peeking at REF.

# ---- the shared vocabulary grows: the paper uses ONE vocabulary for both languages (§3.4) ----
vocab.update({"<s>": 6, "</s>": 7, "die": 8, "katze": 9, "saß": 10})
if len(W_emb) == 6:   # (so rerunning this cell is safe)
  W_emb = np.vstack([W_emb, np.array([
    [ 0.2,  0.3, -0.4, -0.3],   # 6 <s>   start token
    [-0.1,  0.1,  0.3, -0.2],   # 7 </s>  end token
    [-0.4, -0.3, -0.1, -0.1],   # 8 die
    [ 0.4, -0.1,  0.2,  0.1],   # 9 katze
    [-0.4,  0.2, -0.4, -0.4],   # 10 saß
  ])])
words = {i: w for w, i in vocab.items()}
V = len(vocab)   # 11 here, about 37,000 in the paper

# ---- the translation, shifted right ----
target    = "die katze saß"
dec_input = ["<s>", "die", "katze", "saß"]     # what the decoder reads
dec_label = ["die", "katze", "saß", "</s>"]    # what it must predict at each position
m = len(dec_input)                            # 4 positions (the source has n = 3)
inf = np.inf

# ---- masked self-attention weights (the decoder's own; not shared with the encoder) ----
W_mQ1 = np.array([[ 0.0, -0.5], [ 0.0,  0.0], [-1.0, -1.0], [ 0.5,  1.0]])
W_mK1 = np.array([[ 0.5,  0.5], [ 0.0, -1.0], [ 0.0,  0.0], [ 0.5,  0.0]])
W_mV1 = np.array([[ 0.0, -1.0], [ 0.5, -0.5], [ 0.0, -1.0], [ 0.0,  0.0]])
W_mQ2 = np.array([[ 0.0,  0.0], [-1.0,  0.0], [ 0.5,  1.0], [-0.5,  0.5]])
W_mK2 = np.array([[ 0.0,  0.0], [ 0.5, -0.5], [-0.5,  0.0], [ 0.0,  1.0]])
W_mV2 = np.array([[-0.5,  0.0], [ 1.0,  1.0], [ 0.5,  0.0], [ 0.5, -0.5]])
W_mO  = np.array([[ 0.0, -0.5,  0.0, -0.5],
                  [ 0.5, -0.5,  0.0,  0.0],
                  [ 0.5,  0.0, -0.5,  0.0],
                  [-0.5,  0.5,  0.5,  0.0]])

# ---- cross-attention weights: W_xQ is applied to the decoder, W_xK and W_xV to Z ----
W_xQ1 = np.array([[-1.0,  0.0], [-0.5,  0.0], [-0.5,  0.0], [ 0.5, -0.5]])
W_xK1 = np.array([[ 0.0,  0.0], [-0.5,  0.0], [ 0.5,  0.0], [ 0.5,  0.0]])
W_xV1 = np.array([[ 1.0, -1.0], [ 0.0,  1.0], [ 0.0, -1.0], [ 1.0, -0.5]])
W_xQ2 = np.array([[-1.0, -0.5], [ 1.0,  0.0], [-1.0,  0.0], [-1.0,  0.0]])
W_xK2 = np.array([[ 0.0,  0.5], [-1.0,  0.0], [ 0.0,  1.0], [-1.0,  0.0]])
W_xV2 = np.array([[ 0.0,  0.0], [ 0.5,  0.0], [ 0.5, -1.0], [-0.5, -0.5]])
W_xO  = np.array([[ 0.0,  0.0, -0.5,  0.0],
                  [ 0.5, -0.5,  0.0, -0.5],
                  [ 0.5,  0.0,  0.5,  0.5],
                  [ 0.5,  0.0,  0.0,  0.0]])

# ---- the decoder's feed-forward network ----
W1d = np.array([[-0.5,  0.5, -0.5,  0.0,  0.5, -0.5, -0.5,  0.0],
                [-1.0,  0.0, -0.5, -1.0, -1.0, -0.5,  0.0,  0.0],
                [-0.5, -1.0,  0.0,  0.0,  0.0,  0.0, -0.5, -1.0],
                [ 0.0, -0.5, -1.0,  1.0,  0.0,  0.0, -0.5, -0.5]])
b1d = np.array([0.0, 0.1, -0.1, 0.0, -0.1, 0.0, 0.1, 0.1])
W2d = np.array([[ 0.0,  0.0,  0.0, -0.5],
                [-0.5,  0.0, -0.5,  0.5],
                [ 0.5,  0.0,  0.5,  0.0],
                [ 0.0,  0.5,  0.0, -0.5],
                [ 0.0,  0.5,  0.0,  0.0],
                [ 0.0,  0.0,  0.5,  0.5],
                [ 0.0,  0.5,  0.5, -0.5],
                [ 0.0,  0.0,  0.0,  0.0]])
b2d = np.array([-0.1, 0.1, 0.0, -0.1])

MASK = np.triu(np.full((m, m), -np.inf), 1)    # used by the checker only; you build it yourself

# =============== the decoder checker (no need to read below) ===============
_DEC = {
  "ids_d": lambda: np.array([vocab[t] for t in dec_input], dtype=float),
  "Ed":    lambda: W_emb[got("ids_d").astype(int)],
  "Eds":   lambda: got("Ed") * np.sqrt(d_model),
  "PEd":   lambda: np.array([[f(p / got("div")[c // 2]) for c, f in
                              zip(range(d_model), [np.sin, np.cos] * (d_model // 2))] for p in range(m)]),
  "Y0":    lambda: got("Eds") + got("PEd"),
}
for _h in "12":
    _DEC.update({
      f"mQ{_h}": (lambda h: lambda: got("Y0") @ globals()[f"W_mQ{h}"])(_h),
      f"mK{_h}": (lambda h: lambda: got("Y0") @ globals()[f"W_mK{h}"])(_h),
      f"mV{_h}": (lambda h: lambda: got("Y0") @ globals()[f"W_mV{h}"])(_h),
      f"mS{_h}": (lambda h: lambda: got(f"mQ{h}") @ got(f"mK{h}").T)(_h),
      f"mS{_h}s": (lambda h: lambda: got(f"mS{h}") / np.sqrt(d_k))(_h),
      f"mS{_h}m": (lambda h: lambda: got(f"mS{h}s") + MASK)(_h),
      f"mA{_h}": (lambda h: lambda: _sm(got(f"mS{h}m")))(_h),
      f"mH{_h}": (lambda h: lambda: got(f"mA{h}") @ got(f"mV{h}"))(_h),
    })
_DEC.update({
  "mC":   lambda: np.concatenate([got("mH1"), got("mH2")], axis=1),
  "mMHA": lambda: got("mC") @ W_mO,
  "R1d":  lambda: got("Y0") + got("mMHA"),
  "mu1d": lambda: _ln(got("R1d"))[0], "sd1d": lambda: _ln(got("R1d"))[1],
  "Y1":   lambda: (got("R1d") - got("mu1d")[:, None]) / got("sd1d")[:, None],
})
for _h in "12":
    _DEC.update({
      f"xQ{_h}": (lambda h: lambda: got("Y1") @ globals()[f"W_xQ{h}"])(_h),
      f"xK{_h}": (lambda h: lambda: got("Z") @ globals()[f"W_xK{h}"])(_h),
      f"xV{_h}": (lambda h: lambda: got("Z") @ globals()[f"W_xV{h}"])(_h),
      f"xS{_h}": (lambda h: lambda: got(f"xQ{h}") @ got(f"xK{h}").T)(_h),
      f"xS{_h}s": (lambda h: lambda: got(f"xS{h}") / np.sqrt(d_k))(_h),
      f"xA{_h}": (lambda h: lambda: _sm(got(f"xS{h}s")))(_h),
      f"xH{_h}": (lambda h: lambda: got(f"xA{h}") @ got(f"xV{h}"))(_h),
    })
_DEC.update({
  "xC":    lambda: np.concatenate([got("xH1"), got("xH2")], axis=1),
  "xMHA":  lambda: got("xC") @ W_xO,
  "R2d":   lambda: got("Y1") + got("xMHA"),
  "mu2d":  lambda: _ln(got("R2d"))[0], "sd2d": lambda: _ln(got("R2d"))[1],
  "Y2":    lambda: (got("R2d") - got("mu2d")[:, None]) / got("sd2d")[:, None],
  "Pd":    lambda: got("Y2") @ W1d + b1d,
  "Hd":    lambda: np.maximum(0, got("Pd")),
  "Fd":    lambda: got("Hd") @ W2d + b2d,
  "R3d":   lambda: got("Y2") + got("Fd"),
  "mu3d":  lambda: _ln(got("R3d"))[0], "sd3d": lambda: _ln(got("R3d"))[1],
  "Yout":  lambda: (got("R3d") - got("mu3d")[:, None]) / got("sd3d")[:, None],
  "logits": lambda: got("Yout") @ W_emb.T,
  "probs": lambda: _sm(got("logits")),
  "pred":  lambda: got("probs").argmax(axis=1).astype(float),
  "nll":   lambda: -np.log(got("probs")[np.arange(m), [vocab[w] for w in dec_label]]),
})
_RULES.update(_DEC)
for _k, _f in _DEC.items():
    REF[_k] = _f()

_HINTS.update({
  "ids_d": "Look up each decoder INPUT token (⟨s⟩ first), not the words it has to predict.",
  "Ed": "Same lookup as the encoder: row r = W_emb[ids_d[r]].",
  "Eds": "Multiply every entry by sqrt(d_model) = 2, exactly like the encoder.",
  "PEd": "Same formula as the encoder's PE. Rows 0 to 2 are identical to PE; only row 3 (position 3) is new.",
  "Y0": "Eds + PEd, entry by entry.",
  "mS1s": "Divide every score by sqrt(2) ≈ 1.4142.", "mS2s": "Divide every score by sqrt(2) ≈ 1.4142.",
  "mS1m": "Keep entries with column ≤ row. Every entry ABOVE the diagonal (column > row) becomes -inf.",
  "mS2m": "Keep entries with column ≤ row. Every entry ABOVE the diagonal (column > row) becomes -inf.",
  "mA1": "Softmax each row. e**(-inf) = 0, so masked entries get exactly 0 weight; divide by the total of the rest.",
  "mA2": "Softmax each row. e**(-inf) = 0, so masked entries get exactly 0 weight; divide by the total of the rest.",
  "mC": "Put mH1 and mH2 side by side.",
  "R1d": "Residual: Y0 + mMHA.", "mu1d": "Mean of each row of R1d.", "sd1d": "Standard deviation of each row of R1d (divide by 4).",
  "Y1": "(R1d[r, c] - mu1d[r]) / sd1d[r].",
  "xS1s": "Divide every score by sqrt(2).", "xS2s": "Divide every score by sqrt(2).",
  "xA1": "Softmax each row over the 3 ENGLISH words. No mask: the whole source sentence is allowed.",
  "xA2": "Softmax each row over the 3 ENGLISH words. No mask: the whole source sentence is allowed.",
  "xC": "Put xH1 and xH2 side by side.",
  "R2d": "Residual: Y1 + xMHA.", "mu2d": "Mean of each row of R2d.", "sd2d": "Standard deviation of each row of R2d.",
  "Y2": "(R2d[r, c] - mu2d[r]) / sd2d[r].",
  "Pd": "Y2 @ W1d, then add b1d[c] to column c.", "Hd": "ReLU: negatives become 0.",
  "Fd": "Hd @ W2d, then add b2d[c] to column c.",
  "R3d": "Residual: Y2 + Fd.", "mu3d": "Mean of each row of R3d.", "sd3d": "Standard deviation of each row of R3d.",
  "Yout": "(R3d[r, c] - mu3d[r]) / sd3d[r].",
  "logits": "logits[r, w] = Yout[r] · W_emb[w]: the decoder's vector dotted with word w's embedding row.",
  "probs": "Softmax each row of logits over all V words.",
  "pred": "For each row of probs, the ID (column number) of the biggest probability.",
  "nll": "nll[r] = -ln(probs[r, id of dec_label[r]]): minus the log of the probability given to the CORRECT next word.",
})
_DEC_MM = {}
for _h in "12":
    _DEC_MM.update({
      f"mQ{_h}": ("Y0", (lambda h: lambda: globals()[f"W_mQ{h}"])(_h), f"W_mQ{_h}"),
      f"mK{_h}": ("Y0", (lambda h: lambda: globals()[f"W_mK{h}"])(_h), f"W_mK{_h}"),
      f"mV{_h}": ("Y0", (lambda h: lambda: globals()[f"W_mV{h}"])(_h), f"W_mV{_h}"),
      f"mS{_h}": (f"mQ{_h}", (lambda h: lambda: got(f"mK{h}").T)(_h), f"mK{_h}ᵀ"),
      f"mH{_h}": (f"mA{_h}", (lambda h: lambda: got(f"mV{h}"))(_h), f"mV{_h}"),
      f"xQ{_h}": ("Y1", (lambda h: lambda: globals()[f"W_xQ{h}"])(_h), f"W_xQ{_h}"),
      f"xK{_h}": ("Z", (lambda h: lambda: globals()[f"W_xK{h}"])(_h), f"W_xK{_h}"),
      f"xV{_h}": ("Z", (lambda h: lambda: globals()[f"W_xV{h}"])(_h), f"W_xV{_h}"),
      f"xS{_h}": (f"xQ{_h}", (lambda h: lambda: got(f"xK{h}").T)(_h), f"xK{_h}ᵀ"),
      f"xH{_h}": (f"xA{_h}", (lambda h: lambda: got(f"xV{h}"))(_h), f"xV{_h}"),
    })
_DEC_MM.update({"mMHA": ("mC", lambda: W_mO, "W_mO"), "xMHA": ("xC", lambda: W_xO, "W_xO"),
                "logits": ("Yout", lambda: W_emb.T, "W_embᵀ")})
_enc_hint = globals().get('_enc_hint', _hint)
def _hint(name, r, c):
    if name in _DEC_MM:
        a, Bf, bname = _DEC_MM[name]
        A, B = got(a), Bf()
        terms = " + ".join(f"({A[r, j]:.2f})×({B[j, c]:.2f})" for j in range(A.shape[1]))
        return f"{name}[{r},{c}] = row {r} of {a} · column {c} of {bname} = {terms}"
    return _enc_hint(name, r, c)

def check(name, yours):
    """Grade your matrix. Wrong or blank cells are listed with a hint for the first one."""
    exp = _RULES[name]()
    y = np.asarray(yours, dtype=float)
    if y.shape != exp.shape:
        print(f"✗ {name}: shape {y.shape}, but it should be {exp.shape}."); return
    if np.isnan(y).all():
        print(f"… {name} is still blank. Replace every nan with your number."); return
    fin = np.isfinite(exp)
    tol = 0.02 + 0.01 * np.abs(np.where(fin, exp, 0))
    with np.errstate(invalid="ignore"):
        good = np.where(fin, np.abs(y - np.where(fin, exp, 0)) <= tol, y == exp)
    if name == "pred":
        good = y == exp
    bad = np.argwhere(np.isnan(y) | ~good)
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

def check_dec_shapes(answers):
    """Grade a dictionary of decoder shapes for the paper's sizes (6 English tokens, 7 decoder positions)."""
    PAPER = {"Y0": (7, 512), "mS1": (7, 7), "mA1": (7, 7), "xQ1": (7, 64), "xK1": (6, 64),
             "xS1": (7, 6), "xH1": (7, 64), "Pd": (7, 2048), "Yout": (7, 512), "logits": (7, 37000)}
    right = 0
    for k, v in PAPER.items():
        mine = tuple(answers.get(k, ()))
        ok = mine == v; right += ok
        print(f"{'✓' if ok else '✗'} {k:6s} you said {mine or '(blank)'}" + ("" if ok else f", the paper has {v}"))
    print(f"\n{right} of {len(PAPER)} right.")

print("Decoder ready. Source:", sentence.split(), "→ decoder input:", dec_input, "| m =", m, "| V =", V)
'''

def mdc(s): return md(s)

intro = md("""
# The Transformer by hand: encoder and decoder

This is the full homework for **Attention by Hand**. You'll push one short sentence through a whole encoder layer, then translate it with a whole decoder layer, and **write down every matrix yourself**: from the token IDs to the probabilities of the next German word.

**How it works**

1. Run the first code cell. It sets up the sentence, a tiny vocabulary and the encoder's weights. Part 2 has its own setup cell for the decoder.
2. Each step shows a code cell full of `nan`. Replace every `nan` with your number, then run the cell. It checks your answer straight away.
3. If something is off, the checker says which entries and gives a hint for the first one. Still stuck? Call `reveal('name')` to see the answer and carry on.

Work on paper or use Python as a calculator for single numbers (`np.sin(1)`, `np.exp(0.5)`, `0.3*2 + 1.1*0.5`). Try not to use `@` to multiply whole matrices: doing the dot products yourself is the point. Two decimals is enough. Each check uses *your* previous checked matrices, so small rounding differences won't snowball.

**The sizes we use** (the same idea as the site, just small enough to do by hand)

| setting | here | paper |
|---|---|---|
| source length n | 3 ("the cat sat") | any |
| decoder positions m | 4 (⟨s⟩ die katze saß) | any |
| d_model | 4 | 512 |
| heads h | 2 | 8 |
| d_k = d_model / h | 2 | 64 |
| d_ff | 8 | 2048 |
| vocabulary V | 11 | ≈ 37,000 |
| layers N | 1 encoder + 1 decoder | 6 + 6 |

Dropout is skipped, as it would be when the model is used after training.

---

# Part 1: the encoder
""")

# Encoder cells, without their stand-alone intro and the "Want more?" ending.
body = enc_cells[1:]
cut = next(i for i, c in enumerate(body) if c["cell_type"] == "markdown" and c["source"].startswith("## Bonus"))
body = body[:cut]
for c in body:
    if c["cell_type"] == "markdown" and c["source"].startswith("## You did it"):
        c["source"] = ("## Encoder done\n\nZ is the output of one encoder layer: one 4-number vector per English token, "
                       "mixed with context from the whole sentence. In the paper you'd repeat the layer 6 times. "
                       "Here Z goes straight to the decoder.\n\nRun the cell below to see which encoder matrices you did yourself.")

dec = [
md("""
---

# Part 2: the decoder

The decoder is a **next-word predictor**. At every position it answers one question: *given the English sentence and the German words so far, what is the next German word?* Everything new in this part follows from that job:

- it reads the German sentence **shifted right**, starting with a start token ⟨s⟩,
- its self-attention is **masked**, so no position can peek at the word it's supposed to predict,
- its **cross-attention** takes queries from the German side and keys and values from the encoder output Z,
- it ends with a **linear layer and softmax** that turn each row into probabilities over the vocabulary.

We do the training-time pass: the whole (correct) translation goes in at once, and all 4 predictions come out together.

Run the setup cell first.
"""),
code(DEC_HELPERS, tag="helpers-decoder"),

md("""
## Step 15: Shift the target right and tokenize

The translation of "the cat sat" is "die katze saß". The decoder reads it **shifted one place to the right**, behind a start token:

| position | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| decoder input | ⟨s⟩ | die | katze | saß |
| must predict | die | katze | saß | ⟨/s⟩ |

Position 2 has seen `⟨s⟩ die katze` and must predict `saß`. Look up the IDs of the **decoder input** in `vocab` (run `print(vocab)` if you like).

**Shape:** (m,) = (4,)
"""),
code("print(vocab)"),
fill("ids_d", 4, comment="one ID per decoder input token, ⟨s⟩ first"),

md("""
## Step 16: Embed, scale and add positions

Exactly the encoder's steps 2 to 5, now for the 4 German input tokens. The embedding table is the **same** shared `W_emb` (now 11 rows), and the positional encoding uses the same formula. Rows 0 to 2 of `PEd` are identical to the encoder's `PE`, because position 1 means the same thing in both stacks.

**Shape:** (m × d_model) = (4 × 4)
"""),
fill("Ed", 4, 4, "row r = W_emb[ids_d[r]]"),
fill("Eds", 4, 4, "Ed × sqrt(4) = Ed × 2"),
fill("PEd", 4, 4, "positions 0, 1, 2, 3; 4 decimals"),
fill("Y0", 4, 4, "Eds + PEd: the input to decoder layer 1"),

md("""
## Step 17: Masked self-attention, queries, keys and values

The first sublayer works like encoder self-attention: Q, K and V **all come from Y0**. The decoder has its own weights (`W_mQ1` and so on).

(4 × 4) · (4 × 2) = (4 × 2)
"""),
code("print('W_mQ1 =\\n', W_mQ1, '\\nW_mK1 =\\n', W_mK1, '\\nW_mV1 =\\n', W_mV1)"),
md("### Head 1"),
fill("mQ1", 4, 2, "Y0 · W_mQ1"),
fill("mK1", 4, 2, "Y0 · W_mK1"),
fill("mV1", 4, 2, "Y0 · W_mV1"),
md("### Head 2"),
code("print('W_mQ2 =\\n', W_mQ2, '\\nW_mK2 =\\n', W_mK2, '\\nW_mV2 =\\n', W_mV2)"),
fill("mQ2", 4, 2, "Y0 · W_mQ2"),
fill("mK2", 4, 2, "Y0 · W_mK2"),
fill("mV2", 4, 2, "Y0 · W_mV2"),

md("""
## Step 18: Scores, then scale

Every German position against every German position: (4 × 2) · (2 × 4) = (4 × 4). Then divide by √d_k = √2.
"""),
fill("mS1", 4, 4, "mQ1 · mK1ᵀ"),
fill("mS1s", 4, 4, "mS1 / sqrt(2)"),
fill("mS2", 4, 4, "mQ2 · mK2ᵀ"),
fill("mS2s", 4, 4, "mS2 / sqrt(2)"),

md("""
## Step 19: The mask

This is the one new idea in decoder self-attention. Row r may only look at columns 0 to r. Every entry **above the diagonal** (column > row) is replaced with **−∞** (§3.2.3). Type it as `-inf`.

```
row ⟨s⟩    [ ok   -inf -inf -inf ]
row die    [ ok   ok   -inf -inf ]
row katze  [ ok   ok   ok   -inf ]
row saß    [ ok   ok   ok   ok   ]
```

Why? Position 2 must *predict* "saß", and "saß" is sitting right there at input position 3. Without the mask, the model would just copy it.
"""),
fill("mS1m", 4, 4, "mS1s with -inf above the diagonal"),
fill("mS2m", 4, 4, "mS2s with -inf above the diagonal"),

md("""
## Step 20: Softmax with the mask

Softmax each row as before. Since e<sup>−∞</sup> = 0, every masked entry gets **exactly 0**, and the remaining entries of the row share the whole weight. Row ⟨s⟩ can only see itself, so it's `[1, 0, 0, 0]`.
"""),
fill("mA1", 4, 4, "head 1; zeros above the diagonal"),
fill("mA2", 4, 4, "head 2"),

md("""
## Step 21: Blend the values, join the heads, Add & Norm

`mH = mA · mV` (4 × 2). The zeros in mA mean that row r only mixes values from positions 0 to r. Then concatenate the heads, multiply by `W_mO`, add the residual and normalize each row, exactly like the encoder.
"""),
fill("mH1", 4, 2, "mA1 · mV1"),
fill("mH2", 4, 2, "mA2 · mV2"),
fill("mC", 4, 4, "[mH1 | mH2]"),
code("print(W_mO)"),
fill("mMHA", 4, 4, "mC · W_mO"),
fill("R1d", 4, 4, "Y0 + mMHA"),
fill("mu1d", 4, comment="mean of each row of R1d"),
fill("sd1d", 4, comment="standard deviation of each row of R1d"),
fill("Y1", 4, 4, "(R1d - mu1d) / sd1d, row by row"),

md("""
## Step 22: Cross-attention, where Q, K and V come from

This is the bridge between the two halves (§3.2.3, "encoder-decoder attention"). The arithmetic is the same as before. What changes is **where the inputs come from**:

- **Q from the decoder:** `xQ = Y1 · W_xQ` → (4 × 2), one query per German position
- **K and V from the encoder:** `xK = Z · W_xK`, `xV = Z · W_xV` → (3 × 2), one per English word

The question comes from the German side ("which English words matter for my next word?"). What is searched, and what is handed back, comes from the English side.
"""),
code("print('Z (from Part 1) =\\n', np.round(got('Z'), 4))\nprint('W_xQ1 =\\n', W_xQ1, '\\nW_xK1 =\\n', W_xK1, '\\nW_xV1 =\\n', W_xV1)"),
md("### Head 1"),
fill("xQ1", 4, 2, "Y1 · W_xQ1   (decoder side)"),
fill("xK1", 3, 2, "Z · W_xK1    (encoder side: 3 rows!)"),
fill("xV1", 3, 2, "Z · W_xV1    (encoder side)"),
md("### Head 2"),
code("print('W_xQ2 =\\n', W_xQ2, '\\nW_xK2 =\\n', W_xK2, '\\nW_xV2 =\\n', W_xV2)"),
fill("xQ2", 4, 2, "Y1 · W_xQ2"),
fill("xK2", 3, 2, "Z · W_xK2"),
fill("xV2", 3, 2, "Z · W_xV2"),

md("""
## Step 23: Cross-attention scores and softmax

(4 × 2) · (2 × 3) = **(4 × 3)**. The score table is no longer square: rows are German positions, columns are English words. Scale by √2, then softmax each row. **No mask here:** the whole English sentence is known from the start.
"""),
fill("xS1", 4, 3, "xQ1 · xK1ᵀ"),
fill("xS1s", 4, 3, "xS1 / sqrt(2)"),
fill("xA1", 4, 3, "softmax of each row; each row adds up to 1"),
fill("xS2", 4, 3, "xQ2 · xK2ᵀ"),
fill("xS2s", 4, 3, "xS2 / sqrt(2)"),
fill("xA2", 4, 3, "softmax of each row"),

md("""
## Step 24: Blend the English values, join the heads, Add & Norm

(4 × 3) · (3 × 2) = (4 × 2). Each German position now pulls in a mix of the English words' information. The output has 4 rows because there are 4 **queries**.
"""),
fill("xH1", 4, 2, "xA1 · xV1"),
fill("xH2", 4, 2, "xA2 · xV2"),
fill("xC", 4, 4, "[xH1 | xH2]"),
code("print(W_xO)"),
fill("xMHA", 4, 4, "xC · W_xO"),
fill("R2d", 4, 4, "Y1 + xMHA"),
fill("mu2d", 4, comment="mean of each row of R2d"),
fill("sd2d", 4, comment="standard deviation of each row of R2d"),
fill("Y2", 4, 4, "(R2d - mu2d) / sd2d"),

md("""
## Step 25: Feed-forward network and the last Add & Norm

Same recipe as the encoder's FFN, with the decoder's own weights. Each position goes through on its own.
"""),
code("print('W1d =\\n', W1d, '\\nb1d =', b1d)"),
fill("Pd", 4, 8, "Y2 · W1d + b1d"),
fill("Hd", 4, 8, "ReLU(Pd)"),
code("print('W2d =\\n', W2d, '\\nb2d =', b2d)"),
fill("Fd", 4, 4, "Hd · W2d + b2d"),
fill("R3d", 4, 4, "Y2 + Fd"),
fill("mu3d", 4, comment="mean of each row of R3d"),
fill("sd3d", 4, comment="standard deviation of each row of R3d"),
fill("Yout", 4, 4, "(R3d - mu3d) / sd3d: the decoder's output"),

md("""
## Step 26: Linear layer: a score for every word

The decoder turns each row into one score (a *logit*) per vocabulary word. The paper reuses the embedding table for this, transposed (§3.4, "weight tying"):

`logits = Yout · W_embᵀ` → (4 × 4) · (4 × 11) = (4 × 11)

Entry [r, w] is the dot product of position r's vector with word w's embedding: "how much does what I'm saying look like word w?" For this one step it's fine to use Python for each single dot product, like `Yout[0] @ W_emb[8]`, since there are 44 of them.
"""),
code("print(W_emb)\nprint(words)"),
fill("logits", 4, 11, "row = decoder position, column = word ID 0…10"),

md("""
## Step 27: Softmax: probabilities for the next word

Softmax each row over all 11 words. Row r is the model's guess for the word at target position r.

Then read off the **prediction** (the ID with the highest probability) and the **loss** for each position: minus the log of the probability the model gave the correct next word. Training lowers this number.

The weights are made up, so don't expect the right translation: a trained model would put most of the probability on "die", "katze", "saß" and ⟨/s⟩.
"""),
fill("probs", 4, 11, "softmax of each row of logits"),
fill("pred", 4, comment="the word ID with the highest probability in each row"),
fill("nll", 4, comment="-ln(probs[r, correct next word's ID])"),
code("""
print("position  input   must predict   model's guess   P(correct)   loss")
for r in range(m):
    p = got("probs")[r]; t = vocab[dec_label[r]]
    print(f"{r:8d}  {dec_input[r]:6s}  {dec_label[r]:13s}  {words[int(p.argmax())]:14s}  {p[t]:10.3f}  {-np.log(p[t]):5.2f}")
print(f"\\nAverage loss over the {m} positions: {got('nll').mean():.3f}   (cross-entropy, §5.4; the paper also uses label smoothing 0.1)")
"""),

md("""
## You did it

You translated "the cat sat" through a whole Transformer by hand: one encoder layer, one decoder layer, and the output softmax. All 4 predictions came out of **one** pass, because the mask let every position train at once (this is *teacher forcing*).

At inference time there's no German sentence to feed in. The encoder runs once, then the decoder runs again for every new word: start with `[⟨s⟩]`, pick the most likely word, append it, and repeat until it picks ⟨/s⟩.

Run the cell below to see how much of the pass you did yourself.
"""),
code("""
keys = list(_RULES)
mine = [k for k in keys if k in _ok]
print(f"{len(mine)} of {len(keys)} matrices done by you.")
print("still from the answer key:", [k for k in keys if k not in _ok] or "none!")
"""),

md("""
## Bonus: the paper's sizes, by hand

No numbers, just shapes. Imagine a 6-token English sentence, a German translation that makes the decoder input 7 positions long (⟨s⟩ + 6 words), and the paper's sizes (d_model 512, h 8, d_k 64, d_ff 2048, V ≈ 37,000). Fill in each shape as a tuple.
"""),
code("""
paper_shapes = {"E": (), "W_Q1": (), "Q1": (), "S1": (), "A1": (), "H1": (), "C": (), "W_O": (), "P": (), "Z": ()}
check_shapes(paper_shapes)      # the encoder's shapes (6 tokens)
""", tag="fill:paper_shapes"),
code("""
dec_shapes = {
    "Y0":     (),   # decoder input after embedding + PE
    "mS1":    (),   # one head's masked self-attention scores
    "mA1":    (),   # its attention weights
    "xQ1":    (),   # cross-attention queries (one head)
    "xK1":    (),   # cross-attention keys (one head)
    "xS1":    (),   # cross-attention scores
    "xH1":    (),   # cross-attention head output
    "Pd":     (),   # FFN hidden layer
    "Yout":   (),   # decoder output
    "logits": (),   # one score per vocabulary word
}
check_dec_shapes(dec_shapes)
""", tag="fill:dec_shapes"),

md("""
## Want more?

- **Predict one word at a time.** In the decoder setup cell, change `dec_input` to `["<s>", "die"]` and `dec_label` to `["die", "katze"]`, rerun it, and redo Part 2 with 2 rows (the `nan` cells need 2 rows too). Compare with your 4-row answers: the rows for ⟨s⟩ and "die" come out exactly the same, because the mask stops them from seeing later words. That's why real models cache them while generating.
- **Make the weights your own.** Edit any weight matrix and rerun. The checker recomputes the answer key for you.
- **Compare with the site.** Every step here matches a step in Attention by Hand, which also animates each multiplication.
"""),
]

cells = [intro] + body + dec
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:03d}"
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("cells:", len(cells))
