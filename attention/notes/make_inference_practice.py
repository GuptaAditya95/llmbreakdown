"""Builds inference-practice.ipynb: translating "the cat sat" one word at a time.

Usage: python3 make_inference_practice.py OUT.ipynb
Reuses the setup cells of make_transformer_practice.py (same folder), so the model is the same tiny
encoder + decoder. The learner turns logits into probabilities and greedy picks until </s>, pushes one
new token through self-attention with a KV cache, and runs two turns of beam search by hand.
Every "fill" cell is graded against values computed from the learner's own checked answers.
"""
import json, os, sys

_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, "make_transformer_practice.py"), encoding="utf-8").read()
_tns = {"__file__": os.path.join(_here, "make_transformer_practice.py")}
exec(_src.split("def mdc(s)")[0], _tns)        # DEC_HELPERS, plus the encoder generator's namespace
HELPERS, DEC_HELPERS = _tns["ns"]["HELPERS"], _tns["DEC_HELPERS"]
md, code, fill = _tns["md"], _tns["code"], _tns["fill"]

MAX_TURNS = 4   # kept short so the arithmetic stays humane; the paper allows n + 50

INF_HELPERS = r'''# Run this cell third. It adds the decoder as a function you can call on any list of tokens,
# and the checker for this notebook. Please don't peek at the rules: they are the answer key.

def got(k):   # your checked answer if you have one, otherwise the answer key
    return _ok[k] if k in _ok else REF[k] if k in REF else _RULES[k]()

MAX_TURNS = %d   # stop here if </s> never comes (paper: source length + 50, §6.1)
Zf = REF["Z"]    # the encoder output for "the cat sat", computed ONCE and reused every turn

def _pe(pos):
    return np.array([f(pos / 10000 ** (2 * (c // 2) / d_model)) for c, f in zip(range(d_model), [np.sin, np.cos] * (d_model // 2))])
def _y0(tokens):
    return np.array([W_emb[vocab[t]] * np.sqrt(d_model) + _pe(p) for p, t in enumerate(tokens)])
def _mha(Xq, Xkv, Ws, WO, masked):
    outs = []
    for WQ, WK, WV in Ws:
        S = (Xq @ WQ) @ (Xkv @ WK).T / np.sqrt(d_k)
        if masked:
            S = S + np.triu(np.full(S.shape, -np.inf), 1)
        outs.append(_sm(S) @ (Xkv @ WV))
    return np.concatenate(outs, axis=1) @ WO
def _norm(R): return _ln(R)[2]

def decoder_output(tokens):
    """The whole decoder layer on a list of tokens (you built every piece of it in the last notebook)."""
    Y0 = _y0(tokens)
    Y1 = _norm(Y0 + _mha(Y0, Y0, [(W_mQ1, W_mK1, W_mV1), (W_mQ2, W_mK2, W_mV2)], W_mO, True))
    Y2 = _norm(Y1 + _mha(Y1, Zf, [(W_xQ1, W_xK1, W_xV1), (W_xQ2, W_xK2, W_xV2)], W_xO, False))
    return _norm(Y2 + np.maximum(0, Y2 @ W1d + b1d) @ W2d + b2d)

NEVER_WRITTEN = [vocab["<pad>"], vocab["<s>"]]   # the decoder never has to write these two

def last_logits(tokens):
    """Logits of the LAST row only, rounded to 2 decimals so you can work with them by hand.
    <pad> and <s> get -inf: they are never a correct next word, so they are ruled out."""
    L = np.round(decoder_output(tokens)[-1] @ W_emb.T, 2)
    L[NEVER_WRITTEN] = -np.inf
    return L

def tokens_for_turn(t):
    """<s> followed by the words you picked at turns 1 .. t-1."""
    return ["<s>"] + [words[int(got(f"w{k}")[0])] for k in range(1, t)]

def show_logits(t):
    toks = tokens_for_turn(t); L = last_logits(toks)
    print(f"turn {t}: the decoder reads {toks}")
    print("last-row logits:")
    for i in range(V): print(f"   {i:2d} {words[i]:6s} {L[i]:6.2f}")
    return L

# ---- the answer key, as rules that use YOUR earlier checked answers ----
def _greedy_rules(t):
    return {f"p{t}": lambda: _sm(last_logits(tokens_for_turn(t))[None, :])[0],
            f"w{t}": lambda: np.array([float(np.argmax(got(f"p{t}")))])}
for _t in range(1, MAX_TURNS + 1):
    _RULES.update(_greedy_rules(_t))

def _safe_log(P): return np.log(P, where=P > 0, out=np.full(P.shape, -np.inf))
_new = lambda: words[int(got("w1")[0])]          # the word you picked at turn 1
_RULES.update({
  "k_cache": lambda: _y0(["<s>"]) @ W_mK1,      # given: the cache after turn 1
  "v_cache": lambda: _y0(["<s>"]) @ W_mV1,
  "y0_new": lambda: (W_emb[vocab[_new()]] * np.sqrt(d_model) + _pe(1))[None, :],
  "q_new":  lambda: got("y0_new") @ W_mQ1,
  "k_new":  lambda: got("y0_new") @ W_mK1,
  "v_new":  lambda: got("y0_new") @ W_mV1,
  "K_cache": lambda: np.vstack([got("k_cache"), got("k_new")]),
  "V_cache": lambda: np.vstack([got("v_cache"), got("v_new")]),
  "s_new":  lambda: got("q_new") @ got("K_cache").T / np.sqrt(d_k),
  "a_new":  lambda: _sm(got("s_new")),
  "h_new":  lambda: got("a_new") @ got("V_cache"),
  "beam1_ids":    lambda: np.argsort(-got("p1"))[:2].astype(float),
  "beam1_scores": lambda: np.log(got("p1")[got("beam1_ids").astype(int)]),
  "cand": lambda: got("beam1_scores")[:, None] + _safe_log(np.vstack([
              _sm(last_logits(["<s>", words[int(i)]])[None, :])[0] for i in got("beam1_ids")])),
  "beam2": lambda: np.array(sorted([divmod(int(k), V) for k in np.argsort(-got("cand").ravel())[:2]],
              key=lambda rc: -got("cand")[rc]), dtype=float),
  "lp": lambda: np.array([((5 + L) / 6) ** 0.6 for L in (1, 2, 3, 4)]),
})
_EXACT = {f"w{t}" for t in range(1, MAX_TURNS + 1)} | {"beam1_ids", "beam2"}
_HINTS.update({f"p{t}": "Softmax: e**logit for every word (e**-inf = 0), then divide each by their total. The 11 numbers add up to 1." for t in range(1, MAX_TURNS + 1)})
_HINTS.update({f"w{t}": "Greedy: the ID (0 to 10) of the biggest probability in the row you just checked." for t in range(1, MAX_TURNS + 1)})
_HINTS.update({
  "y0_new": "Same recipe as always: W_emb row of the new word × 2, plus PE for position 1 (sin(1), cos(1), sin(0.01), cos(0.01)).",
  "q_new": "y0_new · W_mQ1: one row times a 4 × 2 matrix.", "k_new": "y0_new · W_mK1.", "v_new": "y0_new · W_mV1.",
  "K_cache": "Stack them: row 0 is k_cache (from turn 1), row 1 is k_new.", "V_cache": "Row 0 is v_cache, row 1 is v_new.",
  "s_new": "q_new · K_cacheᵀ, then divide by sqrt(2). One score per cached token.",
  "a_new": "Softmax of the 2 scores. No mask needed: both cached tokens are in the past.",
  "h_new": "a_new · V_cache: a weighted mix of the 2 cached value rows.",
  "beam1_ids": "The IDs of the two biggest probabilities in p1, biggest first.",
  "beam1_scores": "ln of their probabilities (np.log is fine as a calculator).",
  "cand": "cand[b, w] = beam1_scores[b] + ln P(word w | <s>, beam b's word).",
  "beam2": "Find the two biggest entries of cand; write each as [row, column], biggest first.",
  "lp": "((5 + L) / 6) ** 0.6 for L = 1, 2, 3, 4.",
})

def check(name, yours):
    """Grade your answer. Wrong or blank entries are listed with a hint."""
    exp = np.asarray(_RULES[name](), dtype=float)
    y = np.asarray(yours, dtype=float)
    if y.shape != exp.shape:
        print(f"✗ {name}: shape {y.shape}, but it should be {exp.shape}."); return
    if np.isnan(y).all():
        print(f"… {name} is still blank. Replace every nan with your number."); return
    with np.errstate(invalid="ignore"):
        good = (y == exp) if name in _EXACT else (y == exp) | (np.abs(y - exp) <= 0.02 + 0.01 * np.abs(np.where(np.isfinite(exp), exp, 0)))
    bad = np.argwhere(np.isnan(y) | ~good)
    if len(bad) == 0:
        _ok[name] = y
        print(f"✓ {name} {exp.shape} is correct. Nice work!"); return
    print(f"✗ {name}: {len(bad)} of {exp.size} entries need another look:")
    for idx in bad[:6]:
        print(f"   {name}[{', '.join(map(str, idx))}]: you have {y[tuple(idx)]:.4g}")
    print("   Hint:", _HINTS.get(name, ""))
    print(f"   Stuck? reveal('{name}') shows the answer and lets you carry on.")

def reveal(name):
    """Show the answer and use it from here on."""
    exp = np.asarray(_RULES[name](), dtype=float); _ok[name] = exp
    print(f"{name} {exp.shape} =\n{np.round(exp, 4)}")

print("Inference ready. Z is", Zf.shape, "and stays fixed. Vocabulary:", words)
''' % MAX_TURNS

# ---- run the reference greedy decode here, to know how many turns to lay out ----
_g = {}
exec(HELPERS, _g); exec(DEC_HELPERS, _g); exec(INF_HELPERS, _g)
REF_PICKS = []
for t in range(1, MAX_TURNS + 1):
    w = int(_g["_RULES"][f"w{t}"]()[0])
    _g["_ok"][f"w{t}"] = _g["np"].array([float(w)]); _g["_ok"][f"p{t}"] = _g["_RULES"][f"p{t}"]()
    REF_PICKS.append(_g["words"][w])
    if _g["words"][w] == "</s>":
        break
TURNS = len(REF_PICKS)

intro = md(f"""
# The Transformer by hand: inference

The last notebook trained-style: the whole German sentence went in at once. This one does what a finished model does with a sentence it has never seen: it **writes the translation itself, one token at a time**, feeding every word it picks back in, until it writes `</s>`.

It uses the same tiny model as the encoder + decoder notebook ("the cat sat", d_model 4, 2 heads, a vocabulary of 11). You don't need to redo the layers: the decoder is given to you as a function, `last_logits(tokens)`, built from exactly the pieces you filled in last time. Your job is the part that's new.

**What you'll do**

1. **Greedy decoding.** Each turn, turn the last row's logits into probabilities, pick the most likely word, and feed it back in. Repeat until `</s>` (or {MAX_TURNS} turns).
2. **The KV cache.** Push one new token through one self-attention head using cached keys and values, and check it gives the same row as recomputing everything.
3. **Beam search.** Keep the 2 best partial translations for two turns, and compute the length penalty.

**How it works:** run the three setup cells. Each step then has a cell full of `nan`: replace them with your numbers and run it. Wrong entries are listed with a hint; `reveal('name')` shows an answer and lets you carry on. Use Python as a calculator for single numbers (`np.exp(1.3)`, `np.log(0.4)`), but not for whole rows at once. Two or three decimals are enough.

| setting | here | paper |
|---|---|---|
| source sentence | "the cat sat" (n = 3) | any |
| d_model, heads, d_k | 4, 2, 2 | 512, 8, 64 |
| layers | 1 encoder + 1 decoder | 6 + 6 |
| vocabulary V | 11 | ≈ 37,000 |
| maximum output length | {MAX_TURNS} | n + 50 |
| beam size, length penalty α | 2, 0.6 | 4, 0.6 |

The weights are made-up stand-ins, not trained ones, so the "translation" won't be German. The loop is exactly the one a trained model runs.
""")

cells = [intro,
  md("## Setup\n\nRun these three cells in order. The first two are the same as in the encoder + decoder notebook."),
  code(HELPERS, tag="helpers"), code(DEC_HELPERS, tag="helpers-decoder"), code(INF_HELPERS, tag="helpers-inference"),
  md("""
---

# Part 1: greedy decoding

The encoder has already run, once. `Zf` holds its output, and it won't change again however many words the decoder writes.

The decoder starts from `<s>` alone. Each turn:

1. run the decoder on everything written so far and keep **only the last row**,
2. turn its 11 logits into probabilities with a softmax,
3. pick the word with the highest probability (**greedy**),
4. append it and go again, unless it was `</s>`.
""")]

for t in range(1, TURNS + 1):
    first = t == 1
    cells += [
      md(f"""
## Turn {t}

{"The decoder reads one token, `<s>`. Two logits are `-inf`: `<pad>` and `<s>` are never a correct next word, so they're ruled out, just like the masked scores in the last notebook (e**-inf = 0). " if first else f"The decoder reads `<s>` and the {t - 1} word{'s' if t > 2 else ''} you picked so far. "}Run the cell to see the last row's logits.{" Only the last row matters: every earlier row predicts a word you've already written." if t == 2 else ""}

**Shapes:** decoder input ({t},) tokens → decoder output ({t} × 4) → last row (1 × 4) → logits (11,).
"""),
      code(f"L{t} = show_logits({t})"),
      fill(f"p{t}", 11, comment=f"softmax of L{t}: e**L{t}[i] / (sum of all e**L{t}), for i = 0 .. 10"),
      fill(f"w{t}", 1, comment="the ID of the word greedy picks (the biggest probability)"),
      code(f"print('picked:', words[int(got('w{t}')[0])], '| so far:', tokens_for_turn({t + 1}))"),
    ]

cells += [
  md(f"""
## How did it end?

{"The model picked `</s>`, so the loop stopped." if REF_PICKS[-1] == "</s>" else f"`</s>` never came, so the loop stopped at the limit of {MAX_TURNS} turns. The paper would allow 3 + 50 = 53."} {"" if REF_PICKS[-1] == "</s>" else "Notice how it starts repeating itself. Greedy decoding loves to get stuck in loops, even in trained models, which is one reason for the length limit and for the alternatives below. "}With made-up weights the words are nonsense; a trained model would write "die katze saß </s>" through exactly these steps.

Notice that the encoder ran **once**, and the decoder ran **{TURNS} times**, each time on one more token.
"""),
  code("print('greedy translation:', ' '.join(tokens_for_turn(%d)[1:] + [words[int(got('w%d')[0])]]))" % (TURNS, TURNS)),

  md("""
---

# Part 2: the KV cache

At turn 2 the decoder recomputed the `<s>` row in every sublayer, and got exactly the same numbers as at turn 1: the mask means `<s>` can never see the new word. A **KV cache** keeps each layer's keys and values instead, so each turn only the **new token's row** is computed.

Let's do that for turn 2, in the first masked self-attention head. The cache after turn 1 holds one row each:
"""),
  code("k_cache = _RULES['k_cache'](); v_cache = _RULES['v_cache']()\n_ok['k_cache'] = k_cache; _ok['v_cache'] = v_cache\nprint('k_cache (1 × 2) =', np.round(k_cache, 4))\nprint('v_cache (1 × 2) =', np.round(v_cache, 4))\nprint('new token:', words[int(got('w1')[0])], 'at position 1')\nprint('W_mQ1 =\\n', W_mQ1, '\\nW_mK1 =\\n', W_mK1, '\\nW_mV1 =\\n', W_mV1)"),
  md("### The new token's input row\n\nEmbedding × √4 = 2, plus positional encoding for **position 1**. **Shape:** (1 × 4)."),
  fill("y0_new", 1, 4, "W_emb[new word] * 2 + PE[1]"),
  md("### Its query, key and value\n\nOne row each. **Shape:** (1 × 2)."),
  fill("q_new", 1, 2, "y0_new · W_mQ1"),
  fill("k_new", 1, 2, "y0_new · W_mK1"),
  fill("v_new", 1, 2, "y0_new · W_mV1"),
  md("### Append to the cache\n\nThe cache grows by one row. **Shape:** (2 × 2)."),
  fill("K_cache", 2, 2, "row 0 = k_cache, row 1 = k_new"),
  fill("V_cache", 2, 2, "row 0 = v_cache, row 1 = v_new"),
  md("### One row of attention\n\nOnly the new query is scored, against every cached key: (1 × 2) · (2 × 2)ᵀ → **(1 × 2)**. No mask: everything in the cache is in the past."),
  fill("s_new", 1, 2, "q_new · K_cacheᵀ / sqrt(2)"),
  fill("a_new", 1, 2, "softmax of s_new"),
  fill("h_new", 1, 2, "a_new · V_cache"),
  md("### Same answer, less work\n\nNow recompute the head the slow way, on both tokens at once with the mask, and compare its last row with yours."),
  code("""
toks = tokens_for_turn(2)
Y0f = _y0(toks)
S = (Y0f @ W_mQ1) @ (Y0f @ W_mK1).T / np.sqrt(d_k) + np.triu(np.full((2, 2), -np.inf), 1)
H_full = _sm(S) @ (Y0f @ W_mV1)
print('full recompute, both rows:\\n', np.round(H_full, 4))
print('your cached row          :', np.round(got('h_new'), 4))
print('match:', np.allclose(H_full[-1], got('h_new'), atol=0.03))
"""),
  md("""
Row 0 of the full recompute is the `<s>` row from turn 1, unchanged. With a cache you never compute it again. Over T turns that's T rows per layer instead of 1 + 2 + … + T.

---

# Part 3: beam search

Greedy keeps one candidate. Beam search keeps the **k best** (here k = 2; the paper uses 4), scoring each by the **sum of the natural logs** of its words' probabilities.

### Turn 1

Start from your `p1`. The two most likely first words become the two beams.
"""),
  code("print('p1 =', np.round(got('p1'), 4))\nprint(words)"),
  fill("beam1_ids", 2, comment="IDs of the 2 most likely first words, biggest first"),
  fill("beam1_scores", 2, comment="ln of their probabilities"),
  md("""
### Turn 2

Each beam is extended by every one of the 11 words: 2 × 11 = 22 candidates. A candidate's score is its beam's score plus ln P(the new word). Here are both beams' probabilities for the second word:
"""),
  code("""
P2b = np.vstack([_sm(last_logits(["<s>", words[int(i)]])[None, :])[0] for i in got("beam1_ids")])
for b, i in enumerate(got("beam1_ids")):
    print(f"beam {b}: <s> {words[int(i)]}")
    print("   P(next) =", np.round(P2b[b], 3))
"""),
  fill("cand", 2, 11, "cand[b, w] = beam1_scores[b] + ln P2b[b, w]; ln 0 = -inf"),
  fill("beam2", 2, 2, "the 2 best candidates as [beam row, word ID], best first"),
  code("for b, w in got('beam2').astype(int):\n    print('kept: <s>', words[int(got('beam1_ids')[b])], words[w], '| score', round(float(got('cand')[b, w]), 3))"),
  md("""
### The length penalty

Every extra word adds a negative log, so longer translations always score lower. Finished translations are compared after dividing by lp = ((5 + length) / 6)<sup>α</sup>, with α = 0.6 (§6.1, from Wu et al., 2016). Compute it for lengths 1, 2, 3 and 4.
"""),
  fill("lp", 4, comment="((5 + L) / 6) ** 0.6 for L = 1, 2, 3, 4"),
  md("""
## Want more?

- **Finish the beam.** Keep extending both beams until each ends in `</s>`, divide each total by its length penalty, and compare the winner with your greedy translation.
- **Sample instead.** Divide `L1` by a temperature of 2 before the softmax and see how the probabilities flatten. Then draw a word with `np.random.choice(V, p=...)`.
- **Make the weights your own.** Edit the decoder's weights in the second setup cell and rerun: the checker recomputes every answer.
- **Compare with the site.** Part 3 of Attention by Hand runs this loop live, with the KV cache and beam search animated.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:03d}"
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("cells:", len(cells), "| greedy picks:", REF_PICKS)
