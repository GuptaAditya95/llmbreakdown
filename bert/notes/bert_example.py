"""BERT's forward pass at toy size, from a sentence pair to the pre-training loss.

Example: Figure 2 of the paper, "my dog is cute" / "he likes playing".
Toy sizes (paper BERT-base in brackets): H = d_model = 8 [768], A = 2 heads [12],
d_k = 4 [64], d_ff = 32 [3072], one encoder layer shown in full [12 layers].
Weights and embedding tables are random (fixed seed), rounded to one decimal so
they are easy to read; in a real model they are learned. Biases inside the
encoder are left at zero (their starting value) to keep the sums short.

Details marked [code] come from the released implementation
(github.com/google-research/bert, modeling.py and run_pretraining.py),
not from the paper.

Run: python3 bert_example.py
"""
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=150)

# ---------------------------------------------------------------- sizes
H, A, d_ff = 8, 2, 32
d_k = H // A

# ---------------------------------------------------------------- input (§3, Figure 2)
sent_a = ["my", "dog", "is", "cute"]
sent_b = ["he", "likes", "play", "##ing"]          # WordPiece splits "playing"
tokens = ["[CLS]"] + sent_a + ["[SEP]"] + sent_b + ["[SEP]"]
segment = [0] * (len(sent_a) + 2) + [1] * (len(sent_b) + 1)
n = len(tokens)

# Masked LM input (§3.1, Appendix A.1): 15% of the positions, rounded, at least one [code].
# Position 4 "cute" -> [MASK] (the 80% case); position 7 "likes" -> "apple" (the 10% random case).
num_to_predict = max(1, round(n * 0.15))
masked = {4: "[MASK]", 7: "apple"}
assert len(masked) == num_to_predict
inputs = [masked.get(i, t) for i, t in enumerate(tokens)]
targets = {i: tokens[i] for i in masked}

# ---------------------------------------------------------------- a tiny vocabulary
# Special-token IDs follow the released vocab.txt [code]; the rest are illustrative.
vocab = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "my", "dog", "is", "cute",
         "he", "likes", "play", "##ing", "apple", "the", "a", "hairy"]
ID = {t: k for k, t in enumerate(vocab)}
V = len(vocab)

rng = np.random.default_rng(0)
def w(rows, cols, s=0.5):
    return np.round(rng.normal(0, s, (rows, cols)), 1)

E_tok = w(V, H, 0.4)            # token embedding table   (V × H)      [30,000 × 768]
E_seg = w(2, H, 0.4)            # segment embedding table (2 × H)      [2 × 768]
E_pos = w(n, H, 0.4)            # position embedding table rows 0..n-1 [512 × 768, learned]

WQ = [w(H, d_k) for _ in range(A)]
WK = [w(H, d_k) for _ in range(A)]
WV = [w(H, d_k) for _ in range(A)]
WO = w(H, H, 0.35)
W1 = w(H, d_ff, 0.35)
W2 = w(d_ff, H, 0.2)

# heads
W_t = w(H, H, 0.35)              # MLM transform dense [code]
W_p = w(H, H, 0.35)              # pooler dense [code]
W_nsp = w(2, H, 0.5)             # NSP classifier, 2 × H [code]


def softmax_rows(S):
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def layer_norm(Z, eps=1e-12):   # gamma = 1, beta = 0 (their starting values)
    mu = Z.mean(axis=-1, keepdims=True)
    sd = np.sqrt(Z.var(axis=-1, keepdims=True) + eps)
    return (Z - mu) / sd, mu.ravel(), sd.ravel()

def gelu(x):                    # the tanh approximation used in the released code [code]
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def encoder_layer(X):
    heads = []
    for h in range(A):
        Q, K, Vv = X @ WQ[h], X @ WK[h], X @ WV[h]
        S = Q @ K.T                       # no mask: every token sees every token
        Aw = softmax_rows(S / np.sqrt(d_k))
        heads.append(dict(Q=Q, K=K, V=Vv, S=S, A=Aw, O=Aw @ Vv))
    C = np.concatenate([hd["O"] for hd in heads], axis=1)
    M = C @ WO
    X1, mu1, sd1 = layer_norm(X + M)
    P = X1 @ W1
    G = gelu(P)                            # GELU instead of ReLU (Appendix A.2)
    F = G @ W2
    out, mu2, sd2 = layer_norm(X1 + F)
    return dict(heads=heads, C=C, M=M, R1=X + M, X1=X1, mu1=mu1, sd1=sd1, P=P, G=G, F=F,
                R2=X1 + F, out=out, mu2=mu2, sd2=sd2)


def forward(inputs):
    ids = np.array([ID[t] for t in inputs])
    Xt, Xs, Xp = E_tok[ids], E_seg[segment], E_pos[np.arange(n)]
    Xsum = Xt + Xs + Xp                    # §3: sum of token, segment and position embeddings
    X0, mu0, sd0 = layer_norm(Xsum)        # LayerNorm on the sum [code]
    L = encoder_layer(X0)
    return dict(ids=ids, Xt=Xt, Xs=Xs, Xp=Xp, Xsum=Xsum, X0=X0, mu0=mu0, sd0=sd0, L=L, T=L["out"])


def mlm_head(T, positions):
    Tm = T[positions]                      # only the masked positions (§3.1)
    U = gelu(Tm @ W_t)                     # dense + GELU [code]
    Ut, _, _ = layer_norm(U)               # LayerNorm [code]
    logits = Ut @ E_tok.T                  # tied to the input embedding table, output bias 0 [code]
    return Tm, U, Ut, logits, softmax_rows(logits)


def nsp_head(T):
    C = T[0]                               # final hidden vector of [CLS] (§3)
    pooled = np.tanh(C @ W_p)              # pooler: dense + tanh [code]
    logits = W_nsp @ pooled                # 2-way classifier (IsNext, NotNext) [code]
    return C, pooled, logits, softmax_rows(logits)


if __name__ == "__main__":
    f = forward(inputs)
    L = f["L"]
    print("tokens  ", tokens)
    print("inputs  ", inputs, " (after masking)")
    print("segment ", segment)
    print("ids     ", f["ids"])
    print("\nE_tok rows (n × H)\n", f["Xt"])
    print("\nE_seg rows\n", f["Xs"])
    print("\nE_pos rows\n", f["Xp"])
    print("\nsum\n", f["Xsum"])
    print("\nX0 = LayerNorm(sum)\n", f["X0"])
    for h, hd in enumerate(L["heads"]):
        print(f"\n--- head {h + 1}")
        for k in "QKV":
            print(f"{k}\n", hd[k])
        print("S = Q·Kᵀ\n", hd["S"])
        print("A = softmax(S / √d_k)\n", hd["A"])
        print("A·V\n", hd["O"])
    print("\nMHA = concat · W_O\n", L["M"])
    print("\nX1 = LayerNorm(X0 + MHA)\n", L["X1"])
    print("\nP = X1 · W1 (n × d_ff), first 8 columns\n", L["P"][:, :8])
    print("\nGELU(P), first 8 columns\n", L["G"][:, :8])
    print("\nF = GELU(P) · W2\n", L["F"])
    print("\nT = LayerNorm(X1 + F)  (one layer shown; BERT-base stacks 12)\n", L["out"])

    pos = sorted(masked)
    Tm, U, Ut, logits, P = mlm_head(f["T"], pos)
    print("\n=== Masked LM head, positions", pos)
    print("T at masked positions\n", Tm)
    print("dense + GELU\n", U)
    print("LayerNorm\n", Ut)
    print("logits over the vocabulary\n", logits)
    mlm_losses = [-np.log(P[r, ID[targets[p]]]) for r, p in enumerate(pos)]
    for r, p in enumerate(pos):
        top = vocab[int(np.argmax(P[r]))]
        print(f"pos {p}: target {targets[p]!r}  P = {P[r, ID[targets[p]]]:.3f}  loss = {mlm_losses[r]:.3f}  top guess {top!r}")
    mlm_loss = float(np.mean(mlm_losses))

    C, pooled, nl, pn = nsp_head(f["T"])
    label = 0                                       # IsNext: B really follows A
    nsp_loss = float(-np.log(pn[label]))
    print("\n=== NSP head")
    print("C\n", C, "\npooled = tanh(C · W_p)\n", pooled, "\nlogits", nl, " probs", pn)
    print(f"NSP loss = -ln P(IsNext) = {nsp_loss:.3f}")
    print(f"\nTOTAL pre-training loss = MLM {mlm_loss:.3f} + NSP {nsp_loss:.3f} = {mlm_loss + nsp_loss:.3f}  (Appendix A.2)")

    # ---- inference, fill-mask: no loss, no NSP, no random replacement
    inf_inputs = ["[MASK]" if i == 4 else t for i, t in enumerate(tokens)]
    fi = forward(inf_inputs)
    _, _, _, _, Pi = mlm_head(fi["T"], [4])
    order = np.argsort(-Pi[0])[:3]
    print("\n=== Inference: fill-mask on", " ".join(inf_inputs))
    print("top 3 for position 4:", [(vocab[k], round(float(Pi[0, k]), 3)) for k in order])

    # ---- parameter count at the paper's size (BERT-base)
    Hp, Lp, Vp, Pp, ffp = 768, 12, 30522, 512, 3072
    emb = Vp * Hp + Pp * Hp + 2 * Hp + 2 * Hp
    per_layer = 4 * (Hp * Hp + Hp) + (Hp * ffp + ffp) + (ffp * Hp + Hp) + 2 * 2 * Hp
    pooler = Hp * Hp + Hp
    print(f"\nBERT-base parameters: embeddings {emb:,} + {Lp} layers × {per_layer:,} + pooler {pooler:,}"
          f" = {emb + Lp * per_layer + pooler:,}  (paper: 110M)")
