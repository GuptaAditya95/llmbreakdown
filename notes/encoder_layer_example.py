"""One Transformer encoder layer on 'the cat sat on the mat', at toy size.

Toy sizes (paper sizes in brackets): d_model=8 [512], h=2 heads [8],
d_k=d_v=4 [64], d_ff=32 [2048]. Weights are random (fixed seed), rounded to
one decimal so they are easy to read; in a real model they are learned.
Run: python3 encoder_layer_example.py
"""
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=140)
tokens = ["the", "cat", "sat", "on", "the", "mat"]

# X = embeddings + positional encoding, from positional-encoding-calculations.md
X = np.array([
    [ 0.10,  1.20, -0.30,  1.40,  0.00,  1.10, -0.20,  1.30],
    [ 1.14,  0.44,  0.60,  1.20, -0.39,  1.10,  0.60,  0.80],
    [ 0.71, -0.02,  0.30,  0.68,  0.22,  1.50, -0.10,  1.00],
    [ 0.14, -1.29,  0.50,  1.06,  0.33,  0.80,  0.40,  1.10],
    [-0.66, -0.45,  0.09,  1.32,  0.04,  1.10, -0.20,  1.30],
    [-0.56,  0.28,  0.38,  1.18, -0.15,  1.20,  0.10,  0.60],
])
n, d_model = X.shape
h, d_k, d_ff = 2, 4, 32

rng = np.random.default_rng(0)
def w(rows, cols):
    return np.round(rng.normal(0, 0.5, (rows, cols)), 1)

WQ = [w(d_model, d_k) for _ in range(h)]
WK = [w(d_model, d_k) for _ in range(h)]
WV = [w(d_model, d_k) for _ in range(h)]
WO = w(h * d_k, d_model)
W1 = np.round(rng.normal(0, 0.3, (d_model, d_ff)), 1)
W2 = np.round(rng.normal(0, 0.3, (d_ff, d_model)), 1)

def softmax_rows(S):
    e = np.exp(S - S.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def layer_norm(Z, eps=1e-6):
    # gamma = 1, beta = 0 (their starting values)
    mean = Z.mean(axis=1, keepdims=True)
    std = Z.std(axis=1, keepdims=True)
    return (Z - mean) / (std + eps), mean, std

def show(name, M):
    print(f"\n{name}  shape {M.shape}\n{M}")

show("X (input to the layer)", X)

heads = []
for i in range(h):
    print(f"\n========== Head {i+1} ==========")
    show(f"W_Q{i+1}", WQ[i]); show(f"W_K{i+1}", WK[i]); show(f"W_V{i+1}", WV[i])
    Q, K, V = X @ WQ[i], X @ WK[i], X @ WV[i]
    show(f"Q{i+1} = X·W_Q{i+1}", Q); show(f"K{i+1} = X·W_K{i+1}", K); show(f"V{i+1} = X·W_V{i+1}", V)
    S = Q @ K.T
    show(f"S{i+1} = Q{i+1}·K{i+1}ᵀ", S)
    S_scaled = S / np.sqrt(d_k)
    show(f"S{i+1} / sqrt({d_k})", S_scaled)
    A = softmax_rows(S_scaled)
    show(f"A{i+1} = softmax(rows)", A)
    head = A @ V
    show(f"head{i+1} = A{i+1}·V{i+1}", head)
    heads.append(head)

C = np.concatenate(heads, axis=1)
show("Concat(head1, head2)", C)
show("W_O", WO)
MHA = C @ WO
show("MHA = Concat·W_O", MHA)

R1 = X + MHA
show("X + MHA (residual)", R1)
X1, m1, s1 = layer_norm(R1)
show("row means", m1.ravel()); show("row std devs", s1.ravel())
show("X1 = LayerNorm(X + MHA)", X1)

show("W1", W1); show("W2", W2)
H_pre = X1 @ W1
show("X1·W1", H_pre)
H = np.maximum(0, H_pre)
show("H = ReLU(X1·W1)", H)
F = H @ W2
show("F = H·W2", F)

R2 = X1 + F
show("X1 + F (residual)", R2)
X_out, m2, s2 = layer_norm(R2)
show("X_out = LayerNorm(X1 + F)  <- output of encoder layer 1", X_out)
