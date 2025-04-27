import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional



def constrained_viterbi(
    logits: torch.Tensor,	# (T, U+1, vocab_size)
    target: torch.Tensor,	# (U,) - sans SOS
    *,
    blank_idx: int = 0,
    band_ratio: float = 0.5,	# épaisseur du couloir (0‑0.5)
    lambda_diag: float = 0.0	# poids de la pénalité quad.
) -> Tuple[List[Tuple[int, int, Optional[int]]], torch.Tensor]:
    """
    Alignement forcé avec :
      - couloir Sakoe–Chiba : |u/U - t/T| ≤ band_ratio
      - pénalité quadratique λ·(écart)² vers la diagonale.
    """

    # -------- dimensions --------
    T, U1, V = logits.shape     # U1 = U + 1
    U = target.size(0)
    assert U1 == U + 1, "logits dim-2 must equal len(target)+1"

    log_probs = F.log_softmax(logits, dim=-1)

    # -------- fonctions utilitaires --------
    def off_diag(t: int, u: int) -> float:
        """Écart normalisé à la diagonale."""
        return abs(u / U - t / T)

    def quad_penalty(t: int, u: int) -> float:
        return lambda_diag * (off_diag(t, u) ** 2)

    # -------- tableaux DP --------
    neg_inf = -float("inf")
    dp   = torch.full((T + 1, U + 1), neg_inf,
                      dtype = log_probs.dtype,
                      device = log_probs.device)
    back = [[None]*(U + 1) for _ in range(T + 1)]
    dp[0, 0] = 0.0

    # -------- propagation --------
    for t in range(T + 1):		# 0 … T
        for u in range(U + 1):	# 0 … U
            cur = dp[t, u]
            if cur == neg_inf:
                continue

            # --- (a) BLANK : (t,u) → (t+1,u)
            if t < T and off_diag(t+1, u) <= band_ratio:
                score = (cur +
                         log_probs[t, u, blank_idx] -
                         quad_penalty(t+1, u))
                if score > dp[t+1, u]:
                    dp[t+1, u]   = score
                    back[t+1][u] = (t, u, None)	# None = blank

            # --- (b) TOKEN : (t,u) → (t,u+1)
            if (t < T) and (u < U) and off_diag(t, u+1) <= band_ratio:
                tok   = target[u].item()
                score = (cur +
                         log_probs[t, u, tok] -
                         quad_penalty(t, u+1))
                if score > dp[t, u+1]:
                    dp[t, u+1]   = score
                    back[t][u+1] = (t, u, tok)

    best_logp = dp[T, U]

    # -------- rétro‑traçage --------
    path: List[Tuple[int, int, Optional[int]]] = []
    t, u = T, U
    while not (t == 0 and u == 0):
        prev_t, prev_u, emitted = back[t][u]
        path.append((prev_t, prev_u, emitted))
        t, u = prev_t, prev_u
    path.reverse()

    return path, best_logp