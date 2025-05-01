# Note to potential future contributors:
# If you are not absolutely sure of what you are doing, please do not modify this file.

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional

from ..utils import Failure, RawAlignment



def constrained_viterbi(
    logits: Tensor,             # (T, U+1, vocab_size)
    target: Tensor,             # (U,) sequence of target token indices
    *,
    blank_idx: int = 0,         # index of the blank symbol in the vocabulary
    band_ratio: float = 0.5,    # relative width of Sakoe-Chiba band (0.0 - 1.0)
    lambda_diag: float = 0.0,   # weight of quadratic penalty off diagonal
) -> Union[RawAlignment, Failure]:
    """
    Perform a forced-alignment Viterbi decoding under Sakoe-Chiba banding
    with an optional quadratic penalty for drift from the main diagonal.

    This implementation processes the DP grid diagonally for parallelization.

    Args:
        logits (Tensor): FloatTensor of shape (T, U+1, V), raw network scores
                         where T is input length, U is target length, and V
                         is vocabulary size (including blank symbol).
        target (Tensor): LongTensor of shape (U,), the sequence of target token
                         indices (no SOS prepended).
        blank_idx (int): Index of the blank token in the vocabulary.
        band_ratio (float): Maximum allowed |u/U - t/T| for Sakoe-Chiba band.
        lambda_diag (float): Coefficient for quadratic off-diagonal penalty.

    Returns:
        path (List[Tuple[int, int, Optional[int]]]): List of (t, u, emit) steps
            representing the optimal alignment path. 'emit' is the token index 
            or None for blank emissions.
    """
    try:
        # Retrieve dimensions
        T, U1, V = logits.shape
        U = target.numel()
        assert U1 == U + 1, "Expected logits dim 1 to equal len(target)+1"

        device, dtype = logits.device, logits.dtype
        neg_inf = torch.finfo(dtype).min

        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Pre-compute Sakoe-Chiba band mask and quadratic penalty matrix
        t_idx = torch.arange(T + 1, device=device, dtype=dtype)
        u_idx = torch.arange(U + 1, device=device, dtype=dtype)
        # Normalized offset from diagonal: |u/U - t/T|
        offset = (u_idx[None] / max(1, U) - t_idx[:, None] / max(1, T)).abs()
        in_band = offset <= band_ratio        # Boolean mask for allowed cells
        penalty = lambda_diag * offset.pow(2) # Quadratic penalty for drift

        # Initialize DP score and back-pointer tables
        dp = torch.full((T + 1, U + 1), neg_inf, dtype=dtype, device=device)
        choice = torch.full((T + 1, U + 1), -1, dtype=torch.int8, device=device)
        dp[0, 0] = 0.0
        choice[0, 0] = 0  # origin

        # Diagonal-wise dynamic programming
        # Each diagonal s contains cells where t + u = s
        for s in range(T + U):
            t_min = max(0, s - U)
            t_max = min(T - 1, s)
            if t_min > t_max:
                continue

            # Current diagonal coordinates
            t_cur = torch.arange(t_min, t_max + 1, device=device)
            u_cur = s - t_cur
            current_scores = dp[t_cur, u_cur]

            # Transition: BLANK (t, u) -> (t+1, u)
            valid_blank = (t_cur + 1 <= T) & in_band[t_cur + 1, u_cur]
            if valid_blank.any():
                tb = t_cur[valid_blank]; ub = u_cur[valid_blank]
                dest_t = tb + 1; dest_u = ub
                blank_scores = (
                    current_scores[valid_blank]
                    + log_probs[tb, ub, blank_idx]
                    - penalty[dest_t, dest_u]
                )
                better = blank_scores > dp[dest_t, dest_u]
                if better.any():
                    idx = better.nonzero(as_tuple=True)[0]
                    # Update dp and choice in-place at better positions
                    dp.index_put_((dest_t[idx], dest_u[idx]), blank_scores[idx])
                    choice.index_put_((dest_t[idx], dest_u[idx]),
                                    torch.zeros_like(idx, dtype=torch.int8))

            # Transition: TOKEN (t, u) -> (t, u+1)
            # First mask out-of-range indices, then apply band mask
            mask_token = (u_cur + 1 <= U)
            if mask_token.any():
                # Check band constraint at destination (t, u+1)
                inband_tok = in_band[t_cur[mask_token], u_cur[mask_token] + 1]
                valid_token = torch.zeros_like(mask_token)
                valid_token[mask_token] = inband_tok
            else:
                valid_token = mask_token

            if valid_token.any():
                tt = t_cur[valid_token]; ut = u_cur[valid_token]
                dest_t = tt; dest_u = ut + 1
                tok_ids = target[ut]
                token_scores = (
                    current_scores[valid_token]
                    + log_probs[tt, ut, tok_ids]
                    - penalty[dest_t, dest_u]
                )
                better_tok = token_scores > dp[dest_t, dest_u]
                if better_tok.any():
                    idx = better_tok.nonzero(as_tuple=True)[0]
                    dp.index_put_((dest_t[idx], dest_u[idx]), token_scores[idx])
                    choice.index_put_((dest_t[idx], dest_u[idx]),
                                    torch.ones_like(idx, dtype=torch.int8))

        # Final best log-probability
        best_logp = dp[T, U]
        if best_logp == neg_inf:
            raise RuntimeError("No path found within band. Consider increasing band_ratio.")

        # Backtrace to recover alignment path
        path: List[Tuple[int, int, Optional[int]]] = []
        t, u = T, U
        while (t, u) != (0, 0):
            c = int(choice[t, u].item())
            if c == 0:
                prev_t, prev_u = t - 1, u
                emit = None
            elif c == 1:
                prev_t, prev_u = t, u - 1
                emit = int(target[prev_u])
            else:
                raise RuntimeError("Invalid back-pointer detected during traceback.")
            path.append((prev_t, prev_u, emit))
            t, u = prev_t, prev_u

        path.reverse()
        return path

    except Exception as e:
        return Failure(f"Error in Constrained Viterbi: {e}")