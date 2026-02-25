from __future__ import annotations
from typing import Iterable, List, Tuple, Optional
import itertools

def count_params(hidden: List[int], input_dim: int = 1386, output_dim: int = 1) -> int:
    sizes = [input_dim] + hidden + [output_dim]
    return sum(sizes[i] * sizes[i+1] + sizes[i+1] for i in range(len(sizes) - 1))

def generate_hidden_layer_sizes(
    *,
    starts: Tuple[int, ...] = (4,),
    max_width: int = 64,
    min_width: int = 2,
    min_len: int = 4,
    max_len: int = 6,               # keep lists small by default
    dec_steps_range: Tuple[int, int] = (1, 3),
    # ---- repetition control ----
    max_total_repeats: int = 1,      # max count of "same-as-previous" steps in the whole sequence
    prefer_change: bool = True,      # try non-1 ratios first
) -> Iterable[List[int]]:
    """
    Rules:
      - first term in starts
      - next term is prev * r where r ∈ {1, 2, 1/2, 4, 1/4}
      - only two identical in a row (still enforced)
      - values within [min_width, max_width]
      - suffix has k strictly-decreasing steps (k in dec_steps_range),
        prefix before pivot is non-decreasing
      - repetition discouraged/limited via max_total_repeats
    """
    # Explore "change" first to avoid long runs of the same number.
    ratios_change_first = (2, 0.5, 4, 0.25, 1)
    ratios_normal = (1, 2, 0.5, 4, 0.25)
    ratios = ratios_change_first if prefer_change else ratios_normal

    def pivot_ok(seq: List[int]) -> bool:
        n = len(seq)
        for k in range(dec_steps_range[0], dec_steps_range[1] + 1):
            if n >= k + 2:
                # last k transitions strictly decreasing
                if not all(seq[i] > seq[i + 1] for i in range(n - k - 1, n - 1)):
                    continue
                # earlier transitions non-decreasing
                if all(seq[i] <= seq[i + 1] for i in range(0, n - k - 1)):
                    return True
        return False

    def dfs(prefix: List[int], run_same: int, total_same_steps: int):
        # emit if length ok and pivot condition satisfied
        if min_len <= len(prefix) <= max_len and pivot_ok(prefix):
            yield prefix

        if len(prefix) == max_len:
            return

        prev = prefix[-1]
        for r in ratios:
            nxt_float = prev * r
            nxt = int(nxt_float)
            if nxt_float != nxt:  # require exact integer step
                continue
            if not (min_width <= nxt <= max_width):
                continue

            # repetition controls
            if nxt == prev:
                if run_same >= 2:          # no 3 in a row
                    continue
                if total_same_steps >= max_total_repeats:
                    continue
                new_run_same = run_same + 1
                new_total_same = total_same_steps + 1
            else:
                new_run_same = 1
                new_total_same = total_same_steps

            yield from dfs(prefix + [nxt], new_run_same, new_total_same)

    for s in starts:
        yield from dfs([s], 1, 0)

# Example: filter into two parameter bands (adjust as needed)
band_3_4k = []
band_4_5k = []
band_5_6k = []
band_6_7k = []
band_7_75k = []
for seq in generate_hidden_layer_sizes(max_len=7, min_len=4, dec_steps_range=(1, 3)):
    p = count_params(seq)
    if 3000 <= p < 4000:
        band_3_4k.append((seq, p))
    elif 4000 <= p < 5000:
        band_4_5k.append((seq, p))
    elif 5000 <= p < 6000:
        band_5_6k.append((seq, p))
    elif 6000 <= p < 7000:
        band_6_7k.append((seq, p))
    elif 7000 <= p <= 7643:
        band_7_75k.append((seq, p))

print("3-4k-ish examples:", band_3_4k[:10])
print("4-5k-ish examples:", band_4_5k[:10])
# print("5-6k-ish examples:", band_5_6k[:10])
# print("6-7k-ish examples:", band_6_7k[:10])
# print("7-75k-ish examples:", band_7_75k[:10])