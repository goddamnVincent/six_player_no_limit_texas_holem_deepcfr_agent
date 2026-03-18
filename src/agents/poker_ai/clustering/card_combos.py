from __future__ import annotations
import concurrent.futures
import logging
import operator
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from poker_ai.poker.card import Card
from poker_ai.poker.deck import get_all_suits

__all__ = [
    "CardCombosMP",  # main public class
    "load_street_combos",  # helper for CardInfoLutBuilder
    "CardCombos",  # backwards‑compat alias
          ]
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("poker_ai.clustering.card_combos_mp")

# ---------------------------------------------------------------------------
# helper utils (load / save)
# ---------------------------------------------------------------------------

_STREET2FILE = {
    "starting_hands": "starting_hands.npy",
    "flop": "flop.npy",
    "turn": "turn.npy",
    "river": "river.npy",
}

def _save(path: Path, arr: np.ndarray) -> None:
    """Save *object* ndarray with pickle allowed (required for Card objects)."""
    np.save(path, arr, allow_pickle=True)


def _load(path: Path) -> np.ndarray:
    """Load ndarray of `Card` objects (pickle needed)."""
    return np.load(path, allow_pickle=True)


def load_street_combos(cache_dir: os.PathLike) -> Dict[str, np.ndarray]:
    """Utility for `CardInfoLutBuilder`.

    Parameters
    ----------
    cache_dir : path‑like
        Directory where ``CardCombosMP`` previously dumped the ``*.npy`` files.

    Returns
    -------
    dict
        ``{"starting_hands": ndarray, "flop": ndarray, "turn": ndarray, "river": ndarray}``
    """
    root = Path(cache_dir).expanduser().resolve()
    combos: Dict[str, np.ndarray] = {}
    for street, fname in _STREET2FILE.items():
        fpath = root / fname
        if not fpath.exists():
            raise FileNotFoundError(f"{street} file not found: {fpath}")
        combos[street] = _load(fpath)
    return combos

# ---------------------------------------------------------------------------
# worker executed in child process
# ---------------------------------------------------------------------------

def _process_start_combo(args: Tuple[np.ndarray, np.ndarray]):
    """Enumerate legal (hole + board) combos for **one** starting hand.

    Parameters
    ----------
    args
        *(start_combo, publics)* where ``start_combo`` is a tuple of 2 `Card`
        and ``publics`` is a *shared* ndarray of shape (n_public, k).

    Returns
    -------
    list[np.ndarray]
        Every legal arrangement (Card[7]) for this starting hand.
    """
    start_combo, publics = args
    sorted_private = sorted(start_combo, key=operator.attrgetter("eval_card"), reverse=True)
    legal = []
    for public_combo in publics:
        # 'public_combo' is already a tuple of Card; sorting once keeps order consistent
        sorted_public = sorted(public_combo, key=operator.attrgetter("eval_card"), reverse=True)
        # overlap test
        if not np.any(np.isin(sorted_private, sorted_public)):
            legal.append(np.array(sorted_private + sorted_public))
    return legal

# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

class CardCombosMP:
    """Parallel enumerator *with* on‑disk caching.

    Parameters
    ----------
    low_card_rank, high_card_rank
        Poker rank range (typically ``2, 14``).
    cache_dir : str | os.PathLike | None, optional
        Where to dump the ``*.npy`` files.  If *None*, combos stay only in
        memory.  When given, directory is created if missing.
    keep_in_memory : bool, default ``False``
        After dumping, whether to retain the big ndarrays in RAM.  Setting to
        ``False`` allows subsequent GC to free >100 GB during river phase.
    """

    def __init__(
        self,
        low_card_rank: int,   # 如果是经典的52张牌的 就是 2 - 14
        high_card_rank: int,
        *,
        cache_dir: os.PathLike | None = None,
        keep_in_memory: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        print(f'target path:{self.cache_dir}')
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            log.info("[CardCombosMP] dumping combos to %s", self.cache_dir)
        self.keep_in_memory = keep_in_memory

        # Build full deck
        suits: List[str] = sorted(get_all_suits())
        ranks: List[int] = list(range(low_card_rank, high_card_rank + 1))
        self._cards = np.array([Card(rank, suit) for suit in suits for rank in ranks])

        # ⬇︎ Each step: generate public‑card pool once, then fan out via mp
        self.starting_hands = self._get_card_combos(2)
        # 看情况打开
        # self._maybe_dump("starting_hands", self.starting_hands)
        #
        #
        # flop_publics = self._get_card_combos(3)
        # self.flop = self._create_info_combos_mp("flop", flop_publics)
        # self._maybe_dump("flop", self.flop)
        # if not self.keep_in_memory:
        #     self._release_memory("flop")
        #
        # turn_publics = self._get_card_combos(4)
        # self.turn = self._create_info_combos_mp("turn", turn_publics)
        # self._maybe_dump("turn", self.turn)
        # if not self.keep_in_memory:
        #     self._release_memory("turn")

        river_publics = self._get_card_combos(5)
        self.river = self._create_info_combos_mp("river", river_publics)
        self._maybe_dump("river", self.river)
        # if not keeping in mem, wipe heavy attrs now.
        if not self.keep_in_memory:
            self._release_memory("river")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _release_memory(self, attr: str):
        """Release memory for the given attribute."""
        if hasattr(self, attr):
            delattr(self, attr)  # 删除对象属性
        import gc
        gc.collect()  # 手动触发垃圾回收
        log.info(f"[CardCombosMP] Released {attr} from memory")

    def _maybe_dump(self, street: str, arr: np.ndarray) -> None:
        if not self.cache_dir:
            return
        fpath = self.cache_dir / _STREET2FILE[street]
        _save(fpath, arr)
        log.info("[CardCombosMP] %s saved → %s (%.2f MB)", street, fpath, arr.nbytes / 1e6)

    # ------------------------------------------------------------
    # pure‑math helpers
    # ------------------------------------------------------------

    def _get_card_combos(self, n: int) -> np.ndarray:  #生成所有可能的牌面组合
        """C(52, n) unordered combos of *Card*."""
        return np.array(list(combinations(self._cards, n)))

    def _create_info_combos_mp(self, tag: str, publics: np.ndarray, n_workers: int = None) -> np.ndarray:
        if n_workers is None:
            n_workers = os.cpu_count() - 1
        log.info(f'CPU count: {os.cpu_count()}, using {n_workers} workers')

        chunksize = max(1, len(self.starting_hands) // (cpu_cnt * 4))
        with concurrent.futures.ProcessPoolExecutor() as exe:
            iterator = ((sh, publics) for sh in self.starting_hands)
            results = list(
                tqdm(
                    exe.map(_process_start_combo, iterator, chunksize=chunksize),
                    total=len(self.starting_hands),
                    desc=f"{tag.capitalize()} combos (mp)",
                    dynamic_ncols=True,
                )
            )
        flat = [hand for sub in results for hand in sub]
        arr = np.asarray(flat, dtype=object)
        log.info("[CardCombosMP] created %s — %d combos", tag, len(arr))
        return arr

# ---------------------------------------------------------------------------
# backward‑compat: expose old class name so legacy imports work
# ---------------------------------------------------------------------------
CardCombos = CardCombosMP

# ---------------------------------------------------------------------------
# *example* of how CardInfoLutBuilder can consume on‑disk combos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample CLI usage: only build + dump to ./combo_cache and exit
    import argparse, time

    parser = argparse.ArgumentParser(description="Build & persist street combos")
    parser.add_argument("--out", type=Path, default="/home/jml/poker_ai/research/all_results", help="Output directory")
    parser.add_argument("--keep", action="store_true", help="Keep arrays in memory (debug)")
    args = parser.parse_args()

    t0 = time.time()
    CardCombosMP(2, 14, cache_dir=args.out, keep_in_memory=args.keep)
    log.info("Done in %.1f s", time.time() - t0)

