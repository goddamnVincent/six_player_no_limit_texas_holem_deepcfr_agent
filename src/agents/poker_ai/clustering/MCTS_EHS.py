import numpy as np
from typing import Sequence
import time
from ..poker.card import Card, get_all_suits
from ..poker.evaluation import Evaluator

def card_to_short(c: Card) -> str:
    rank_map = {
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "T",
        11: "J",
        12: "Q",
        13: "K",
        14: "A",
    }
    suit_map = {
        "hearts": "h",
        "diamonds": "d",
        "clubs": "c",
        "spades": "s",
    }
    return rank_map[c.rank_int] + suit_map[c.suit]

def cards_to_str(cards: Sequence[Card]) -> str:
    if len(cards) == 0:
        return "[]"
    return " ".join(card_to_short(c) for c in cards)

def build_full_deck(
    low_rank: int = 2,
    high_rank: int = 14,
) -> np.ndarray:
    """
    按照你当前的 Card 类构造一整副牌（52 张）。

    返回
    ----
    full_deck : np.ndarray[Card], dtype=object, shape (n_cards,)
    """
    suits = sorted(get_all_suits())  # {'clubs','spades','hearts','diamonds'}
    ranks = list(range(low_rank, high_rank + 1))
    cards = [Card(rank, suit) for suit in suits for rank in ranks]
    return np.array(cards, dtype=object)


def monte_carlo_ehs_multi(
    our_hand: Sequence[Card],
    board: Sequence[Card],
    n_simulations: int = 10_000,
    n_opponents: int = 5,      # 六人桌 = 5 个对手
    low_rank: int = 2,
    high_rank: int = 14,
) -> np.ndarray:
    """
    多人桌 Monte Carlo EHS：Hero vs n_opponents 个随机对手。

    返回 [win_rate, lose_rate, tie_rate]，
    win 表示 Hero 在这条随机牌路里击败所有对手的概率。
    """
    evaluator = Evaluator()

    our_hand = np.asarray(our_hand, dtype=object)
    board = np.asarray(board, dtype=object)

    full_deck = build_full_deck(low_rank, high_rank)

    if board.size > 0:
        used = np.concatenate([our_hand, board])
    else:
        used = our_hand
    used_set = set(used.tolist())
    available_cards = np.array([c for c in full_deck if c not in used_set], dtype=object)

    n_board_now = len(board)
    assert 0 <= n_board_now <= 5, f"board 长度不合法: {n_board_now}"
    n_board_missing = 5 - n_board_now

    ehs = np.zeros(3, dtype=np.float64)

    for _ in range(n_simulations):
        # 需要的牌数：所有对手 2*n_opponents + 未来公共牌 n_board_missing
        total_needed = 2 * n_opponents + n_board_missing
        sample = np.random.choice(available_cards, total_needed, replace=False)

        opp_cards = sample[: 2 * n_opponents]
        future_board_cards = sample[2 * n_opponents :]

        if n_board_now > 0:
            full_board = np.concatenate([board, future_board_cards])
        else:
            full_board = future_board_cards

        board_int = [int(c) for c in full_board]
        our_int = [int(c) for c in our_hand]

        our_rank = evaluator.evaluate(board=board_int, cards=our_int)

        # 逐个对手算牌力
        opp_ranks = []
        for i in range(n_opponents):
            hand_i = opp_cards[2 * i : 2 * i + 2]
            opp_int = [int(c) for c in hand_i]
            r_i = evaluator.evaluate(board=board_int, cards=opp_int)
            opp_ranks.append(r_i)

        # 注意：rank 越小越好
        best_opp = min(opp_ranks)

        if our_rank < best_opp:
            ehs[0] += 1.0  # 我比所有对手都好
        elif our_rank > best_opp:
            ehs[1] += 1.0  # 至少有一个人比我好
        else:
            ehs[2] += 1.0  # 极端少见：我和最佳对手完全平手

    ehs /= float(n_simulations)
    return ehs

def _pretty_print_ehs(ehs: np.ndarray):
    win, lose, tie = ehs.tolist()
    print(f"win={win:.4f}, lose={lose:.4f}, tie={tie:.4f}")


def main():
    n_sims = 1000
    n_opps = 5  # 六人桌 = 5 个对手

    # 示例 1：preflop AA，对抗随机手牌
    our_hand = np.array([
        Card(13, "hearts"),   # Ah
        Card(14, "clubs"),    # Ac
    ], dtype=object)
    board = np.array([], dtype=object)  # preflop 没有公共牌

    hand_str = cards_to_str(our_hand)
    board_str = cards_to_str(board)

    print(f"=== Preflop: hand={hand_str}, board={board_str}, "
          f"vs random range (opps={n_opps}, sims={n_sims}) ===")
    t0 = time.time()
    ehs_pre = monte_carlo_ehs_multi(our_hand, board,
                                    n_simulations=n_sims,
                                    n_opponents=n_opps)
    t1 = time.time()
    _pretty_print_ehs(ehs_pre)
    print(f"n_simulations = {n_sims}, time = {t1 - t0:.3f}s")

    # 示例 2：flop 场景
    # "spades", "diamonds", "clubs", "hearts"
    board_flop = np.array([
        Card(14, "spades"),
        Card(12, "diamonds"),
        Card(13, "clubs"),
    ], dtype=object)

    board_flop_str = cards_to_str(board_flop)

    print(f"\n=== Flop: hand={hand_str}, board={board_flop_str}, "
          f"vs random range (opps={n_opps}, sims={n_sims}) ===")
    t2 = time.time()
    ehs_flop = monte_carlo_ehs_multi(our_hand, board_flop,
                                     n_simulations=n_sims,
                                     n_opponents=n_opps)
    t3 = time.time()
    _pretty_print_ehs(ehs_flop)
    print(f"n_simulations = {n_sims}, time = {t3 - t2:.3f}s")

    print(f"\nTotal time = {t3 - t0:.3f}s")


if __name__ == "__main__":
    main()