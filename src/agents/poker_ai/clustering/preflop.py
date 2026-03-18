from typing import Dict, Tuple, List
import operator
import math

from poker_ai.poker.card import Card


def make_starting_hand_lossless(starting_hand, short_deck) -> int:
    """"""
    ranks = []
    suits = []
    for card in starting_hand:
        ranks.append(card.rank_int)
        suits.append(card.suit)
    if len(set(suits)) == 1:
        suited = True
    else:
        suited = False
    if all(c_rank == 14 for c_rank in ranks):
        return 0
    elif all(c_rank == 13 for c_rank in ranks):
        return 1
    elif all(c_rank == 12 for c_rank in ranks):
        return 2
    elif all(c_rank == 11 for c_rank in ranks):
        return 3
    elif all(c_rank == 10 for c_rank in ranks):
        return 4
    elif 14 in ranks and 13 in ranks:
        return 5 if suited else 15
    elif 14 in ranks and 12 in ranks:
        return 6 if suited else 16
    elif 14 in ranks and 11 in ranks:
        return 7 if suited else 17
    elif 14 in ranks and 10 in ranks:
        return 8 if suited else 18
    elif 13 in ranks and 12 in ranks:
        return 9 if suited else 19
    elif 13 in ranks and 11 in ranks:
        return 10 if suited else 20
    elif 13 in ranks and 10 in ranks:
        return 11 if suited else 21
    elif 12 in ranks and 11 in ranks:
        return 12 if suited else 22
    elif 12 in ranks and 10 in ranks:
        return 13 if suited else 23
    elif 11 in ranks and 10 in ranks:
        return 14 if suited else 24


def compute_preflop_lossless_abstraction(builder) -> Dict[Tuple[Card, Card], int]:
    """Compute the preflop abstraction dictionary.

    Only works for the short deck presently.
    """
    # Making sure this is 20 card deck with 2-9 removed
    allowed_ranks = {10, 11, 12, 13, 14}
    found_ranks = set([c.rank_int for c in builder._cards])
    if found_ranks != allowed_ranks:
        raise ValueError(
            f"Preflop lossless abstraction only works for a short deck with "
            f"ranks [10, jack, queen, king, ace]. What was specified="
            f"{found_ranks} doesn't equal what is allowed={allowed_ranks}"
        )
    # Getting combos and indexing with lossless abstraction
    preflop_lossless: Dict[Tuple[Card, Card], int] = {}
    for starting_hand in builder.starting_hands:
        starting_hand = sorted(
            list(starting_hand),
            key=operator.attrgetter("eval_card"),
            reverse=True
        )
        preflop_lossless[tuple(starting_hand)] = make_starting_hand_lossless(
            starting_hand, builder
        )
    return preflop_lossless

def make_starting_hand_lossless_2_to_A(starting_hand, builder) -> int:
    ranks = [card.rank_int for card in starting_hand]
    suits = [card.suit for card in starting_hand]
    suited = len(set(suits)) == 1

    high_rank = max(ranks)
    low_rank = min(ranks)

    if high_rank == low_rank:
        return 14 - high_rank

    combinations = []
    for hr in range(14, 1, -1):
        for lr in range(hr - 1, 1, -1):  # 注意这里改成1，排除1的rank
            combinations.append((hr, lr))

    idx = combinations.index((high_rank, low_rank))

    if suited:
        return 13 + idx
    else:
        return 13 + 78 + idx

def compute_preflop_lossless_abstraction_2_to_A(builder) -> Dict[Tuple[Card, Card], int]:
    print('using it')
    """Compute the preflop abstraction dictionary for full Texas Hold'em (2-A)."""
    # 确保使用完整的 2-A 扑克牌
    allowed_ranks = set(range(2, 15))  # 2-14
    found_ranks = set([c.rank_int for c in builder._cards])
    if not allowed_ranks.issubset(found_ranks):
        raise ValueError(
            f"Preflop lossless abstraction only works for a full deck with "
            f"ranks [2-14]. What was specified={found_ranks} doesn't contain all required ranks."
        )

    preflop_lossless: Dict[Tuple[Card, Card], int] = {}
    for starting_hand in builder.starting_hands:
        starting_hand = sorted(
            list(starting_hand),
            key=operator.attrgetter("eval_card"),
            reverse=True
        )
        preflop_lossless[tuple(starting_hand)] = make_starting_hand_lossless(
            starting_hand, builder
        )
    return preflop_lossless

# card1 = Card(rank=14, suit="spades")  # A♠️
# card2 = Card(rank=2, suit="spades")
# starting_hand = (card1, card2)
# print(make_starting_hand_lossless_2_to_A(starting_hand, None))





