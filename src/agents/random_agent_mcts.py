import time
from src.agents.poker_ai.clustering.MCTS_EHS import monte_carlo_ehs_multi, card_to_short, cards_to_str, build_full_deck, _pretty_print_ehs
from src.agents.poker_ai.poker.card import Card as EvalCard
import numpy as np
from typing import Sequence
import time
from src.agents.poker_ai.poker.card import Card,get_all_suits
import random
from typing import Optional, List, Set
import pokers as pkrs
from src.utils.logging import log_game_error
import logging
logger = logging.getLogger(__name__)
import copy
# 1000 次 mcts 六人桌 每次大概在 0.09s左右
# 2000 次 mcts 六人桌 每次大概在 0.18s左右

def get_mcts_result(player_id, hand_card, board_card, n_simulations:int = 1000, n_opponents:int = 2, verbose: bool = False):
    mcts_result = monte_carlo_ehs_multi(hand_card, board_card, n_simulations, n_opponents)
    if verbose == True:
        logging.info(f'player_id:{player_id}, hand: {cards_to_str(hand_card)}, board: {cards_to_str(board_card)}, n_simulations: {n_simulations}, n_opponents: {n_opponents}')
    return mcts_result

def convert_cards_to_eval(cards):
    """
    把 pokers 环境里的牌（rank:0..12, suit:0..3）
    转成 poker_ai.poker.card.Card(rank:2..14, suit:英文字符串)。
    """
    if not cards:
        return np.array([], dtype=object)

    suit_map = {
        0: "clubs",    # C 梅花
        1: "diamonds", # D 方片
        2: "hearts",   # H 红桃
        3: "spades",   # S 黑桃
    }
    out = []
    for c in cards:
        r_env = int(c.rank)   # 0..12 -> 2..14
        s_env = int(c.suit)   # 0..3  -> 文本
        rank = r_env + 2
        suit = suit_map[s_env]
        out.append(EvalCard(rank, suit))
    return np.array(out, dtype=object)

stage_names = {
    0: "PreFlop",
    1: "Flop",
    2: "Turn",
    3: "River",
    4: "Showdown"
}

# "spades", "diamonds", "clubs", "hearts"
# 已知AA 胜率最高 23 24 25 26 27胜率都很低
# 六人桌情况下 23：win=0.0666, lose=0.9174, tie=0.0160
# 24 ：win=0.0785, lose=0.9053, tie=0.0162
# 27 ： win=0.0633, lose=0.9153, tie=0.0214
'''
hand: 2h 7c, board: 3s 4d 5c 6s, n_simulations: 10000, n_opponents: 6
win=0.3170, lose=0.2084, tie=0.4746
hand: 2h 7c, board: 3s 4d 5c Ks Td, n_simulations: 10000, n_opponents: 6
win=0.0000, lose=1.0000, tie=0.0000
'''
class RandomAgent_mcts:     # 用于在扑克环境中 随机选择 合法动作，并且在raise阶段 做了一些合法性和尺寸上的校验。 这个agent 并不做策略优化。
    # 所以这个agent 用来做什么呢？ baseline-test，或者self-play 产生随机又合法的数据。
    """
    用 MCTS 估计牌力的随机 Agent：
    - win 概率 → Raise 的采样权重
    - lose 概率 → Fold 的采样权重
    - tie 概率 → Check/Call 的采样权重
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.name = f"RandomAgent_hard{player_id}" # Added name for clarity
        # 统一：用整型作键 0=Preflop, 1=Flop, 2=Turn, 3=River
        self.raise_caps = {0: 1, 1: 1, 2: 1, 3: 2}
        self._street_raises = {0: 0, 1: 0, 2: 0, 3: 0}
        self._last_stage = None

    def _reset_if_new_hand_or_street(self, state):
        k = int(state.stage)  # 统一成整型键 0/1/2/3

        # 新手牌：preflop 且没有上一动作（根据你环境的字段名调整判定）
        if k == 0 and getattr(state, "from_action", None) is None:
            for kk in self._street_raises.keys():
                self._street_raises[kk] = 0

        # 换街：重置本街计数
        if self._last_stage is None or k != self._last_stage:
            self._street_raises[k] = 0
            self._last_stage = k

    def _can_raise_this_street(self, state) -> bool:
        cur_stage = int(state.stage)
        cap = self.raise_caps.get(cur_stage, 2)  # 如果 str(Stage.Preflop) 能得到稳定名字
        used = self._street_raises.get(cur_stage, 0)
        return used < cap

    # --------- 工具：环境 Card -> 评估用 Card ---------
    @staticmethod
    def _convert_cards_to_eval(cards):
        """
        把 pokers 环境里的牌（rank:0..12, suit:0..3）
        转成 poker_ai.poker.card.Card(rank:2..14, suit:英文字符串)。
        """
        if not cards:
            return np.array([], dtype=object)

        suit_map = {
            0: "clubs",    # C 梅花
            1: "diamonds", # D 方片
            2: "hearts",   # H 红桃
            3: "spades",   # S 黑桃
        }
        out = []
        for c in cards:
            r_env = int(c.rank)   # 0..12 -> 2..14
            s_env = int(c.suit)   # 0..3  -> 文本
            rank = r_env + 2
            suit = suit_map[s_env]
            out.append(EvalCard(rank, suit))
        return np.array(out, dtype=object)
    # --------- 工具：不加注时的兜底动作 ---------
    @staticmethod
    def _fallback_no_raise(state):
        """加注不行时的兜底：优先 Check > Call > Fold。"""
        if pkrs.ActionEnum.Check in state.legal_actions:
            return pkrs.Action(pkrs.ActionEnum.Check), None
        if pkrs.ActionEnum.Call in state.legal_actions:
            return pkrs.Action(pkrs.ActionEnum.Call), None
        if pkrs.ActionEnum.Fold in state.legal_actions:
            return pkrs.Action(pkrs.ActionEnum.Fold), None
        # 实在没动作可选（理论上不会发生）
        logging.warning("[RandomAgent_mcts] no legal fallback action, force Fold")
        return pkrs.Action(pkrs.ActionEnum.Fold), None

    # --------- 核心：用固定 50 额外加注，带完整合法性检查 ---------
    def _do_mcts_raise_fixed_50(self, state, fixed_additional: float = 50.0):
        """
        固定额外加注 fixed_additional（默认 50），并做以下检查：
        - 当前是否允许 Raise
        - 筹码是否足够 call
        - call 后是否还有多余筹码
        不合法则回退到 check/call/fold。
        """
        # 先看环境是否允许 Raise
        if pkrs.ActionEnum.Raise not in state.legal_actions:
            return self._fallback_no_raise(state)

        player_state = state.players_state[state.current_player]
        current_bet = float(player_state.bet_chips)
        available_stake = float(player_state.stake)

        call_amount = max(0.0, float(state.min_bet) - current_bet)
        eps = 1e-6

        # 1）连 Call 都不够，直接退化
        if available_stake + eps < call_amount:
            return self._fallback_no_raise(state)

        # 2）Call 后剩余筹码
        remaining_after_call = max(available_stake - call_amount, 0.0)
        if remaining_after_call <= eps:
            # 说明只能平跟，不能再加注
            return self._fallback_no_raise(state)

        # 3) 我们只想“再加 50”，但不能超过 remaining_after_call
        additional = min(remaining_after_call, float(fixed_additional))
        if additional <= eps:
            # 相当于没法多加了，也退化
            return self._fallback_no_raise(state)

        # 4) 构造 Raise 动作（amount 语义：额外加注额，与 random_agent_hard 保持一致）
        action_enum = pkrs.ActionEnum.Raise
        action = pkrs.Action(action_enum, amount=additional)

        # 这里不再做 deepcopy/apply_action 预演，完全依赖上述筹码与合法性判断
        # 记录本街加注次数
        st = int(state.stage)
        self._street_raises[st] = self._street_raises.get(st, 0) + 1

        return action, None

    def choose_action(self, state, verbose: bool = False, selection_mode = ''):
        self._reset_if_new_hand_or_street(state) # 首先 重置 统计次数
        if not state.legal_actions:  #如果legal_actions为空， 就直接进行 fold。
            # This should ideally not happen in a valid game state
            logging.warning(f"WARNING: No legal actions available for player {self.player_id}. Attempting Fold.")
            # Attempt Fold as fallback, though it might also be illegal
            return pkrs.Action(pkrs.ActionEnum.Fold), None
        # preflop:0 flop:1 turn:2 river:3 showdown:4
        # public_cards: state.public_cards
        # hand_card:判断 实例：int(my_state.hand[0].suit) int(my_state.hand[0].rank)
        # 2 3 4 5 6 7 8 9 T J  Q  K  A   牌值大小编码
        # 0 1 2 3 4 5 6 7 8 9 10 11 12
        # C梅花 D方片 H红桃 S黑桃  牌型编码
        #  0    1    2     3
        my_state = state.players_state[self.player_id]  # 先获取当前玩家的 state信息()
        # 1) 构造 MCTS 所需的牌：hero 手牌 + 公共牌
        hero_hand_eval = self._convert_cards_to_eval(my_state.hand)
        board_eval = self._convert_cards_to_eval(state.public_cards)
        # 2) 统计在场对手个数（用来传给 n_opponents）
        alive_opps = sum(
            1
            for ps in state.players_state
            if ps.active and ps.player != self.player_id
        )
        n_opps = max(1, alive_opps)
        # 3) 跑一轮 MCTS 估计牌力 [win, lose, tie]
        mcts_result = get_mcts_result(
            self.player_id,
            hero_hand_eval,
            board_eval,
            n_simulations=1000,  # 这里可以根据性能调整
            n_opponents=2,
            verbose=verbose,
        )
        win_p, lose_p, tie_p = mcts_result
        # 4) 根据牌力结果，构建 三类动作 的基础权重
        #    win -> raise, lose -> fold, tie -> check/call
        base_probs = {
            "raise": float(win_p),
            "fold": float(lose_p),
            "call": float(tie_p),
        }
        # 5) 根据 legal_actions 屏蔽不合法的类别
        # 5) 根据 legal_actions 屏蔽不合法的类别
        legal_list = list(state.legal_actions)  # 保留成列表就行，避免 set() 触发哈希

        can_fold = pkrs.ActionEnum.Fold in legal_list
        can_check = pkrs.ActionEnum.Check in legal_list
        can_call = pkrs.ActionEnum.Call in legal_list
        can_raise = (pkrs.ActionEnum.Raise in legal_list) and self._can_raise_this_street(state)

        cat_probs = {}

        if can_raise:
            cat_probs["raise"] = base_probs["raise"]
        if can_call or can_check:
            cat_probs["call"] = base_probs["call"]
        if can_fold:
            cat_probs["fold"] = base_probs["fold"]

        # 如果某些动作类别全被 mask 掉，做一下兜底：至少要有一种
        if not cat_probs:
            # 理论上不会到这儿，兜底给 call / check / fold 之一
            if can_call or can_check:
                cat_probs["call"] = 1.0
            elif can_fold:
                cat_probs["fold"] = 1.0
            elif can_raise:
                cat_probs["raise"] = 1.0
        cats = list(cat_probs.keys())
        probs = np.array([cat_probs[c] for c in cats], dtype=np.float64)

        # 归一化，如果总和为 0，则退化为均匀
        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            probs[:] = 1.0 / len(probs)
        else:
            probs /= s

        # ========= 这里改成 argmax，而不是按分布抽样 =========
        if len(cats) == 1:
            chosen_cat = cats[0]
        else:
            cat_idx = int(np.argmax(probs))
            chosen_cat = cats[cat_idx]

        if verbose:
            logging.info(
                f"player: {self.player_id}, "
                f"current stage: {stage_names.get(int(state.stage), str(state.stage))}, "
                f"[RandomAgent_mcts] mcts={mcts_result}, "
                f"cat_probs={ {c: float(p) for c, p in zip(cats, probs)} }, "
                f"chosen_cat={chosen_cat}, "
                f"can_raise={can_raise}"
            )

        # 6) 映射到具体动作（带 raise 上限 + 非法 raise 过滤）
        if chosen_cat == "fold" and can_fold:
            return pkrs.Action(pkrs.ActionEnum.Fold), None

        if chosen_cat == "call" and (can_call or can_check):
            if can_call:
                return pkrs.Action(pkrs.ActionEnum.Call), None
            else:
                return pkrs.Action(pkrs.ActionEnum.Check), None

        if chosen_cat == "raise" and can_raise:
            return self._do_mcts_raise_fixed_50(state, fixed_additional=50.0)

        # 如果 chosen_cat 对应的动作最后不可用，再兜底一次
        logging.warning(
            f"[RandomAgent_mcts] Fallback due to illegal chosen_cat={chosen_cat} "
            f"(can_fold={can_fold}, can_call={can_call}, can_check={can_check}, can_raise={can_raise})"
        )
        return self._fallback_no_raise(state)




if __name__ == "__main__":
    # 0: "clubs",  # C 梅花
    # 1: "diamonds",  # D 方片
    # 2: "hearts",  # H 红桃
    # 3: "spades",  # S 黑桃
    our_hand = np.array([
        Card(12, "spades"),
        Card(13, "clubs"),
    ], dtype=object)
    board = np.array([
        Card(7, "diamonds"),
        Card(6, "clubs"),
        Card(4, "diamonds"),
        Card(9, "spades"),
        Card(6, "diamonds"),
    ], dtype=object)  # preflop 没有公共牌

    result = get_mcts_result(player_id=0, hand_card = our_hand, board_card = board, n_simulations = 10000, n_opponents = 6, verbose= True)
    _pretty_print_ehs(result)