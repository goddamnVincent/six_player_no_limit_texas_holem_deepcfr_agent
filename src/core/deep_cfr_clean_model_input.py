# deep_cfr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pokers as pkrs  #本环境是 raise_by模式
from collections import deque
from src.core.model_clean_model_input import PokerNetwork, encode_state, VERBOSE, set_verbose
from src.utils.logging import log_game_error
import os
import logging
import traceback
import math
import copy
from src.core.scheduler_vince import Vince_scheduler
from src.agents.random_agent_mcts import convert_cards_to_eval, get_mcts_result
import copy
logger = logging.getLogger(__name__)
# 如何使用？
# lr_scheduler = Vince_scheduler(optimizer, 0, args.lr_epochs, 3, 0.7, False)
# lr_scheduler.step_epoch(epoch=epoch)

# 8.12 修改了 原始的 PrioritizedMemory 类：
# TODO:训练中 beta 从 0.4 线性退火到 1.0, 这个未来再说吧。
# 11.18： 加入 加注上限设置 preflop，flop，turn，river 加注次数上限分别是 1 1 1 2 也算是比较温和的剪枝了
# 1.22： 把2-pot 这个挡位 进行删除 变成0.5p 1p all-in 这三个档位 以此观察模型在turn river阶段更深层次的博弈？
A_FOLD, A_CHECK_CALL, A_HALF, A_POT, A_ALLIN = range(5) # 目前的legal_actions

# 弱牌 不参与 cfr - traverse
WEAK_169_HANDS = {
    "J3s", "T5s", "95s", "Q4o", "75s", "J6o", "T4s", "J2s", "22",
    "85s", "65s", "86o", "T3s", "Q3o", "T6o", "J5o", "54s", "96o",
    "Q2o", "64s", "76o", "94s", "T2s", "J4o", "84s", "93s", "74s",
    "J3o", "92s", "53s", "75o", "T5o", "83s", "J2o", "63s", "95o",
    "65o", "43s", "85o", "73s", "T4o", "82s", "52s", "54o", "T3o",
    "72s", "62s", "74o", "64o", "42s", "84o", "94o", "32s", "93o",
    "T2o", "53o", "92o", "73o", "83o", "43o", "63o", "82o", "62o",
    "52o", "72o", "42o", "32o",
}
def _rank_index_to_char(idx: int) -> str:
    """
    把 Card.rank (0..12) 转成 '2','3',...,'T','J','Q','K','A'
    0 -> '2', 8 -> 'T', 9 -> 'J', 12 -> 'A'
    """
    mapping = {
        0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7",
        6: "8", 7: "9", 8: "T", 9: "J", 10: "Q", 11: "K", 12: "A",
    }
    return mapping[int(idx)]


def log_state_and_action(state):  #辅助函数
    """打印最近动作和所有玩家状态到日志"""
    logging.info(f'current_player:{state.current_player}, stage:{state.stage}, pot:{state.pot}, min_bet:{state.min_bet}')
    if state.current_player == 0:
        logging.info('turn for deepCFR agent')
    else:
        assert state.current_player in (1,2,3,4,5)
        logging.info('turn for random agent')
    ar = getattr(state, "from_action", None)
    if ar is not None:
        logging.info(
            "Player %d at %s did %s amount=%.2f (legal: %s)",
            ar.player,
            ar.stage,
            ar.action.action,
            float(ar.action.amount),
            [a for a in ar.legal_actions],
        )
    else:
        logging.info("No previous action recorded.")

    for ps in state.players_state:
        logging.info(
            "P%d: bet=%.2f, pot=%.2f, stake=%.2f, reward=%.2f, active=%s",
            ps.player,
            float(ps.bet_chips),
            float(ps.pot_chips),
            float(ps.stake),
            float(ps.reward),
            ps.active,
        )

class DeepCFRAgent:
    def __init__(self, player_id=0, num_players=6, memory_size=100000, device='cpu'):
        self.player_id = player_id
        self.num_players = num_players
        self.device = device
        # Define action types (Fold, Check/Call, 0.5
        self.num_actions = 5 #重新改成5维度

        # Calculate input size based on state encoding
        input_size = 172 + 10 * self.num_players #
        
        # Create advantage network with bet sizing
        self.advantage_net = PokerNetwork(
            input_size=input_size, hidden_size=256,
            bounded_advantage=True,  # 关键：tanh * adv_range，抑制爆炸
            adv_range=5.0, # 优势网络 把 边界值打开，因为要预测的是一个 具体的值？
        ).to(device)
        
        # Use a smaller learning rate for more stable training
        #TODO: 这里需要注意的是， random_agent博弈完之后， 学习率应该减半， 应该是没问题的。
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=5e-5, weight_decay=1e-3) #self-play 阶段进行砍半
        #self.lr_scheduler_advantage = Vince_scheduler(self.optimizer, 0, 3.0, 10, 0.7, False) # 我们这次先不用 学习率衰减了
        # Create prioritized memory buffer

        # Strategy network
        self.strategy_net = PokerNetwork(input_size=input_size, hidden_size=256).to(device)
        # strategy nerwork 并没有使用 bounded，就是原始的 output，也没用softmax输出，因为是概率吧
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=2e-5, weight_decay=1e-4)
        #self.lr_scheduler_strategy = Vince_scheduler(self.strategy_optimizer, 0, 3.0, 10, 0.7, False) # 我们这次先不用学习率衰减了

        # For keeping statistics
        self.iteration_count = 0

        # 加注上限 设置：# stage: 0=Preflop, 1=Flop, 2=Turn, 3=River
        self.raise_caps = {
            0: 1,  # Preflop
            1: 1,  # Flop
            2: 1,  # Turn
            3: 2,  # River
        }
        # 5档版本
        self.raise_actions = [A_HALF, A_POT, A_ALLIN]
        # —— 实战用的加注次数统计（不是 cfr_traverse 那个递归参数）——
        # key: stage(0..3), value: int  这个加注次数上限 是没什么问题的
        self.live_street_raise_counts = {
            pid: {0: 0, 1: 0, 2: 0, 3: 0}
            for pid in range(self.num_players)
        }
        self._last_stage_for_live = None
    # ====== 起手牌编码 & 弱牌判断 ======
    def _hand_to_169_code(self, c1, c2) -> str:
        """
        把两张牌转换成 169 表里用的编码：
          - 对子：'22', 'JJ', ...
          - 同花：'J3s'
          - 不同花：'J3o'
        """
        r1, r2 = int(c1.rank), int(c2.rank)
        s1, s2 = c1.suit, c2.suit

        # 对子：不用 s/o 后缀
        if r1 == r2:
            ch = _rank_index_to_char(r1)
            return ch + ch

        # 非对子：高牌在前
        if r1 > r2:
            hi_r, lo_r = r1, r2
            hi_s, lo_s = s1, s2
        else:
            hi_r, lo_r = r2, r1
            hi_s, lo_s = s2, s1

        hi_ch = _rank_index_to_char(hi_r)
        lo_ch = _rank_index_to_char(lo_r)

        suited = (hi_s == lo_s)
        suffix = "s" if suited else "o"
        return hi_ch + lo_ch + suffix

    def _is_weak_preflop_hand(self, state) -> bool:
        """
        判断当前我方在 preflop 是否拿到了“想过滤掉的 103~169 号弱牌”。
        """
        stage = int(state.stage)
        if stage != 0:
            return False  # 只在 preflop 过滤

        hero = self.player_id
        try:
            c1, c2 = state.players_state[hero].hand  # 两张牌
        except Exception:
            return False  # 防御性兜底

        code = self._hand_to_169_code(c1, c2)
        return code in WEAK_169_HANDS

    def reset_for_new_hand(self):
        """实战对局开始时调用，重置每条街加注次数。"""
        self.live_street_raise_counts = {
            pid: {0: 0, 1: 0, 2: 0, 3: 0}
            for pid in range(self.num_players)
        }
        self._last_stage_for_live = None

    # 将五个离散的合法动作 尤其是下注的 2 ： 0.5p / 3 ：1.0p / 4：all-in
    def _raise_additional_from_idx(self, state, bin_idx, player_index=None):
        """
        将离散 raise 档位(2..5) → 'call 之上的增量'，并判断可行性。
        返回 (additional_amount, feasible: bool)。

        约定：
          2: 0.5 * pot       additional = max(0, 0.5 * pot)
          3: 1.0 * pot       additional = max(0, 1.0 * pot)
          4: all-in          additional = remain_after_call   # 允许 < min_inc
        """
        EPS = 1e-6
        if player_index is None:
            player_index = state.current_player
        ps = state.players_state[player_index]

        current_bet = float(ps.bet_chips) #当前下注
        available = float(ps.stake) #可用筹码
        call_amount = max(0.0, float(state.min_bet) - current_bet) #call所需的筹码数量
        remain_after_call = available - call_amount #call完之后 还剩下多少筹码
        if remain_after_call <= EPS:
            return 0.0, False  # 连 call 后都没余筹，无法加注
        pot = float(state.pot)

        # 目标追加额（相对 call）
        if bin_idx == 2:
            desired = max(0.0, 0.5 * pot)
        elif bin_idx == 3:
            desired = max(0.0, 1.0 * pot)
        elif bin_idx == 4:  # all-in
            return float(remain_after_call), remain_after_call > EPS
        else:
            # 不应该出现
            return 0.0, False

        # 只要求：>0 且不超过余筹
        feasible = (desired > EPS) and (desired <= remain_after_call + EPS)
        if not feasible:
            # 应该也不会发生
            return 0.0, False

        # 不超过 all-in
        desired = min(desired, remain_after_call)
        return float(desired), True
    # 这个只是 根据当前的环境 提取出所有的合法动作 并非mask
    def get_legal_action_types(self,
                               state):  # 返回了一个 legal_action_types: 一个列表 将 state.legal_actions 里面的合法动作 转换为 一个列表[0，1，2]
        """Get the legal action types for the current state."""
        legal_action_types = []  # 合法动作
        # Check each action type
        if pkrs.ActionEnum.Fold in state.legal_actions:
            legal_action_types.append(0)

        if (pkrs.ActionEnum.Check in state.legal_actions) or (
                pkrs.ActionEnum.Call in state.legal_actions):  # 对 check/call 进行合并
            legal_action_types.append(1)

        if pkrs.ActionEnum.Raise in state.legal_actions:
            for b in (2, 3, 4):  # 三个下注 挡位 0.5pot 1pot all-in
                add, ok = self._raise_additional_from_idx(state, b)
                if ok:
                    legal_action_types.append(b)
        return legal_action_types

    def action_type_to_pokers_action(self, action_type, state): #这边只是执行动作的，不涉及对网络输出结果的处理
        # action_type: 0 1 2 3 4  代表 fold check/call 0.5pot 1.0pot all-in
        # 这一步就是把神经网络输出的结果，动作 以及 raise挡位预测 换成可以进行内部 cfr 循环的方式 来进行推进 game过程
        try:
            if action_type == 0:  # Fold
                if pkrs.ActionEnum.Fold in state.legal_actions: # state.legal_actions: [ActionEnum.Fold, ActionEnum.Call, ActionEnum.Raise]
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                if pkrs.ActionEnum.Check in state.legal_actions:
                    logging.warning('i want to fold but can not, so i choose to check?')
                    return pkrs.Action(pkrs.ActionEnum.Check)
                if pkrs.ActionEnum.Call in state.legal_actions:
                    logging.warning('i want to fold or check but can not, so i choose to call?')
                    return pkrs.Action(pkrs.ActionEnum.Call)
                logging.warning('i want to fold or check or call, but cannot, so i enforce to fold whatever...')
                return pkrs.Action(pkrs.ActionEnum.Fold) # Last resort

            elif action_type == 1:  # check/call
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                elif pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Fold in state.legal_actions:  #should not happen...
                    logging.warning('i want to check/call, but i can not, so i fold?')
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                logging.warning('i want to check/call or fold, but i can not, so i enforce to fold...')
                return pkrs.Action(pkrs.ActionEnum.Fold) # Last resort

            elif action_type in (2, 3, 4):
                if pkrs.ActionEnum.Raise not in state.legal_actions:
                    logging.warning('i want to raise, but cannot?')
                    if pkrs.ActionEnum.Call in state.legal_actions:   #call first.
                        logging.warning('cannot raise, so call')
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    if pkrs.ActionEnum.Check in state.legal_actions:
                        logging.warning('cannot raise, so check')
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    logging.warning('cannot raise, and i fold, this case should not happen...')
                    return pkrs.Action(pkrs.ActionEnum.Fold)

                player_state = state.players_state[state.current_player]
                current_bet = player_state.bet_chips        # What player already has in pot this round
                available_stake = player_state.stake        # Player's remaining chips
                #min_bet 是什么，是当前别人下注的最高额度
                # current_bet 则是，本回合你已经放入的筹码， 所以二者之差 就代表着 你 to-call时候需要的总筹码量
                call_amount = max(0.0, state.min_bet - current_bet) # Additional chips needed to call

                if available_stake <= call_amount:  #不过一般不会发生，因为我已经有mask 做了兜底 所以理论上 应该不会走进这个分支
                    if pkrs.ActionEnum.Check in state.legal_actions:
                        logging.warning(
                            "choose_to_raise but cannot even CALL "
                            f"(call={call_amount:.2f}, avail={available_stake:.2f}); fallback -> CHECK"
                        )
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        logging.warning(
                            "choose_to_raise but cannot even CALL "
                            f"(call={call_amount:.2f}, avail={available_stake:.2f}); fallback -> CALL"
                        )
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    logging.warning("choose_to_raise but cannot even CALL; fallback -> FOLD")
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                # 在满足 call的前提下 你才能去 raise 此处删除了 min_raise的计算 以及 判断 traverse 应该会快捷很多

                # 如果进入到这里，代码才开始 真正的开始审视raise 多少，上面都是 是否可以raise的检查
                add, ok = self._raise_additional_from_idx(state, action_type)
                if ok:
                    return pkrs.Action(pkrs.ActionEnum.Raise, add)  #如果 ok，直接返回 raise 和 raise amount，并且执行这个合法的raise
                # 一般而言，应该不会触发到下面的代码

                # 理论上不会到这里；兜底
                logging.warning('fixed RAISE infeasible unexpectedly; fallback')
                if pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                return pkrs.Action(pkrs.ActionEnum.Fold)

            else: # Should not be reached if action_type is 0, 1, or 2
                logging.warning(f"DeepCFRAgent ERROR: Unknown action type: {action_type}")
                if pkrs.ActionEnum.Call in state.legal_actions: return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Check in state.legal_actions: return pkrs.Action(pkrs.ActionEnum.Check)
                return pkrs.Action(pkrs.ActionEnum.Fold)

        except Exception as e:
            logging.warning(f"DeepCFRAgent CRITICAL ERROR in action_type_to_pokers_action: Type {action_type} for player {self.player_id}: {e}")
            logging.warning(f"  State: current_player={state.current_player}, stage={state.stage}, legal_actions={state.legal_actions}")
            if hasattr(state, 'players_state') and self.player_id < len(state.players_state):
                logging.warning(f"  Player {self.player_id} stake: {state.players_state[self.player_id].stake}, bet: {state.players_state[self.player_id].bet_chips}")
            else:
                logging.warning(f"  Player state for player {self.player_id} not accessible.")
            import traceback
            traceback.print_exc()
            if hasattr(state, 'legal_actions'):
                if pkrs.ActionEnum.Call in state.legal_actions: return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Check in state.legal_actions: return pkrs.Action(pkrs.ActionEnum.Check)
                if pkrs.ActionEnum.Fold in state.legal_actions: return pkrs.Action(pkrs.ActionEnum.Fold)
            return pkrs.Action(pkrs.ActionEnum.Fold)

    def _legal_with_raise_cap_live(self, state):
        """
        实战 choose_action 使用的合法动作过滤：
        - 基于 get_legal_action_types(state) 的原始合法动作
        - 再结合 self.raise_caps 和 self.live_street_raise_counts 做“加注次数上限”
        """
        base_legal = self.get_legal_action_types(state)
        stage = int(state.stage)
        # 1) preflop / flop 阶段：屏蔽 all-in（无论是否设置 raise cap）
        if stage in (0, 1):
            if A_ALLIN in base_legal:
                # 可选 debug：如果你想确认是否真的发生过
                # logging.info(f"[LIVE LEGAL] stage={stage} removing ALLIN from {base_legal}")
                base_legal = [a for a in base_legal if a != A_ALLIN]
        pid = state.current_player #当前玩家
        cap = self.raise_caps.get(stage, None)
        if cap is None:
            # 这条街没有加注上限要求，原样返回
            return base_legal

        used = self.live_street_raise_counts[pid][stage]
        if used >= cap:
            # 已经到上限：把所有 raise_actions 从合法动作里删掉
            return [a for a in base_legal if a not in self.raise_actions]
        else:
            return base_legal

    def choose_action(self, state, verbose: bool = False,
                      selection_mode: str = 'sample', #sample / argmax / eps_greedy
                      epslion: float = 0.05):
        """Choose an action for the given state during actual play (greedy on probs)."""
        # 这个choose_action 不会影响训练 就是单纯的推理时候用到的

        stage = int(state.stage)
        # ==== preflop 弱牌处理 ====
        if stage == 0 and self._is_weak_preflop_hand(state):

            # ---- 先判断：自己是不是大盲位？ ----
            num_players = len(state.players_state)
            # 常规规则：button+1 = SB, button+2 = BB
            bb_seat = (state.button + 2) % num_players
            #logging.info(f'bb seat:{bb_seat}')
            hero = state.players_state[self.player_id]
            to_call = max(0.0, float(state.min_bet) - float(hero.bet_chips))

            # “免费过牌”的判定：可以 Check，
            # 或者只能 Call 但跟注额为 0（很多引擎会用 Call 代替 Check）
            can_free_check = (
                    (pkrs.ActionEnum.Check in state.legal_actions)
                    or (
                            pkrs.ActionEnum.Call in state.legal_actions
                            and to_call <= 1e-6
                    )
            )

            # === 大盲位 + 免费 check：弱牌也不 FOLD，白看一张牌 ===
            if (self.player_id == bb_seat) and can_free_check:
                action_type = A_CHECK_CALL
                if verbose:
                    logging.info(
                        "[choose_action] Weak preflop hand in BB, but free-check is available. "
                        "Force CHECK/CALL instead of FOLD. hand=%s",
                        self._hand_to_169_code(*hero.hand) if hero.hand else "??",
                    )
            else:
                # 其它位置仍然按照原来的策略：弱牌直接 FOLD
                action_type = A_FOLD
                if verbose:
                    logging.info(
                        "[choose_action] Weak preflop hand (%s) detected, force FOLD.",
                        self._hand_to_169_code(*hero.hand) if hero.hand else "??",
                    )

            return self.action_type_to_pokers_action(action_type, state), action_type

        # —— 如果街道变化了（比如从翻前到翻牌），重置这一街的加注次数 —— #
        if self._last_stage_for_live is None or stage != self._last_stage_for_live:
            # 新进入这一街，清零这一街的加注计数
            for pid in range(self.num_players):
                self.live_street_raise_counts[pid][stage] = 0
            self._last_stage_for_live = stage
        legal_action_types = self._legal_with_raise_cap_live(state)  # [0,1,2,3,4]

        # logging.info('cfr is choosing!!!!!!!!!')
        if not legal_action_types: #这都是列表的。
            # Default fallbacks
            logging.warning('when cfr agent choose action for play, no legal actions?')
            if pkrs.ActionEnum.Call in state.legal_actions:
                logging.warning('when cfr agent choose action for play, no legal actions? choose call as default')
                return pkrs.Action(pkrs.ActionEnum.Call), 1 # call : 1
            elif pkrs.ActionEnum.Check in state.legal_actions:
                logging.warning('when cfr agent choose action for play, no legal actions? choose check as default')
                return pkrs.Action(pkrs.ActionEnum.Check), 1 # check:1
            else:
                logging.warning('when cfr agent choose action for play, no legal actions? choose fold as default')
                return pkrs.Action(pkrs.ActionEnum.Fold), 0 #fold: 0

        # 前向计算得到动作概率与下注倍率
        x = torch.as_tensor(
            encode_state(state, self.player_id, legal_action_types),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        self.strategy_net.eval()
        with torch.no_grad():
            act_logits, raise_logits = self.strategy_net(x)  # act:[1,3], raise:[1,4]
            act_logits = act_logits[0]  # [3]
            raise_logits = raise_logits[0]  # [3]

            # 组装 5 档原子动作的 logits
            # 0:Fold, 1:Check/Call, 2..4: Raise = base(=act_logits[2]) + raise_logits[0..2]
            logits5 = torch.full((5,), -1e30, device=self.device, dtype=act_logits.dtype)
            logits5[0] = act_logits[0]  # Fold
            logits5[1] = act_logits[1]  # Check/Call
            logits5[2:5] = act_logits[2] + raise_logits  # Raise三档
            logits5 = logits5.detach().cpu().numpy().astype(np.float64)

        masked_logits = np.full_like(logits5, -1e30)
        masked_logits[legal_action_types] = logits5[legal_action_types]

        sub_logits = masked_logits[legal_action_types]  # 子集 logits
        sub_logits = sub_logits - sub_logits.max()  # 数值稳定
        sub_probs = np.exp(sub_logits)
        z = sub_probs.sum()
        if z <= 0 or not np.isfinite(z):
            # 极端数值兜底：均匀
            sub_probs = np.ones_like(sub_probs) / len(sub_probs)
            logging.warning(f'sub_probs:{sub_probs}, 出现极端数值了, 需要看看！')
        else:
            sub_probs = sub_probs / z  # 子集上的 softmax 概率

        if selection_mode == "argmax":
            pick_idx = int(np.argmax(sub_probs))
        elif selection_mode == "eps_greedy":
            if np.random.rand() < max(0.0, min(1.0, float(epslion))):
                # 探索：在合法动作里均匀随机
                pick_idx = int(np.random.randint(0, len(legal_action_types)))
            else:
                # 开发：取最大概率
                pick_idx = int(np.argmax(sub_probs))
        else:
            # "sample"（默认）：按概率抽样
            pick_idx = int(np.random.choice(len(legal_action_types), p=sub_probs))

        action_type = int(legal_action_types[pick_idx])
        # 选出 动作之后 我们要在 preflop 和 flop阶段 把 all-in 挡位进行屏蔽 不要出现这种极端情况
        if (stage == 0 or stage == 1) and action_type == A_ALLIN:
            preferred = None
            if A_POT in legal_action_types:
                preferred = A_POT
            elif A_HALF in legal_action_types:
                preferred = A_HALF
                #logging.warning('不是 一池 为什么不能选呢？ 选半池了那就')
            elif A_2P in legal_action_types:
                preferred = A_2P
                #logging.warning('不是 一池 为什么不能选呢？ 选二池了那就')
            elif A_CHECK_CALL in legal_action_types:
                preferred = A_CHECK_CALL
                #logging.warning('不是 一池 为什么不能选呢？ 只能call了')
            else:
                # 极端兜底：随便挑一个合法动作
                preferred = legal_action_types[0]
                logging.warning('不是 一池 为什么不能选呢？ 随便选了？')
            action_type = int(preferred)

        # turn / river: 阶段：如果 pot 不深 + 胜率很高，就别直接 all-in，先“打价值”
        # 训练的时候 先注释掉了 推理的时候 记得打开
        if stage in (2, 3) and action_type == A_ALLIN:
            logging.info('you choose all-in!!!!!!!!!!!!!')
            hero = state.players_state[self.player_id]
            pot = float(max(state.pot, 1.0))

            eff_stack = min(
                float(hero.stake),
                max(
                    (ps.stake for ps in state.players_state
                     if ps.active and ps.player != self.player_id),
                    default=0.0
                )
            )
            spr = eff_stack / pot  # raw SPR
            # === 用 MCTS 看一下当前牌力
            try:
                hero_hand_eval = convert_cards_to_eval(hero.hand)
                board_eval = convert_cards_to_eval(state.public_cards)
                alive_opps = sum(
                    1
                    for ps in state.players_state
                    if ps.active and ps.player != self.player_id
                )
                n_opps = max(1, alive_opps)

                mcts_result = get_mcts_result(
                    self.player_id,
                    hero_hand_eval,
                    board_eval,
                    n_simulations=1000,
                    n_opponents= n_opps, #自己 + 对手
                    verbose=True,
                )
                logging.info(f'[all-in MCTS]: {mcts_result}')
                hero_equity = float(mcts_result[0])
            except Exception as e:
                logging.warning(f"[choose_action] MCTS equity failed in all-in control: {e}")
                hero_equity = None

            # 阈值可以之后慢慢调：高胜率 + 深 SPR → 倾向“慢打”
            HIGH_EQUITY = 0.70   # 胜率很高
            LOW_EQUITY = 0.5 # 胜率不高
            HIGH_SPR    = 4.0    # SPR 比较深（可以慢慢打）
            # if (hero_equity is not None) and (hero_equity >= HIGH_EQUITY) and (spr >= HIGH_SPR):
            if (hero_equity is not None) and (hero_equity >= HIGH_EQUITY): #暂时不要 spr 来辅助判断了
                preferred = None
                # 首选1P > 2P > 0.5P > Check/Call
                for cand in (A_POT, A_HALF, A_CHECK_CALL):
                    if cand in legal_action_types:
                        preferred = cand
                        break

                # extreme fallback：如果只有 all-in 合法，那就保留 all-in
                if preferred is not None and preferred != A_ALLIN:
                    logging.info(
                        "[ALL_IN Warning!] downgrade ALL-IN -> %s due to high equity=%.3f and SPR=%.2f "
                        "to extract more value.",
                        preferred, hero_equity, spr
                    )
                    action_type = int(preferred)
            elif hero_equity <= HIGH_EQUITY: #如果 mcts 胜率不是很高，直接fold 不冒险(可能比较粗糙)
                logging.info(
                    "[ALL_IN Warning!] downgrade ALL-IN -> FOLD/check  due to low equity=%.3f and SPR=%.2f "
                    "to avoid losing too much.",
                    hero_equity, spr
                )
                if pkrs.ActionEnum.Check in state.legal_actions: #如果能免费check 就check一下 绝对不call
                    action_type = A_CHECK_CALL
                else:
                    action_type = A_FOLD
            else: #卡在50 - 70 中间 选择 check/call 试试
                logging.info(
                    "[ALL_IN Warning!] downgrade ALL-IN -> check/call  due to low equity=%.3f and SPR=%.2f "
                    "to avoid losing too much.",
                    hero_equity, spr
                )
                if pkrs.ActionEnum.Check in state.legal_actions:
                    action_type = A_CHECK_CALL
                elif pkrs.ActionEnum.Call in state.legal_actions:
                    action_type = A_CHECK_CALL
                else:
                    action_type = A_FOLD


        # === 关键修复：如果可以 free-check，就永远不要真正 fold ===  free-call 也行 free-call 代表的是 BB位置
        if action_type == A_FOLD:
            hero = state.players_state[self.player_id]
            to_call = max(0.0, float(state.min_bet) - float(hero.bet_chips))

            can_free_check = (
                (pkrs.ActionEnum.Check in state.legal_actions)
                or (
                    pkrs.ActionEnum.Call in state.legal_actions
                    and to_call <= 1e-6   # call 0，相当于check
                )
            )
            if can_free_check:
                logging.info(
                    "[Enforce to Check Warning] override: model chose FOLD but a free CHECK is legal, "
                    "force action_type -> A_CHECK_CALL"
                )
                action_type = A_CHECK_CALL

        if stage in self.raise_caps and action_type in self.raise_actions:
            pid = state.current_player
            self.live_street_raise_counts[pid][stage] += 1
        if verbose:
            print(f'当前玩家: {state.current_player}')
            action_names = {
                0: "Fold", 1: "Check/Call", 2: "0.5P", 3: "1P", 4: "All-in"
            }
            msg = ", ".join(
                f"{action_names.get(a, str(a))}={sub_probs[i]:.4f}"
                for i, a in enumerate(legal_action_types)
            )
            print(f"[choose_action] legal probs: {msg}")
            try:
                if any(a >= 2 for a in legal_action_types):
                    details = []
                    for i, a in enumerate(legal_action_types):
                        if a >= 2:
                            add, ok = self._raise_additional_from_idx(state, a)
                            details.append(f"{action_names[a]}: add={add:.2f}, ok={ok}, p={sub_probs[i]:.3f}")
                    if details:
                        print("[choose_action] raise tiers: " + " | ".join(details))
            except Exception:
                pass
            # my_state = state.players_state[self.player_id]  # 先获取当前玩家的 state信息()
            # # 1) 构造 MCTS 所需的牌：hero 手牌 + 公共牌
            # hero_hand_eval = convert_cards_to_eval(my_state.hand)
            # board_eval = convert_cards_to_eval(state.public_cards)
            # alive_opps = sum(
            #     1
            #     for ps in state.players_state
            #     if ps.active and ps.player != self.player_id
            # )
            # n_opps = max(1, alive_opps)
            # mcts_result = get_mcts_result(
            #     self.player_id,
            #     hero_hand_eval,
            #     board_eval,
            #     n_simulations=1000,  # 这里可以根据性能调整
            #     n_opponents=n_opps,
            #     verbose=verbose,
            # )
            # print(f'[mcts_result] alive people: {n_opps} mcts_result:{mcts_result}')
        return self.action_type_to_pokers_action(action_type, state), action_type

    def load_model(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.iteration_count = checkpoint['iteration']
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
