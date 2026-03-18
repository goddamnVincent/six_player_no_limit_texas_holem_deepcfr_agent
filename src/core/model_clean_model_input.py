# src/code/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
VERBOSE = False
import logging
import pokers as pkrs
from collections import Counter
# 9.15: 直接改成8头输出。不需要使用额外的 头 去预测下注额度了
# legal_actions: fold\check\call   half-pot\full-pot\all-in
# 0.5 min 1.0 all-in 2.0
# 9.18自检： 应该没有什么问题
# 10.10：重新改成双头进行训练。不再使用直接预测八头
# 10.23：仍然是双头 但是 取消min_raise这个档位 这个挡位有点多余 会严重拖慢 traverse的速度。
# 11.14：重新改回双头
# 1.22： 删除 2pot 这个挡位
logger = logging.getLogger(__name__)

def set_verbose(verbose_mode):
    """Set the global verbosity level"""
    global VERBOSE
    VERBOSE = verbose_mode

class ResBlock(nn.Module):
    """Pre-LN 残差 MLP 块：LN -> Linear -> GELU -> Dropout -> Linear(zero-init) -> 残差"""
    def __init__(self, hidden_size: int, dropout: float = 0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 让块一开始近似恒等：第二层权重置零（残差=输入）
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = self.ln1(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return x + y

class PokerNetwork(nn.Module):
    """
    稳定版 Poker 网络（advantage-net / strategy-net 通用）：
    - bounded_advantage=True 时，动作头输出为 tanh*adv_range（回归优势更稳）
    - 否则输出 logits（给策略网做交叉熵）
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_actions: int = 3, # fold,check/call,raise   check和call本来就是一体的，两者是互斥关系，所以直接合并成一个就行了
        num_raise_amount: int = 3, # half-pot \ full-pot \ all-in 取消min_raise，twice_pot 这两个档位 留下三个挡位
        num_blocks: int = 3,      # 残差块个数，2~4 都可  要么试试三个？
        dropout: float = 0.05,
        bounded_advantage: bool = False,
        adv_range: float = 5.0,  # 优势值范围（与你的 target clip 一致） 暂时不用
    ):
        super().__init__()
        self.bounded_advantage = bounded_advantage
        self.adv_range = float(adv_range)


        # Stem：Linear -> (LN) -> GELU
        self.stem = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # 残差块
        self.blocks = nn.ModuleList([ResBlock(hidden_size, dropout) for _ in range(num_blocks)])

        # 末端 LN（让不同 batch 的量纲更稳）
        self.out_ln = nn.LayerNorm(hidden_size)

        # 动作头（优势/策略共用）
        self.action_head = nn.Linear(hidden_size, num_actions)
        # 下注金额头
        self.raise_amount_head = nn.Linear(hidden_size, num_raise_amount) #不用

        # 初始化
        self._init_weights()
        logging.info('模型初始化完成！准备开始训练！')
    def _init_weights(self):
        # 干路：Kaiming
        def kaiming(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

        self.apply(kaiming)

        if self.bounded_advantage:
            # 优势网：小初始化，防止一上来就把 tanh 推满
            nn.init.uniform_(self.action_head.weight, -1e-3, 1e-3)
            nn.init.zeros_(self.action_head.bias)
            nn.init.uniform_(self.raise_amount_head.weight, -1e-3, 1e-3) # 下注头的初始化
            nn.init.zeros_(self.raise_amount_head.bias)
        else:
            # 策略网：标准 xavier 更适合 logits
            nn.init.xavier_uniform_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)
            nn.init.xavier_uniform_(self.raise_amount_head.weight) # 下注头的初始化
            nn.init.zeros_(self.raise_amount_head.bias)

    def forward(self, x):
        # x: [B, input_size]
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_ln(h)

        action_raw = self.action_head(h) # 三类动作的 后悔值？
        if self.bounded_advantage:
            # 有界优势（[-adv_range, adv_range]）
            action_out = self.adv_range * torch.tanh(action_raw)  #5.0 * (-1, 1)   # 使用的是 强制tanh映射，而不是 期望和方差的形式 进行归一化的，所以这一点还需要确定一下。
            # 因为要计算 一个 类似于 mse 的损失，所以我在这里直接进行一个归一化到 -1，1之间，因为gt也是 -1到1之间 应该没什么问题吧
        else:
            # 策略 logits（不加非线性）
            action_out = action_raw
        raise_bin_logits = self.raise_amount_head(h)

        return action_out, raise_bin_logits


def encode_state(state, player_id: int = 0, legal_action_types = None):  #编码长度为220
    # ---- 动作索引（固定 8 档）----
    A_FOLD, A_CHECK_CALL, A_HALF, A_POT, A_ALLIN = range(5) #0 1 2 3 4 5

    # ---- 归一化工具 ----
    def pot_features(state, hero_id, init_stack=200.0):
        EPS = 1e-6
        N = len(state.players_state)
        players = state.players_state
        alive_now = sum(1 for ps in players if ps.active)
        alive_now = max(1, alive_now)

        # 1) 原始量
        pot_total = float(max(state.pot, 0.0))  # 整个底池
        street_pot = float(sum(max(ps.bet_chips, 0.0) for ps in players))  # 本轮下注之和
        live_commit_total = float(sum(  # 仍在场上玩家的“活筹码贡献”
            max(ps.pot_chips, 0.0) + max(ps.bet_chips, 0.0)
            for ps in players if ps.active
        ))

        # 2) 参考尺度（两种视角，避免“活人少但历史筹码多”的错位）
        #   - 对“本轮下注”更合适：按当前活人定标
        ref_alive = min(init_stack, 50.0 * alive_now)  # 例如 6-max 时上限 200，HU 时 100
        ref_alive = max(ref_alive, 1.0)

        #   - 对“总底池”更合适：按“曾经投过钱的人”定标（包含已弃牌但有 pot_chips 的）
        committed_cnt = sum(1 for ps in players if (ps.pot_chips > 0.0))
        committed_cnt = max(committed_cnt, alive_now)  # 至少不低于当前活人
        ref_comm = min(init_stack, 50.0 * committed_cnt)
        ref_comm = max(ref_comm, 1.0)

        # 3) 三个特征（都在 0~1）
        pot_total_norm = np.tanh(pot_total / ref_comm)  # “总底池有多大”（考虑历史投入者人数）
        street_pot_norm = np.tanh(street_pot / ref_alive)  # “本轮下注有多大”（对当前行动相关）
        folded_frac = 0.0
        if pot_total > EPS:
            folded_contrib = pot_total - live_commit_total  # 底池中来自已弃牌玩家的那一部分
            folded_frac = float(np.clip(folded_contrib / pot_total, 0.0, 1.0))

        return np.array([pot_total_norm, street_pot_norm, folded_frac], dtype=np.float32)

    def norm_bet(x):  #对于 bet 的下注归一化：  中小额度敏感，大额自然饱和
        u =  float(x) / 200.0 #比例  如果你下了100 那么就是0.75  下了50就是0.46  下了200就是 0.96
        return np.tanh( u / 0.5)  #100也是比较大的金额了

    def norm_stack(x: float) -> float:  #这就是 纯线性的 专门对 stack进行优化的
        u = float(x) / 200.0
        if not (0.0 <= u <= 1.0):
            logging.warning(f'[encode_state] stake normalized out of [0,1]: raw={x}, u={u}')
            u = float(np.clip(u, 0.0, 1.0))
        return u

    N = len(state.players_state)
    hero = state.players_state[player_id]
    #logging.info(f'player_id:{player_id}')
    encoded = []

    # ===== 1) 手牌 52 one-hot（自己） ===== 52
    hand_enc = np.zeros(52, dtype=np.float32)
    for c in hero.hand:
        ridx, sidx = int(c.rank), int(c.suit)  # 0..12 / 0..3
        hand_enc[sidx * 13 + ridx] = 1.0
    encoded.append(hand_enc)
    #logging.info(f'hand_enc: {hand_enc}')
    # ===== 2) 公共牌 52 one-hot ===== 104
    board_enc = np.zeros(52, dtype=np.float32)
    board_ranks = []  #牌值  #但是这里的长度 应该不是固定的吧？ 公共牌不得看你翻几张么？
    board_suit_cnt = np.zeros(4, dtype=np.int32)  #花色
    for c in state.public_cards:
        ridx, sidx = int(c.rank), int(c.suit)
        board_enc[sidx * 13 + ridx] = 1.0
        board_ranks.append(ridx)
        board_suit_cnt[sidx] += 1
    encoded.append(board_enc)
    #logging.info(f'board_enc: {board_enc}')
    # ===== 3) 阶段 one-hot（5） ===== 109
    stage_enc = np.zeros(5, dtype=np.float32)  # Preflop, Flop, Turn, River, Showdown 没毛病
    stage_enc[int(state.stage)] = 1.0
    encoded.append(stage_enc)
    #logging.info(f'stage_enc: {stage_enc}')
    # ===== 4) 底池规模（1） ===== 112
    pot = float(max(state.pot, 1e-6)) #使用norm_pot 归一化
    encoded.append(pot_features(state, player_id))   # 3 维： [pot_total_norm, street_pot_norm, folded_frac]
    #logging.info(f'pot: {pot}, encoded_pot:{np.array([norm_pot(pot)], dtype=np.float32)}')
    # ===== 5)     # --- 固定位置角色编码（长度6: SB, BB, UTG, HJ, CO, BTN） ---
    btn = int(state.button)
    # 定义角色索引映射
    pos_id_by_seat = {
        (btn + 1) % N: 0,  # SB
        (btn + 2) % N: 1,  # BB
        (btn + 3) % N: 2,  # UTG
        (btn + 4) % N: 3,  # HJ
        (btn + 5) % N: 4,  # CO
        btn: 5,  # BTN
    }
    my_pos_enc = np.zeros(6, dtype=np.float32)
    my_pos_idx = pos_id_by_seat.get(player_id, -1)
    if my_pos_idx >= 0:
        my_pos_enc[my_pos_idx] = 1.0
    encoded.append(my_pos_enc) #116
    #logging.info(f'my_pos_enc:{my_pos_enc}')
    # ===== 6)  我前面还有几个 我后面还有几个
    # 位置索引（0..5）：SB, BB, UTG, HJ, CO, BTN
    def is_active(seat):
        ps = state.players_state[seat]
        return bool(ps.active)

    # 顺时针（你出手之后要行动的人）=> behind
    active_behind = 0
    for k in range(1, N):
        seat = (player_id + k) % N
        if is_active(seat):
            active_behind += 1

    den = float(max(N - 1, 1))
    still_alive = active_behind / den
    encoded.append(np.array([still_alive], dtype=np.float32)) #119
    #logging.info(f'still_alive:{still_alive}')

    # ===== 7) 逐人公开量（每人 10 维：4公开量 + 6角色） ===== 119 + 60 = 179
    for seat in range(N):  # 按原座位顺序 0..N-1
        ps = state.players_state[seat]

        active = 1.0 if ps.active else 0.0
        bet = norm_bet(ps.bet_chips)  # 下注(本轮)归一化
        potc = norm_stack(ps.pot_chips)  # 累计投入归一化
        stake = norm_stack(ps.stake)  # 剩余筹码归一化

        # 固定角色 one-hot：SB/BB/UTG/HJ/CO/BTN
        pos_vec = np.zeros(6, dtype=np.float32)
        pos_idx = pos_id_by_seat.get(seat, -1)
        if pos_idx >= 0:
            pos_vec[pos_idx] = 1.0

        # 合并为 10 维特征
        per_seat_feats = np.concatenate(
            [np.array([active, bet, potc, stake], dtype=np.float32), pos_vec],
            dtype=np.float32
        )
        encoded.append(per_seat_feats)

        #role_str = role_names[pos_idx] if pos_idx >= 0 else "UNK"
        # logging.info(
        #     f"seat={seat} role={role_str} | "
        #     f"active={active:.1f}, bet_n={bet:.4f}(raw={ps.bet_chips}), "
        #     f"pot_n={potc:.4f}(raw={ps.pot_chips}), stake_n={stake:.4f}(raw={ps.stake}), "
        #     f"pos_vec={pos_vec.tolist()}"
        # )

    # ===== 8) 最小下注（1）===== 这是当前别人下注的最高额度 179 + 1 = 180
    encoded.append(np.array([norm_bet(state.min_bet)], dtype=np.float32)) #暂时不用min_bet

    # ===== 9) 合法动作 5 档 ===== A_FOLD, A_CHECK_CALL, A_HALF, A_POT, A_ALLIN 这边的合法动作 直接使用 cfr-traverse 里面的mask进来了，不做单独的判断，要对齐同步！
    legal_actions_enc = np.zeros(5, dtype=np.float32)  # 初始化所有合法动作为 0
    if legal_action_types is not None:
        if 0 in legal_action_types:
            legal_actions_enc[A_FOLD] = 1.0  # 如果合法动作中包含 Fold
        if 1 in legal_action_types:
            legal_actions_enc[A_CHECK_CALL] = 1.0  # 如果合法动作中包含 Check/Call
        if 2 in legal_action_types:
            legal_actions_enc[A_HALF] = 1.0  # 如果合法动作中包含 0.5P
        if 3 in legal_action_types:
            legal_actions_enc[A_POT] = 1.0  # 如果合法动作中包含 1.0P
        if 4 in legal_action_types:
            legal_actions_enc[A_ALLIN] = 1.0  # 如果合法动作中包含 All-in
    else:
        raise KeyError('no legal actions?')

    encoded.append(legal_actions_enc)  # 添加合法动作编码到状态编码中
    #logging.info(f'legal_actions_enc:{legal_actions_enc},0-fold, 1-check-call, 2-raise')

    # ===== 10) 上个动作（3 类 one-hot（合并 Check/Call）+ 金额 1 = 4）=====  186 + 4 = 190
    prev_action_enc = np.zeros(4, dtype=np.float32)
    if state.from_action is not None:
        a = state.from_action.action.action  # 这是一个 ActionEnum
        if a == pkrs.ActionEnum.Fold:
            prev_action_enc[0] = 1.0
        elif a in (pkrs.ActionEnum.Check, pkrs.ActionEnum.Call):
            prev_action_enc[1] = 1.0
        elif a == pkrs.ActionEnum.Raise:
            prev_action_enc[2] = 1.0
        # 金额（对 Raise 时是加注额；Check/Call/Fold 通常为 0）
        prev_action_enc[3] = norm_bet(state.from_action.action.amount)
    encoded.append(prev_action_enc)
    #logging.info(f'prev_action_enc: {prev_action_enc}')
    # ===== A) 代价/尺度特征（3）===== 190 + 2 = 192
    to_call_amt = max(0.0, float(state.min_bet) - float(hero.bet_chips))
    to_call_amt = norm_stack(to_call_amt) # call_amount 使用stack进行归一化
    eff_stack = min(float(hero.stake),
                    max((ps.stake for ps in state.players_state if ps.active and ps.player != player_id), default=0.0))
    # 先找出 当前还在场上的 玩家 手里筹码最多的，然后取最小值，就是这手牌有可能 打到的最大有效对抗筹码量  是一个 >0 <200的量吧
    spr = float(eff_stack) / pot # （stack-to-pot ratio）：有效筹码 / 底池（对是否持续激进非常关键）
    C = 4.0 #4.0 保证一定 在0-1 之间 因为 sb + bb = 3，最小的pot就是3
    spr = spr / (spr + C)  # ∈ (0,1)
    cost_feats = np.array([
        to_call_amt, spr
    ], dtype=np.float32)
    encoded.append(cost_feats)
    #logging.info(f'cost_feats: {cost_feats}, spr:{float(eff_stack) / pot}')
    # ===== D) 牌面信息===== 要把手牌 和 公共牌的 组合 全部打进来
    # ===== D) “手牌×公共牌”组合语义（11维）=====  仍然替代原先的 5 + 6 = 11 维
    # ===== D) 手牌×公共牌 组合语义（34维）===== 192 + 42 = 234
    from collections import Counter

    # ---------- 基础集合/统计 ----------
    hero_ranks = [int(c.rank) for c in hero.hand]  # 两张洞牌点数 0..12 (A=12)
    hero_suits = [int(c.suit) for c in hero.hand]  # 两张洞牌花色 0..3
    r1, r2 = hero_ranks

    board_ranks = [int(c.rank) for c in state.public_cards]  # 牌面点数
    board_suits = [int(c.suit) for c in state.public_cards]  # 牌面花色
    bc = Counter(board_ranks)  # 牌面点数计数
    sc = Counter(board_suits)  # 牌面花色计数

    board_set = set(board_ranks)
    board_max = max(board_ranks) if board_ranks else -1 #公共牌 最高点数
    board_min = min(board_ranks) if board_ranks else 99 #公共牌 最低点数
    n_board = len(board_ranks) # 多少张 公共牌

    # 轮顺视角（A 也当 -1）
    def _with_wheel(ranks): #A2345 / TJQKA
        S = set(ranks)
        if 12 in S:
            S = set(S)
            S.add(-1)
        return S

    board_set_wheel = _with_wheel(board_ranks)
    hero_set_std = set(hero_ranks)
    hero_set_wheel = _with_wheel(hero_ranks)
    # 首先判断自己的手牌强度
    # 1. 口袋对子？
    is_pair_pp = 1.0 if r1 == r2 else 0.0
    # 2. 同色？
    is_suited = 1.0 if hero_suits[0] == hero_suits[1] else 0.0
    # 3. 两张T+？
    both_broadway = 1.0 if (min(r1, r2) >= 8) else 0.0
    # 4. Ace + High？
    # 这里定义为：有一张 A，另一张牌 >= T（也算强 A 牌）
    if 12 in hero_ranks:
        other = r1 if r2 == 12 else r2
        ace_plus_high = 1.0 if other >= 8 else 0.0
    else:
        ace_plus_high = 0.0
    # 5. avg_rank_strength 平均手牌强度？
    avg_rank_strength = (r1 + r2) / (2.0 * 12.0)
    # 6. gap？
    gap_norm = abs(r1 - r2) / 12.0
    # 7. 同花 + 相邻？
    is_suited_and_connected = 1.0 if (is_suited == 1.0 and abs(r1 - r2) == 1) else 0.0
    # 8. min_gap <= 2?
    min_gap_le2 = 1.0 if abs(r1 - r2) <= 2 else 0.0

    # 汇总成 8 维 hero-only 手牌特征
    hero_only_feats = np.array([
        is_pair_pp,  # 1  口袋对子
        is_suited,  # 2  同花
        both_broadway,  # 3  两张 T+
        ace_plus_high,  # 4  A + 高张
        avg_rank_strength,  # 5  平均点数强度
        gap_norm,  # 6  点数差归一
        is_suited_and_connected,  # 7  同花连张
        min_gap_le2,  # 8  至多隔一张牌（连接性好）
    ], dtype=np.float32)

    # 公共牌强度？
    # 1. 是否至少一对？
    # 2. 是否是三条/葫芦/四条的起点
    # ---------- 1 & 2: 是否至少一对 / 是否三条或以上 ----------
    board_pair_flag = 1.0 if any(v >= 2 for v in bc.values()) else 0.0
    board_trips_flag = 1.0 if any(v >= 3 for v in bc.values()) else 0.0
    # 3. 最长顺子段 只看公共牌
    def _best_run_len(ranks):
        if not ranks:
            return 0
        r = sorted(set(ranks))
        best = cur = 1
        for i in range(1, len(r)):
            if r[i] == r[i - 1] + 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best  # 1..5

    best_run_len = _best_run_len(board_ranks)  # 最长连续段长度
    board_run_len_norm = min(best_run_len, 5) / 5.0  # 归一到 [0,1]

    # 4. max同花数
    dom_cnt = max(sc.values()) if sc else 0  # 某一花色在公共牌中出现最多的张数
    dom_cnt_norm = min(dom_cnt, 5) / 5.0  # [0,1]
    # 6. n_board_norm = n_board / 5.0（公共牌数量）
    n_board_norm = n_board / 5.0  # 0, 0.6, 0.8, 1.0 等
    # 7. board 最大/最小点数归一：board_max/12, board_min/12
    if n_board > 0:
        board_max_norm = board_max / 12.0
        board_min_norm = board_min / 12.0
    else:
        # 没有公共牌时给0.0
        board_max_norm = 0.0
        board_min_norm = 0.0
    # 8. 是否 monotone / two-tone   is_rainbow, is_two_tone, is_monotone
    if n_board == 0:
        is_rainbow = 0.0
        is_two_tone = 0.0
        is_monotone = 0.0
    else:
        distinct_suits = len(sc)
        is_monotone = 1.0 if distinct_suits == 1 else 0.0
        is_two_tone = 1.0 if distinct_suits == 2 else 0.0
        # rainbow 定义为：公共牌至少 3 张且有 3 种或以上花色
        is_rainbow = 1.0 if (n_board >= 3 and distinct_suits >= 3) else 0.0
    # 9. 是否「超级湿面」？
    # 这里给一个比较“人类直觉”的启发式：
    # 满足下列任意一种就认为牌面非常湿：
    #   - 有三张及以上同花（容易成同花 / 强 fd）
    #   - 有长度 >=4 的顺子段（容易成顺 / 强 sd）
    #   - 有对子牌面且同时有两张及以上同花（pair + flush draw，典型大锅）
    #   - 有对子牌面且顺子段长度 >=3（pair + straight draw，多种强牌/听牌）
    is_super_wet = 0.0
    if n_board >= 3:
        if (dom_cnt >= 3) or (best_run_len >= 4):
            is_super_wet = 1.0
        elif board_pair_flag == 1.0 and dom_cnt >= 2:
            is_super_wet = 1.0
        elif board_pair_flag == 1.0 and best_run_len >= 3:
            is_super_wet = 1.0
    # ---------- 汇总成 board-only 特征向量 ----------
    board_only_feats = np.array([
        # 1 & 2: 牌面对子 / 三条或以上
        board_pair_flag,  # 是否至少一对：K77x 之类
        board_trips_flag,  # 是否存在某点数>=3：777 / 7779 / full house 等

        # 3: 顺子纹理
        board_run_len_norm,  # 最长顺子段长度归一：A23x5 之类会很高

        # 4: 同花纹理
        dom_cnt_norm,  # 某花色在牌面上最多张数（3+ 非常危险）

        # 6 : 公共牌数量信息
        n_board_norm,  # 公共牌数量归一到 [0,1]

        # 7: 点数区间位置
        board_max_norm,  # 最高公共牌点数 / 12
        board_min_norm,  # 最低公共牌点数 / 12

        # 8: 花色结构类型
        is_rainbow,  # 三花或以上（无明显同花威胁）
        is_two_tone,  # 两种花色为主（典型有同花听牌）
        is_monotone,  # 全是同花（超 flush heavy）

        # 9: 综合湿度
        is_super_wet,  # 综合判定的 super-wet board
    ], dtype=np.float32)

    # 自己手牌 + 公共牌强度？
    # 1. 成牌层级：
    # 是否命中任意公共牌（一对+）
    any_pair_with_board = 1.0 if (n_board > 0 and ((r1 in board_set) or (r2 in board_set))) else 0.0

    # is_set_or_trips
    is_set_or_trips = 0.0
    if n_board > 0:
        has_board_pair = any(v >= 2 for v in bc.values())
        # 暗三条：口袋对 + 牌面同点
        if is_pair_pp and (r1 in board_set):
            is_set_or_trips = 1.0
        # 明三条：牌面对子 + 你有一张同点
        elif has_board_pair and any((rk in bc and bc[rk] >= 2) for rk in hero_ranks):
            is_set_or_trips = 1.0
    # is_two_pair_with_board 两对（洞牌两张分别命中两个不同的牌面点；排除已经算成 set/trips 的情况）
    is_two_pair_with_board = 0.0
    if n_board > 0 and is_set_or_trips == 0.0:
        if (r1 != r2) and (r1 in board_set) and (r2 in board_set):
            is_two_pair_with_board = 1.0

    # is_overpair
    # 超对（口袋对 > 牌面最大点数）
    is_overpair = 0.0
    if n_board > 0 and is_pair_pp and (max(r1, r2) > board_max):
        is_overpair = 1.0

    # is_top_pair
    # 顶对（命中最高公共牌，且未被更强牌型覆盖）
    is_top_pair = 0.0
    if n_board > 0 and is_set_or_trips == 0.0 and is_two_pair_with_board == 0.0 and is_overpair == 0.0:
        top = board_max
        if (r1 == top) or (r2 == top):
            is_top_pair = 1.0

    # is_second_pair
    # is_bottom_pair
    # 第二对 / 底对（在未成 set/两对的前提下，对齐公共牌第二高 / 最低点）
    is_second_pair = 0.0
    is_bottom_pair = 0.0
    if n_board > 0 and any_pair_with_board == 1.0 and is_set_or_trips == 0.0 and is_two_pair_with_board == 0.0:
        uniq_sorted = sorted(set(board_ranks), reverse=True)
        top = uniq_sorted[0]
        sec = uniq_sorted[1] if len(uniq_sorted) >= 2 else -1
        bot = min(uniq_sorted)
        if (r1 == sec) or (r2 == sec):
            is_second_pair = 1.0
        if (r1 == bot) or (r2 == bot):
            is_bottom_pair = 1.0

    # ---------- 顺子成牌检查：只负责“已经成顺子” ----------
    def _scan_made_straight(S):
        """
        S: 包含 rank 的整数集合（可以是 std 或 wheel 视角）
        返回 1.0 表示至少存在一个 5 连（已经成顺子），否则 0.0
        """
        for lo in range(-1, 9):  # A2345(-1..3) ... TJQKA(8..12)
            window = set(range(lo, lo + 5))
            if len(S & window) == 5:
                return 1.0
        return 0.0
    all_std = set(board_ranks) | hero_set_std
    all_wheel = set(board_set_wheel) | hero_set_wheel
    made_straight = max(_scan_made_straight(all_std), _scan_made_straight(all_wheel))
    # made_flush
    # made_flush：是否已成同花（board + hero 任一花色 >= 5）
    all_suits = board_suits + hero_suits
    suit_counts = Counter(all_suits)
    made_flush = 1.0 if any(v >= 5 for v in suit_counts.values()) else 0.0

    # made_fullhouse
    # made_fullhouse：是否已成葫芦/四条（用所有牌的点数计数）
    all_ranks = board_ranks + hero_ranks
    rc = Counter(all_ranks)
    has_three = any(v >= 3 for v in rc.values())

    # 另一种两张以上的点数（可以是另一个 trips 或 pair）
    has_pair_like = sum(1 for v in rc.values() if v >= 2) >= 2
    has_quads = any(v >= 4 for v in rc.values())
    made_fullhouse = 1.0 if (has_quads or (has_three and has_pair_like)) else 0.0

    # 2. 听牌 / 阻断？

    # 牌面主花色（出现最多的花色）
    if n_board > 0:
        dom_suit, dom_cnt = max(sc.items(), key=lambda kv: kv[1])
    else:
        dom_suit, dom_cnt = 0, 0

    hero_dom = sum(1 for s in hero_suits if s == dom_suit)
    total_dom = dom_cnt + hero_dom

    # 同花听牌
    has_flush_draw = 1.0 if total_dom >= 4 else 0.0

    # 后门同花听（主要在 flop 阶段有意义）
    has_bdfd = 0.0
    if n_board == 3 and has_flush_draw == 0.0:
        # 典型：board 两张同花 + 你一张同花，或者 board 一张同花 + 你两张同花
        if (dom_cnt == 2 and hero_dom >= 1) or (dom_cnt == 1 and hero_suits[0] == hero_suits[1]):
            has_bdfd = 1.0

    # 坚果同花听（你在主花色上的最高牌 >= 牌面主花色最高牌）
    has_nut_fd = 0.0
    if has_flush_draw == 1.0:
        board_dom_ranks = [int(c.rank) for c in state.public_cards if int(c.suit) == dom_suit]
        board_dom_max = max(board_dom_ranks) if board_dom_ranks else -1
        hero_dom_ranks = [rk for rk, s in zip(hero_ranks, hero_suits) if s == dom_suit]
        hero_dom_max = max(hero_dom_ranks) if hero_dom_ranks else -1
        if hero_dom_ranks and hero_dom_max >= board_dom_max:
            has_nut_fd = 1.0

    # 坚果同花阻断：牌面有两张以上同花时，你持有该花色的 A
    has_nut_flush_blocker = 0.0
    if dom_cnt >= 2:
        if any((rk == 12 and s == dom_suit) for rk, s in zip(hero_ranks, hero_suits)):
            has_nut_flush_blocker = 1.0

    # 你在主花色上的持有数归一
    hero_dom_norm = min(hero_dom, 2) / 2.0

    # 顺子听牌相关：OESD / gutshot / 后门顺
    def _scan_oesd_gut(S):
        """
        只统计 miss == 1 的窗口：
          - gutshot: 任何 miss==1 的 5 连窗口都算
          - oesd:    miss==1 且窗口两端至少有一侧可以继续延伸（你原来的条件：left_open or right_open）
        已经成顺子（miss==0）在这里不会触发 oesd/gut。
        """
        oesd = 0.0
        gut = 0.0
        for lo in range(-1, 9):  # A2345 .. TJQKA
            window = set(range(lo, lo + 5))
            miss = 5 - len(S & window)
            if miss == 1:
                # 任何 miss==1 都给 gutshot
                gut = 1.0
                # 至少一端可以延伸，就认为是 open-ended（你原来的定义）
                left_open = (lo - 1 >= 0)
                right_open = (lo + 5 <= 12)
                if left_open or right_open:
                    oesd = 1.0
            # miss == 0 的情况完全交给 _scan_made_straight，不在这里处理
        return oesd, gut

    has_oesd1, has_gut1 = _scan_oesd_gut(all_std)
    has_oesd2, has_gut2 = _scan_oesd_gut(all_wheel)
    has_oesd = max(has_oesd1, has_oesd2)
    has_gutshot = max(has_gut1, has_gut2)

    # 后门顺（flop 上没有显式 OESD / gutshot，但存在 rank 间隔<=2 的两张）
    has_bdsd = 0.0
    if n_board == 3 and has_oesd == 0.0 and has_gutshot == 0.0:
        tri = sorted(all_std)
        for i in range(len(tri)):
            for j in range(i + 1, len(tri)):
                if abs(tri[j] - tri[i]) <= 2:
                    has_bdsd = 1.0
                    break
            if has_bdsd:
                break

    # 直顺端点阻断：牌面存在连续段，你持有该连续段左右的延伸点
    straight_blocker_count = 0
    if n_board > 0:
        br = sorted(set(board_ranks))
        runs = []
        start = br[0]
        for i in range(1, len(br)):
            if br[i] != br[i - 1] + 1:
                runs.append((start, br[i]))
                start = br[i]
        runs.append((start, br[-1]))
        for lo, hi in runs:
            left = lo - 1
            right = hi + 1
            if (left in hero_set_std) or (left == -1 and 12 in hero_set_std):
                straight_blocker_count += 1
            if (right in hero_set_std) or (right == 13 and 12 in hero_set_std):
                straight_blocker_count += 1
    straight_blocker_norm = min(straight_blocker_count, 2) / 2.0  # [0,1]


    # 3. 相对强弱 / 踢脚？
    # kicker_norm
    # rel_over_board_max
    # rel_over_board_min
    # ---------- 3) 相对强弱 / 踢脚 ----------
    kicker_norm = 0.0
    # 若顶对/两对/三条/超对，取“未成对的那张”为踢脚（或超对用口袋对点数）
    if (is_top_pair == 1.0) or (is_two_pair_with_board == 1.0) or (is_set_or_trips == 1.0) or (is_overpair == 1.0):
        if (r1 in board_set) and (r2 not in board_set):
            kicker_norm = r2 / 12.0
        elif (r2 in board_set) and (r1 not in board_set):
            kicker_norm = r1 / 12.0
        elif is_overpair == 1.0 or is_pair_pp == 1.0:
            kicker_norm = max(r1, r2) / 12.0

    # 你的最大洞牌相对牌面最大点数的“超出量”（没有牌面时=1）
    rel_over_board_max = 1.0 if n_board == 0 else max(0.0, (max(r1, r2) - board_max) / 12.0)
    # 与牌面最小点的“高于量”（没有牌面时=1）
    rel_over_board_min = 1.0 if n_board == 0 else max(0.0, (max(r1, r2) - board_min) / 12.0)

    hero_board_combo_feats = np.array([
        # 1) 成牌层级（10 维）
        is_top_pair,
        is_second_pair,
        is_bottom_pair,
        is_overpair,
        is_set_or_trips,
        is_two_pair_with_board,
        any_pair_with_board,
        made_straight,
        made_flush,
        made_fullhouse,

        # 2) 听牌 / 阻断（9 维）
        has_flush_draw,
        has_bdfd,
        has_nut_fd,
        has_nut_flush_blocker,
        hero_dom_norm,
        has_oesd,
        has_gutshot,
        has_bdsd,
        straight_blocker_norm,

        # 3) 相对强弱 / 踢脚（3 维）
        kicker_norm,
        rel_over_board_max,
        rel_over_board_min,
    ], dtype=np.float32)

    # 你最终的牌面信息特征，可以这样拼：
    hero_board_feats = np.concatenate(
        [hero_only_feats, board_only_feats, hero_board_combo_feats],
        axis=0
    ).astype(np.float32)

    assert hero_board_feats.shape[0] == 41, f"hero_board_feats length={hero_board_feats.shape[0]} != 43" # 8 + 11 + 23

    encoded.append(hero_board_feats)

    #logging.info(f'hero_board_feats: {hero_board_feats}')
    # ===== 拼接 & 健康检查 =====
    x = np.concatenate(encoded).astype(np.float32)

    if not np.all(np.isfinite(x)):
        logging.warning('[encode_state] found NaN/Inf in features, auto-fixing via nan_to_num.')
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

    expected = 131 + 41 + 10 * N #174 + 10 N = 234
    assert x.shape[0] == expected, f"encode_state size mismatch: got {x.shape[0]}, expect {expected} (N={N})"
    return x
























