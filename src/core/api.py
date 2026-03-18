import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pokers as pkrs  #本环境是 raise_by模式
from collections import deque
from src.core.model_clean_model_input import PokerNetwork, encode_state, VERBOSE, set_verbose
from src.core.deep_cfr_clean_model_input import DeepCFRAgent
from src.utils.settings import STRICT_CHECKING
from src.utils.logging import log_game_error
import os
import logging
import traceback
import math
import copy
from src.core.scheduler_vince import Vince_scheduler
from src.agents.random_agent_mcts import convert_cards_to_eval, get_mcts_result
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Optional, List, Dict, Any


# ========= 1. 前端请求格式 =========
class StateInput(BaseModel):
    position: int
    # 兼容两种：0~51 的 int，或者 "DK"/"SK" 这类字符串
    hand_cards: List[Union[int, str]]
    board_cards: List[Union[int, str]]
    stage: int
    pot: float
    alive: int
    bet_chips: List[float]
    pot_chips: List[float]
    stake: List[float]
    min_bet: float
    # 协议示例: [fold_ok, check/call_ok, pot_ok, allin_ok, halfpot_ok]
    legal_actions: List[int]
    # 协议示例: [is_fold, is_call_check, is_raise, amount]
    last_action: List[float]

# ========= 2. 牌面解析 =========
# ranks: 2..9,T/X,J,Q,K,A  -> 0..12
RANK_CHAR_TO_IDX = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5,
    "8": 6, "9": 7, "T": 8, "X": 8, "J": 9, "Q": 10, "K": 11, "A": 12
}
SUIT_CHAR_TO_IDX = {"D": 0, "C": 1, "H": 2, "S": 3}

def parse_card_any(v):
    """
    支持两种输入格式:
    1) int 0..51   -> suit = idx // 13, rank = idx % 13
    2) str "DK"/"Ah"/"XT" 这种 -> rank + suit
    返回: (rank_idx[0~12], suit_idx[0~3])
    """
    # 数字：0..51
    if isinstance(v, int):
        if not (0 <= v <= 51):
            raise ValueError(f"card int out of range: {v}")
        suit = v // 13         # 0:♦,1:♣,2:♥,3:♠
        rank = v % 13          # 0..12 对应 2..A
        return rank, suit

    # 字符串："DK","Sk" 之类
    if isinstance(v, str):
        s = v.strip().upper()
        if len(s) != 2:
            raise ValueError(f"card str format invalid: {v}")
        rank_ch, suit_ch = s[0], s[1]
        if rank_ch not in RANK_CHAR_TO_IDX or suit_ch not in SUIT_CHAR_TO_IDX:
            raise ValueError(f"card str unknown rank/suit: {v}")
        return RANK_CHAR_TO_IDX[rank_ch], SUIT_CHAR_TO_IDX[suit_ch]

    raise TypeError(f"unsupported card type: {type(v)}")

# ========= 3. 伪 State / PlayerState，只给 encode_state & agent 用 =========
class MiniCard:
    def __init__(self, rank_idx, suit_idx):
        self.rank = rank_idx   # 0..12
        self.suit = suit_idx   # 0..3


class MiniPlayerState:
    def __init__(self, player, hand, bet, potc, stake, active=True):
        self.player = player
        self.hand = hand              # List[MiniCard]
        self.bet_chips = bet
        self.pot_chips = potc
        self.stake = stake
        self.active = active
        # 下面这些字段 encode_state 不用，就不强求：reward 等


class MiniFromAction:
    def __init__(self, action_enum, amount):
        self.action = type("MiniAction", (), {})()
        self.action.action = action_enum   # pkrs.ActionEnum
        self.action.amount = amount        # float

class MiniState:
    def __init__(self, inp: StateInput):
        N = len(inp.stake)
        self.stage = inp.stage
        self.pot = inp.pot
        self.min_bet = inp.min_bet
        self.button = 0        # 随便给个 BTN，用于位置编码
        self.current_player = inp.position
        self.final_state = False

        # --- 公共牌 ---
        self.public_cards = [
            MiniCard(*parse_card_any(c)) for c in inp.board_cards
        ]

        # --- 玩家状态 ---
        self.players_state = []
        for i in range(N):
            if i == inp.position:
                hand = [MiniCard(*parse_card_any(c)) for c in inp.hand_cards]
            else:
                hand = []  # 对手手牌不公开
            ps = MiniPlayerState(
                player=i,
                hand=hand,
                bet=inp.bet_chips[i],
                potc=inp.pot_chips[i],
                stake=inp.stake[i],
                active=True if inp.stake[i] > 0 else False,
            )
            self.players_state.append(ps)

        # --- legal_actions: 用前端 bit 信息汇总成基础 ActionEnum 集合 ---
        las = []
        if len(inp.legal_actions) >= 1 and inp.legal_actions[0] == 1:
            las.append(pkrs.ActionEnum.Fold)
        if len(inp.legal_actions) >= 2 and inp.legal_actions[1] == 1:
            las.append(pkrs.ActionEnum.Check)
            las.append(pkrs.ActionEnum.Call)
        # 只要有任意一种加注合法，就放一个 Raise 进去，其余细分挡位由 encode_state 自己算
        if any(x == 1 for x in inp.legal_actions[2:]):
            las.append(pkrs.ActionEnum.Raise)
        self.legal_actions = las

        # --- 上一个动作 ---
        la = inp.last_action if inp.last_action is not None else [0, 0, 0, 0.0]
        a_enum = None
        if len(la) >= 3:
            if la[0] == 1:
                a_enum = pkrs.ActionEnum.Fold
                amt = 0.0
            elif la[1] == 1:
                a_enum = pkrs.ActionEnum.Call
                amt = float(la[3]) if len(la) >= 4 else 0.0
            elif la[2] == 1:
                a_enum = pkrs.ActionEnum.Raise
                amt = float(la[3]) if len(la) >= 4 else 0.0

        if a_enum is not None:
            self.from_action = MiniFromAction(a_enum, amt)
        else:
            self.from_action = None

# ========= 4. 初始化 DeepCFRAgent 并加载模型 =========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DeepCFRAgent(player_id=0, num_players=6, device=DEVICE)

MODEL_PATH = "/home/inspur/deepcfr_stable_v11_es/models/self_test/checkpoint_iter_8000.pt"  # 换成你真是的 pt 路径
agent.load_model(MODEL_PATH)      # 用你自己写的 load_model
agent.strategy_net.eval()
agent.advantage_net.eval()

print(f"[INFO] DeepCFR model loaded from {MODEL_PATH} on {DEVICE}")

# ============ 推理逻辑（choose_action 简化版） ============
def infer_action_from_input(inp: StateInput) -> int:
    state = MiniState(inp)
    # 推理时你自己之前喜欢用 sample / argmax 都可以，这里给你留参数
    _, action_type = agent.choose_action(
        state,
        verbose=True,
        selection_mode="argmax",   # 或 "argmax" 看你需要
    )
    return int(action_type)   # 0..5, A_FOLD..A_ALLIN


# ============ FastAPI ==============
app = FastAPI()

@app.post("/api/act")
def api_act(inp: StateInput):
    print(f'request: {inp.hand_cards}')
    action = infer_action_from_input(inp)
    return {"action": action}


if __name__ == "__main__":
    uvicorn.run("src.core.api:app", host="0.0.0.0", port=8000, reload=True)

'''
curl -X POST http://?:8000/api/act -H "Content-Type: application/json" -d '{"position": 0,"hand_cards": [11, 51],"board_cards": [0, 13, 26],"stage": 1,"pot": 100.0,"alive": 3,"bet_chips": [0, 0, 0, 100, 100, 0],"pot_chips": [1, 2, 0, 100, 100, 0],"stake": [199, 198, 200, 100, 100, 0],"min_bet": 50.0,"legal_actions": [1, 1, 1, 0, 0, 1],"last_action": [0, 0, 1, 50]}'
'''