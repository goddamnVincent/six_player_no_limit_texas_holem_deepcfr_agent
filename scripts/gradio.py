import os
import glob
import random
import traceback
from typing import List, Optional, Dict, Tuple
import gradio as gr
import torch
import pokers as pkrs
from src.core.deep_cfr_clean_model_input import DeepCFRAgent
from src.utils import log_game_error
import src.core.deep_cfr_clean_model_input as dcm
import time
import numpy as np
# === matplotlib visuals ===
import matplotlib
import re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# ===================== CONSTANTS (hardcoded config) =====================
MODELS_DIR: Optional[str] = 'models/self_test_v10'
MODEL_PATTERN: str = "*.pt"
NUM_MODELS: int = 5
PLAYER_POS: int = 0
INITIAL_STAKE: float = 200.0
SB: float = 1.0
BB: float = 2.0
SHUFFLE_MODELS: bool = False
STRICT: bool = False

# === image assets ===
CARD_IMG_DIR = "scripts/ui"  # C2.png, DQ.png, etc.
CARD_ASPECT = (92, 128)  # w,h
DPI = 120

stage_names = {
    0: "PreFlop",
    1: "Flop",
    2: "Turn",
    3: "River",
    4: "Showdown"
}

# ===================== Helpers =====================
def action_desc(action: pkrs.Action) -> str:
    if action.action == pkrs.ActionEnum.Fold:
        return "Fold"
    elif action.action == pkrs.ActionEnum.Check:
        return "Check"
    elif action.action == pkrs.ActionEnum.Call:
        return "Call"
    elif action.action == pkrs.ActionEnum.Raise:
        return f"Raise({action.amount:.2f})"
    return f"Unknown({action.action})"

def card_to_str(card: pkrs.Card) -> str:
    # Suit-first: "C2", "DQ", etc. (Ten = 'X')
    suits = {0: "C", 1: "D", 2: "H", 3: "S"}
    ranks = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8",
             7: "9", 8: "X", 9: "J", 10: "Q", 11: "K", 12: "A"}
    return f"{suits[int(card.suit)]}{ranks[int(card.rank)]}"

def card_img_path(card: pkrs.Card) -> Optional[str]:
    fp = os.path.join(CARD_IMG_DIR, f"{card_to_str(card)}.png")
    return fp if os.path.isfile(fp) else None

def select_ordered_models(models_dir: Optional[str], pattern: str, k: int) -> List[str]:
    if not models_dir or not os.path.isdir(models_dir):
        return []
    files = glob.glob(os.path.join(models_dir, pattern))
    if not files:
        return []

    def iter_of(p: str) -> int:
        # 支持 checkpoint_iter_9500.pt 这种命名
        m = re.search(r"checkpoint_iter_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else 10**18  # 没匹配到就丢到最后

    files_sorted = sorted(files, key=iter_of)   # 升序：9500, 9600, ...
    return files_sorted[:min(k, len(files_sorted))]

def seat_role_name(button_pos: int, seat: int) -> str:
    rel = (seat - button_pos) % 6
    mapping = {
        0: "Button",
        1: "SB",
        2: "BB",
        3: "UTG",
        4: "Hijack",
        5: "Cutoff",
    }
    return mapping[rel]

# ===================== Session =====================
class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_actions: Dict[int, str] = {}
        self.models_dir: Optional[str] = MODELS_DIR
        self.model_pattern: str = MODEL_PATTERN
        self.num_models: int = NUM_MODELS
        self.player_pos: int = PLAYER_POS
        self.initial_stake: float = INITIAL_STAKE
        self.sb: float = SB
        self.bb: float = BB
        self.shuffle_models: bool = SHUFFLE_MODELS
        self.strict: bool = STRICT

        self.device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
        dcm.device = torch.device(self.device_str)

        # ======= 累计统计（浏览器刷新才会重置；按钮“重新开始对局”不清零）=======
        if not hasattr(self, "cum_profit"):
            self.cum_profit: List[float] = [0.0] * 6   # 每个 seat 的累计 reward
        if not hasattr(self, "hands_completed"):
            self.hands_completed: int = 0
        if not hasattr(self, "last_settled_game"):
            self.last_settled_game: int = -1   # 用于防止同一局重复结算

        self.num_games: int = 0
        self.total_profit: float = 0.0
        self.player_bankroll: float = self.initial_stake

        self.current_state: Optional[pkrs.State] = None
        self.agents: List[Optional[DeepCFRAgent]] = [None] * 6
        self.model_paths_current: List[str] = []

        self.initial_hands: Dict[int, Tuple[pkrs.Card, pkrs.Card]] = {}
        self.log_lines: List[str] = []
        self.button_pos: Optional[int] = None  # set at start of hand
        self.last_actions = {}
        self.last_result_html: str = ""  # <— 新增：结算展示区内容

    def log(self, msg: str):
        self.log_lines.append(msg)

    def dump_log(self) -> str:
        return "\n".join(self.log_lines[-800:])

# ===================== Core logic (aligned with play_clean) =====================
def build_agents(sess: Session):
    sess.agents = []
    for i in range(6):
        if i == sess.player_pos:
            sess.agents.append(None)  # human
            continue

        model_idx = (i - 1) if i > sess.player_pos else i
        if sess.models_dir and model_idx < len(sess.model_paths_current):
            path = sess.model_paths_current[model_idx]
            try:
                agent = DeepCFRAgent(player_id=i, num_players=6, device=sess.device_str)
                agent.reset_for_new_hand()
                agent.load_model(path)
                agent.advantage_net.eval()
                agent.strategy_net.eval()
                sess.agents.append(agent)
                sess.log(f"Loaded model for Player {i}: {os.path.basename(path)}")
            except Exception as e:
                sess.log(f"Model load failed for P{i}: {e}")
        else:
            sess.log(f"没有机器人，玩不了了")

def _advance_ai_until_player_or_end(sess: Session):
    s = sess.current_state  # state
    pid = sess.player_pos  # 0 就是玩家
    if s is None:
        return
    while (not s.final_state) and s.current_player != pid:
        ai = sess.agents[s.current_player]  #第 current_player 个 AI deep CFR agent
        try:
            act, _ = ai.choose_action(s, verbose = True, selection_mode= 'argmax')  #act 选出动作
        except Exception as e:
            sess.log(f"AI choose_action error @P{s.current_player}: {e}")
            break
        last = action_desc(act) # 什么动作
        sess.last_actions[s.current_player] = last  #一个列表用于存放动作
        sess.log(f"➔ P{s.current_player} 选择: {last} ")
        ns = s.apply_action(act)
        if ns.status != pkrs.StateStatus.Ok: #记录
            log_file = log_game_error(s, act, f"State not OK: {ns.status}")
            if sess.strict:
                raise ValueError(f"State not OK: {ns.status}; details: {log_file}")
            sess.log(f"WARNING: State not OK: {ns.status}; details: {log_file}")
            break
        s = ns
    sess.current_state = s
    #轮到你玩了
    # Log stage + hints
    stage_name = stage_names.get(int(sess.current_state.stage), str(sess.current_state.stage))
    sess.log(f"\n====== 轮到你了 ======")
    sess.log(f"当前阶段: {stage_name}") #没问题 stage
    community_cards = " ".join(card_to_str(c) for c in sess.current_state.public_cards)
    sess.log(f"公共牌: {community_cards if community_cards else '暂无'}") #没问题 公共牌
    player_state = sess.current_state.players_state[PLAYER_POS]
    current_bet = float(player_state.bet_chips)
    available_stake = float(player_state.stake)
    call_amount = max(0.0, float(sess.current_state.min_bet) - current_bet)
    remaining_stake = available_stake - call_amount
    pot_now = float(sess.current_state.pot)

    half_pot_raise = max(0.0, 0.5 * pot_now)
    full_pot_raise = max(0.0, 1.0 * pot_now)

    sess.log(f'可用筹码数量：{available_stake:.2f}')
    sess.log(f'当前call所需要的筹码数量：{call_amount:.2f}')
    sess.log(f'legal_actions:')

    for action_enum in sess.current_state.legal_actions:
        if action_enum == pkrs.ActionEnum.Fold:
            sess.log(" Fold")
        elif action_enum == pkrs.ActionEnum.Check:
            sess.log(" Check")
        elif action_enum == pkrs.ActionEnum.Call:
            sess.log(f" Call ${call_amount:.2f}")
        elif action_enum == pkrs.ActionEnum.Raise:
            if remaining_stake <= 0.0:
                sess.log("  （无法 Raise：跟注后无余筹）")
                continue

            can_half = (half_pot_raise <= remaining_stake + 1e-6)
            can_full = (full_pot_raise <= remaining_stake + 1e-6)

            msg_parts = []
            if can_half:
                msg_parts.append(f"半池≈{half_pot_raise:.2f}")
            if can_full:
                msg_parts.append(f"一池≈{full_pot_raise:.2f}")
            msg_parts.append(f"All-in≤{remaining_stake:.2f}")

            sess.log(" Raise 挡位可选: " + " | ".join(msg_parts))
def _settle(sess: Session):
    s = sess.current_state
    if s is None or not s.final_state:
        return
    # 防止同一局被重复结算（有些流程可能会多次触发 _settle）
    if getattr(sess, "last_settled_game", -1) == sess.num_games:
        return
    sess.last_settled_game = sess.num_games

    # 统计：累计每个玩家的收益（reward）
    if not hasattr(sess, "cum_profit"):
        sess.cum_profit = [0.0] * 6
    for i, ps in enumerate(s.players_state):
        sess.cum_profit[i] += float(ps.reward)

    if not hasattr(sess, "hands_completed"):
        sess.hands_completed = 0
    sess.hands_completed += 1

    board = " ".join(card_to_str(c) for c in s.public_cards)
    sess.log("\n--- Game Over ---")
    sess.log(f"Board: {board}")
    sess.log("Final hands (including folded):")
    for i in range(6):
        cards = sess.initial_hands.get(i, ())
        hand_str = " ".join(card_to_str(c) for c in cards) if cards else "N/A"
        sess.log(f"Player {i}: {hand_str}")
    sess.log("\nResults:")
    for i, p in enumerate(s.players_state):
        who = "YOU" if i == sess.player_pos else "AI"
        sess.log(f"Player {i} ({who}): ${p.reward:.2f}")
    prof = s.players_state[sess.player_pos].reward
    sess.total_profit += prof
    sess.player_bankroll += prof

    # 找出赢家们（reward > 0）
    winners = [(i, float(p.reward)) for i, p in enumerate(s.players_state) if float(p.reward) > 0.0]
    losers  = [(i, float(p.reward)) for i, p in enumerate(s.players_state) if float(p.reward) < 0.0]

    if winners:
        win_strs = [f"Player {i} ({'YOU' if i == sess.player_pos else 'AI'})"
                    for i, _ in winners]
        sess.log("\n赢家: " + ", ".join(win_strs))
    if losers:
        lose_strs = [f"Player {i} ({'YOU' if i == sess.player_pos else 'AI'})"
                     for i, _ in losers]
        sess.log("输家: " + ", ".join(lose_strs))

    if winners:
        win_lines = []
        for i, amt in winners:
            tag = "YOU" if i == sess.player_pos else "AI"
            win_lines.append(f"<span style='color:#16a34a;font-weight:700;'>P{i} ({tag}) +${amt:.2f}</span>")
        sess.last_result_html = "🏁 结算结果：<br>" + "<br>".join(win_lines)
    else:
        sess.last_result_html = "🏁 结算结果：<br><span style='color:#555;'>平分或无正盈利玩家</span>"

    sess.log(f"\n这是第{sess.num_games}局游戏")
    outcome = "Won" if prof > 0 else ("Lost" if prof < 0 else "Draw")
    sess.log(f"This game: {outcome} ${abs(prof):.2f}")
    sess.log(f"你总共赢得: ${sess.total_profit:.2f} 筹码")
    sess.log(f"目前总共有: ${sess.player_bankroll:.2f} 筹码")

# ===================== Matplotlib visuals (return PILs) =====================
def _pil_or_placeholder(card: Optional[pkrs.Card]) -> Image.Image:
    """Load card image, or a blank placeholder."""
    W, H = CARD_ASPECT
    if card is None:
        return Image.new("RGB", (W, H), (240, 240, 240))
    p = card_img_path(card)
    if p and os.path.isfile(p):
        return Image.open(p).convert("RGB")
    return Image.new("RGB", (W, H), (240, 240, 240))

def ai_status_texts(sess: Session) -> List[str]:
    """返回 5 个 AI 窗口的状态文本：P{seat} | 位置 | 动作 | 已下 | 剩余"""
    texts: List[str] = []
    s = sess.current_state
    btn = sess.button_pos if sess.button_pos is not None else 0
    ai_seats = [i for i in range(6) if i != sess.player_pos]

    for seat in ai_seats[:5]:
        role = seat_role_name(btn, seat)
        act  = sess.last_actions.get(seat, "—")
        # 取该座位本轮已下筹码与剩余筹码（若还没发牌/无状态则为 0）
        bet   = 0.0
        stake = 0.0
        if s is not None and 0 <= seat < len(s.players_state):
            ps = s.players_state[seat]
            bet   = float(getattr(ps, "bet_chips", 0.0) or 0.0)
            stake = float(getattr(ps, "stake", 0.0) or 0.0)

        line1 = f"P{seat}|:{role}|:{act}"
        line2 = f"本轮已下:{bet:.2f}|剩余:{stake:.2f}"
        line = f"<div class='l1'><b>{line1}</b></div><div class='l2'><b>{line2}</b></div>"
        if act.lower() == 'fold':
            line = f"<div class='l1'>{line1}</div><div class='l2'>{line2}</div>"
        if role == "SB" and act.lower() != 'fold':
            line = f'<span style="color: blue; font-weight: bold;">' \
                   f'{line1}<br>{line2}' \
                   f'</span>'
        elif role == "SB" and act.lower() == 'fold':
            f"<div class='l1'>{line1}</div><div class='l2'>{line2}</div>"
        elif role == "BB" and act.lower() != 'fold':
            line = f'<span style="color: red; font-weight: bold;">' \
                   f'{line1}<br>{line2}' \
                   f'</span>'
        elif role == "BB" and act.lower() == 'fold':
            f"<div class='l1'>{line1}</div><div class='l2'>{line2}</div>"
        texts.append(line)

    while len(texts) < 5:
        texts.append(" ")
    return texts

def my_status_text(sess: Session) -> str:
    btn = sess.button_pos if sess.button_pos is not None else 0
    role = seat_role_name(btn, sess.player_pos)
    act  = sess.last_actions.get(sess.player_pos, "—")

    bet   = 0.0
    stake = 0.0
    if sess.current_state is not None:
        ps = sess.current_state.players_state[sess.player_pos]
        bet   = float(getattr(ps, "bet_chips", 0.0) or 0.0)
        stake = float(getattr(ps, "stake", 0.0) or 0.0)

    # 第一行：身份、动作
    line1 = f"P{sess.player_pos}|:{role}|:{act}"
    # 第二行：筹码信息
    line2 = f"本轮已下:{bet:.2f}|剩余:{stake:.2f}"

    return (
        f'<span style="background-color: yellow; color: black; font-weight: bold; padding:2px;">'
        f"{line1}<br>{line2}"
        f'</span>'
    )

def session_stats_md(sess: Session) -> str:
    """统计窗口：每个 seat 的累计收益 / 对局数 / 平均收益（刷新浏览器才重置）"""
    n = int(getattr(sess, "hands_completed", 0) or 0)
    profits = getattr(sess, "cum_profit", [0.0] * 6)

    lines = []
    lines.append(f"### 累计统计（已完成对局：**{n}**）")
    lines.append("")
    lines.append("| Player | Total Profit | Avg / Hand |")
    lines.append("|---:|---:|---:|")
    for i in range(6):
        total = float(profits[i])
        avg = (total / n) if n > 0 else 0.0
        tag = "YOU" if i == sess.player_pos else "AI"
        lines.append(f"| P{i} ({tag}) | {total:+.2f} | {avg:+.2f} |")
    return "\n".join(lines)

def render_board_pil(sess: Session) -> Image.Image:
    """Compose a pure board view (community only) -> PIL image."""
    s = sess.current_state
    fig_w = 5 * CARD_ASPECT[0] / DPI + 1.0
    fig_h = 1 * CARD_ASPECT[1] / DPI + 0.8     # 只一行
    fig, axes = plt.subplots(1, 5, figsize=(fig_w, fig_h), dpi=DPI)
    for ax in (axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]):
        ax.axis("off")

    if s is None:
        fig.suptitle("No game", y=0.98)
    else:
        stage = int(s.stage)
        reveal = 0
        if stage >= 1: reveal = 3   # Flop
        if stage >= 2: reveal = 4   # Turn
        if stage >= 3: reveal = 5   # River

        board_cards = list(s.public_cards[:reveal])
        while len(board_cards) < 5:
            board_cards.append(None)

        for i in range(5):
            axes[i].imshow(_pil_or_placeholder(board_cards[i]))

        btn = sess.button_pos if sess.button_pos is not None else 0
        title = f"Stage: {stage_names.get(stage, str(stage))} | BTN=P{btn} | You=P{sess.player_pos}"
        fig.suptitle(title, y=0.99, fontsize=14)
        fig.text(
            0.5, 0.86,  # x,y 坐标 (相对 figure, 0~1)
            f"Pot = ${s.pot:.2f}",  # 显示内容
            ha="center", va="top",  # 居中对齐
            fontsize=18, color="red", weight="bold",  # 超大字号 + 鲜红色 + 粗体
        )

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img

def render_my_hand_pil(sess: Session, title: Optional[str] = None) -> Image.Image:
    """和 AI 小窗一致的我的手牌小窗（两张底牌）。"""
    W, H = CARD_ASPECT
    fig_w = 2 * W / DPI + 0.8
    fig_h = H / DPI + 0.8
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=DPI)
    for ax in axes.ravel():
        ax.axis("off")

    s = sess.current_state
    cards = []
    if s is not None:
        try:
            cards = list(getattr(s.players_state[sess.player_pos], "hand", []))
        except Exception:
            cards = []
    while len(cards) < 2:
        cards.append(None)

    axes[0].imshow(_pil_or_placeholder(cards[0]))
    axes[1].imshow(_pil_or_placeholder(cards[1]))

    if title is None:
        btn = sess.button_pos if sess.button_pos is not None else 0
        role = seat_role_name(btn, sess.player_pos)
        title = f"P{sess.player_pos} {role}"

    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img

def compose_two_cards_panel(seat_id: int, c1: Optional[pkrs.Card], c2: Optional[pkrs.Card], title: Optional[str] = None) -> Image.Image:
    """Small panel: label row + two cards side-by-side."""
    W, H = CARD_ASPECT[0] - 20, CARD_ASPECT[1] - 28
    pad = 4
    label_h = 20
    canvas_w = W * 2 + pad * 3
    canvas_h = H + pad * 2 + label_h
    bg = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(bg)
    # label
    draw.rectangle([(0, 0), (canvas_w, label_h)], fill=(230, 230, 230))
    if title is None:
        title = f"P{seat_id} 手牌"
    draw.text((10, 3), title, fill=(20, 20, 20))
    # cards
    bg.paste(_pil_or_placeholder(c1), (pad, label_h + pad))
    bg.paste(_pil_or_placeholder(c2), (pad * 2 + W, label_h + pad))
    return bg

def render_showdown_pil(sess: Session, seat_id: int, title: Optional[str] = None) -> Image.Image:
    """Matplotlib 风格的小窗：显示某个座位的两张底牌。返回 PIL。"""
    W, H = CARD_ASPECT
    fig_w = 2 * W / DPI + 0.8
    fig_h = H / DPI + 0.8
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=DPI)
    for ax in axes.ravel():
        ax.axis("off")

    # 获取该座位牌
    cards = list(sess.initial_hands.get(seat_id, ())) if sess.initial_hands.get(seat_id) else []
    while len(cards) < 2:
        cards.append(None)

    # 左右各一张牌
    try:
        axes[0].imshow(_pil_or_placeholder(cards[0]))
        axes[1].imshow(_pil_or_placeholder(cards[1]))
    except Exception:
        # 兜底（某些环境 axes 不是可下标对象时）
        for i, c in enumerate(cards[:2]):
            axes.imshow(_pil_or_placeholder(c))

    if title is None:
        btn = sess.button_pos if sess.button_pos is not None else 0
        role = seat_role_name(btn, seat_id)
        title = f"P{seat_id} {role}"
    fig.suptitle(title, y=0.98)

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img

def ai_showdown_panels(sess: Session) -> List[Optional[Image.Image]]:
    """Showdown 时用 matplotlib 风格生成 5 个小窗口（每个 AI 一个）。其他阶段返回 [None]*5。"""
    s = sess.current_state
    panels: List[Optional[Image.Image]] = [None, None, None, None, None]
    if s is None or not s.final_state:
        return panels
    ai_seats = [i for i in range(6) if i != sess.player_pos]
    btn = sess.button_pos if sess.button_pos is not None else 0
    for idx, seat in enumerate(ai_seats[:5]):
        role = seat_role_name(btn, seat)
        title = f"P{seat} {role}"
        panels[idx] = render_showdown_pil(sess, seat, title=title)
    return panels

def ai_hand_panels(sess: Session) -> List[Optional[Image.Image]]:
    """
    调试用：在任意阶段都显示 5 个 AI 的两张底牌（从 initial_hands 里拿），
    而不是只在 showdown 展示。
    """
    s = sess.current_state
    panels: List[Optional[Image.Image]] = [None, None, None, None, None]
    if s is None:
        return panels

    # 除了玩家自己之外的 5 个座位
    ai_seats = [i for i in range(6) if i != sess.player_pos]
    btn = sess.button_pos if sess.button_pos is not None else 0

    for idx, seat in enumerate(ai_seats[:5]):
        role = seat_role_name(btn, seat)
        title = f"P{seat} {role}"
        # 这里仍然复用 render_showdown_pil，它本来就是从 sess.initial_hands 取两张底牌
        panels[idx] = render_showdown_pil(sess, seat, title=title)

    return panels

def visuals(sess: Session):
    """Return (board_pil, my_pil, ai1, ai2, ai3, ai4, ai5)."""
    board = render_board_pil(sess)
    my_pil = render_my_hand_pil(sess)
    # 显示 或者 不显示 AI手牌 的开关
    ai5 = ai_showdown_panels(sess)
    #ai5 = ai_hand_panels(sess)
    return (board, my_pil, *ai5)

def compute_action_enables(sess: Session):
    """
    根据当前状态，统一计算 6 个按钮是否可点击：
      can_fold, can_cc, can_r_half, can_r_full, can_r_allin
    规则：
      - 只有轮到你、且局面未结束时才可能为 True
      - Raise 相关要同时满足：牌面允许 Raise + 你的筹码在 call 之后还能支持该挡位
      - All-in 只要 call 后还有筹码，就允许
    """
    import logging
    s = sess.current_state
    # 默认全部不可点
    can_fold = can_cc = False
    can_r_half = can_r_full = can_r_allin = False

    if s is None or s.final_state or s.current_player != sess.player_pos:
        return can_fold, can_cc, can_r_half, can_r_full, can_r_allin

    legal = list(s.legal_actions)
    # Fold / CC 判定
    can_fold = (pkrs.ActionEnum.Fold in legal)
    can_cc = (pkrs.ActionEnum.Check in legal) or (pkrs.ActionEnum.Call in legal)
    # Log current stage and legal actions
    # logging.info(f"Current stage: {s.stage}")
    # logging.info(f"Legal actions: {s.legal_actions}")
    # Raise 挡位判定
    if pkrs.ActionEnum.Raise in legal:
        ps = s.players_state[sess.player_pos]
        current_bet = float(ps.bet_chips)
        available = float(ps.stake)
        call_amount = max(0.0, float(s.min_bet) - current_bet)
        remain_after_call = available - call_amount  # 跟注之后还能加多少

        if remain_after_call > 1e-6:
            pot = float(s.pot)
            half = max(0.0, 0.5 * pot)
            full = max(0.0, 1.0 * pot)

            # 只有“该挡位金额 <= remain_after_call”才允许点
            can_r_half = (half <= remain_after_call + 1e-6)
            can_r_full = (full <= remain_after_call + 1e-6)

            # All-in 只要剩余 > 0 就合法
            can_r_allin = True

    return can_fold, can_cc, can_r_half, can_r_full, can_r_allin

def start_new_game(sess: Session):
    sess.log(">>> 开始一局新游戏")
    #ai_texts = ai_status_texts(sess)  # 移到函数开头
    sess.last_actions = {}
    if sess.player_bankroll <= 0:
        sess.log("⚠  你输光了筹码，已经自动重置游戏环境，重新开始...")
        sess.reset()
        sess.log_lines = []
        board_pil, my_pil, ai1, ai2, ai3, ai4, ai5 = visuals(sess)
        ai_texts = ai_status_texts(sess)
        return (sess.dump_log(), '<span style="color:red;">你输光了，重新开始吧.</span>', '',
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), #四个挡位
                board_pil, my_status_text(sess), my_pil, ai_texts[0], ai1, ai_texts[1], ai2, ai_texts[2], ai3, ai_texts[3], ai4, ai_texts[4], ai5, session_stats_md(sess))
        #    return (sess.dump_log(), state_text, result_html, gr.update(interactive=can_fold), gr.update(interactive=can_cc),
        #    gr.update(interactive=can_raise),
        #    board_pil, my_status_text(sess), my_pil, ai_texts[0], ai1, ai_texts[1], ai2, ai_texts[2], ai3, ai_texts[3], ai4, ai_texts[4], ai5)
    # ✅ 只在第一局随机抽一次模型，后续局数复用
    if sess.num_games == 0 and sess.models_dir:
        sess.model_paths_current = select_ordered_models(
            sess.models_dir, sess.model_pattern, sess.num_models
        )
        sess.log(f"random picked {len(sess.model_paths_current)} model(s) for this session:")
        for i, pth in enumerate(sess.model_paths_current):
            sess.log(f"  Model {i+1}: {os.path.basename(pth)}")
    elif not sess.model_paths_current and sess.models_dir:
        # 兜底：如果因为某些原因列表空了，再抽一次，避免“没有机器人”
        sess.model_paths_current = select_ordered_models(
            sess.models_dir, sess.model_pattern, sess.num_models
        )
        sess.log(f"[fallback] picked {len(sess.model_paths_current)} model(s):")
        for i, pth in enumerate(sess.model_paths_current):
            sess.log(f"  Model {i+1}: {os.path.basename(pth)}")
    build_agents(sess)

    sess.num_games += 1
    button_pos = (sess.num_games - 1) % 6
    sess.button_pos = button_pos  # store for visuals/labels
    lines = [
        f"小盲位(SB):P{(button_pos + 1) % 6}",
        f"大盲位(BB):P{(button_pos + 2) % 6}",
        f"枪口位(UTG):P{(button_pos + 3) % 6}",
        f"劫持位(Hijack):P{(button_pos + 4) % 6}",
        f"关煞位(Cutoff):P{(button_pos + 5) % 6}",
        f"庄家位(Button):P{button_pos}",
    ]
    sess.log("当前位置分布：\n" + "\n".join(lines))
    my_role = seat_role_name(button_pos, sess.player_pos)
    sess.log(f"你的位置: P{sess.player_pos} — {my_role}")
    sess.log(f'小盲位置玩家P{(button_pos+1)%6} -> bet(1.00)\n大盲位置玩家P{(button_pos+2)%6} -> bet(2.00)')

    try:
        s = pkrs.State.from_seed(
            n_players=6,
            button=button_pos,
            sb=sess.sb,
            bb=sess.bb,
            stake=sess.initial_stake,
            seed=random.randint(0, 10000)
        )
    except Exception as e:
        sess.log(f"Failed to create game: {e}\n{traceback.format_exc()}")
        board_pil, my_pil, ai1, ai2, ai3, ai4, ai5 = visuals(sess)
        return (sess.dump_log(), "初始化失败了？", '',
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False),
                board_pil, my_status_text(sess), my_pil, ai_texts[0], ai1, ai_texts[1], ai2, ai_texts[2], ai3, ai_texts[3], ai4, ai_texts[4], ai5, session_stats_md(sess))

    sess.current_state = s
    sess.initial_hands = {}
    for i, ps in enumerate(s.players_state):
        try:
            cards = list(getattr(ps, "hand", []))
        except Exception:
            cards = []
        sess.initial_hands[i] = tuple(cards)

    _advance_ai_until_player_or_end(sess)

    s = sess.current_state
    if s and not s.final_state:
        you = s.players_state[sess.player_pos]
        your_hand = " ".join(card_to_str(c) for c in getattr(you, "hand", []))
        state_text = f"<span style='color:red;'>轮到你了.</span>. Hand: {your_hand}. Pot=${s.pot:.2f}."
        # 统一用 compute_action_enables
        can_fold, can_cc, can_r_half, can_r_full, can_r_allin = compute_action_enables(sess)
        result_html = ""
    else:
        _settle(sess)
        state_text = "<span style='color:red;'>这把结束了.</span>"
        can_fold = can_cc = can_r_half = can_r_full = can_r_allin = False
        result_html = sess.last_result_html

    board_pil, my_pil, ai1, ai2, ai3, ai4, ai5 = visuals(sess)
    ai_texts = ai_status_texts(sess)
    return (sess.dump_log(), state_text, result_html,
            gr.update(interactive=can_fold),
            gr.update(interactive=can_cc),
            gr.update(interactive=can_r_half),
            gr.update(interactive=can_r_full),
            gr.update(interactive=can_r_allin),
            board_pil, my_status_text(sess), my_pil,
            ai_texts[0], ai1, ai_texts[1], ai2,
            ai_texts[2], ai3, ai_texts[3], ai4,
            ai_texts[4], ai5, session_stats_md(sess))

def _apply_and_progress(sess: Session, act: pkrs.Action):
    s = sess.current_state
    if s is None:
        return
    you_last = action_desc(act)
    sess.last_actions[sess.player_pos] = you_last
    sess.log(f"You -> {you_last}")
    ns = s.apply_action(act)
    if ns.status != pkrs.StateStatus.Ok:
        log_file = log_game_error(s, act, f"State not OK: {ns.status}")
        if sess.strict:
            raise ValueError(f"State not OK: {ns.status}; details: {log_file}")
        sess.log(f"WARNING: State not OK: {ns.status}; details: {log_file}")
    sess.current_state = ns
    _advance_ai_until_player_or_end(sess)
    _settle(sess)


def on_fold(sess: Session):
    sess.log(">>> on_fold()")
    s = sess.current_state
    if s and s.current_player == sess.player_pos:
        legal = list(s.legal_actions)
        if pkrs.ActionEnum.Fold in legal:
            _apply_and_progress(sess, pkrs.Action(pkrs.ActionEnum.Fold))

    s = sess.current_state
    if s and not s.final_state:
        result_html = ""
        you = s.players_state[sess.player_pos]
        your_hand = " ".join(card_to_str(c) for c in getattr(you, "hand", []))
        state_text = f"<span style='color:red;'>轮到你了.</span>. Hand: {your_hand}. Pot=${s.pot:.2f}."
        can_fold, can_cc, can_r_half, can_r_full, can_r_allin = compute_action_enables(sess)
    else:
        result_html = sess.last_result_html
        state_text = "<span style='color:red;'>这把结束了.</span>"
        can_fold = can_cc = can_r_half = can_r_full = can_r_allin = False

    board_pil, my_pil, ai1, ai2, ai3, ai4, ai5 = visuals(sess)
    ai_texts = ai_status_texts(sess)
    return (sess.dump_log(), state_text, result_html,
            gr.update(interactive=can_fold),
            gr.update(interactive=can_cc),
            gr.update(interactive=can_r_half),
            gr.update(interactive=can_r_full),
            gr.update(interactive=can_r_allin),
            board_pil, my_status_text(sess), my_pil,
            ai_texts[0], ai1, ai_texts[1], ai2,
            ai_texts[2], ai3, ai_texts[3], ai4,
            ai_texts[4], ai5, session_stats_md(sess))


def on_check_call(sess: Session):
    sess.log(">>> on_check_call()")
    s = sess.current_state
    if s and s.current_player == sess.player_pos:
        legal = list(s.legal_actions)
        if pkrs.ActionEnum.Check in legal:
            _apply_and_progress(sess, pkrs.Action(pkrs.ActionEnum.Check))
        elif pkrs.ActionEnum.Call in legal:
            _apply_and_progress(sess, pkrs.Action(pkrs.ActionEnum.Call))

    s = sess.current_state
    if s and not s.final_state:
        result_html = ""
        you = s.players_state[sess.player_pos]
        your_hand = " ".join(card_to_str(c) for c in getattr(you, "hand", []))
        state_text = f"<span style='color:red;'>轮到你了.</span>. Hand: {your_hand}. Pot=${s.pot:.2f}."
        can_fold, can_cc, can_r_half, can_r_full, can_r_allin = compute_action_enables(sess)
    else:
        result_html = sess.last_result_html
        state_text = "<span style='color:red;'>这把结束了.</span>"
        can_fold = can_cc = can_r_half = can_r_full = can_r_allin = False

    board_pil, my_pil, ai1, ai2, ai3, ai4, ai5 = visuals(sess)
    ai_texts = ai_status_texts(sess)
    return (sess.dump_log(), state_text, result_html,
            gr.update(interactive=can_fold),
            gr.update(interactive=can_cc),
            gr.update(interactive=can_r_half),
            gr.update(interactive=can_r_full),
            gr.update(interactive=can_r_allin),
            board_pil, my_status_text(sess), my_pil,
            ai_texts[0], ai1, ai_texts[1], ai2,
            ai_texts[2], ai3, ai_texts[3], ai4,
            ai_texts[4], ai5, session_stats_md(sess))

def on_raise_half_pot_amount(sess: Session):
    sess.log('>>>raise half pot amount')
    s = sess.current_state #state
    if s and s.current_player == sess.player_pos:
        legal = list(s.legal_actions)
        if pkrs.ActionEnum.Raise in legal:
            pid = sess.player_pos
            pstate = s.players_state[pid]
            current_bet = pstate.bet_chips
            available = pstate.stake
            call_amount = max(0.0, s.min_bet - current_bet)
            remain_after_call = available - call_amount
            half_pot_amount = 0.5 * s.pot
            if remain_after_call <= 1e-6:
                sess.log("⚠ 跟注后已无多余筹码，无法半池加注。")
            elif half_pot_amount > remain_after_call + 1e-6:
                sess.log(f"⚠ 筹码不足以支持半池加注(需要≈{half_pot_amount:.2f}，剩余≈{remain_after_call:.2f})，操作已忽略。")
            else:
                sess.log(f"你选择加注，加注额度为半池 [{half_pot_amount:.2f}]")
                act = pkrs.Action(pkrs.ActionEnum.Raise, half_pot_amount)
                _apply_and_progress(sess, act)
    s = sess.current_state
    if s and not s.final_state:
        result_html = ""
        you = s.players_state[sess.player_pos]
        your_hand = " ".join(card_to_str(c) for c in getattr(you, "hand", []))
        state_text = f"<span style='color:red;'>轮到你了.</span>. Hand: {your_hand}. Pot=${s.pot:.2f}."
        can_fold, can_cc, can_r_half, can_r_full, can_r_allin = compute_action_enables(sess)
    else:
        result_html = sess.last_result_html
        state_text = "<span style='color:red;'>这把结束了.</span>"
        can_fold = can_cc = can_r_half = can_r_full = can_r_allin = False

    board_pil, my_pil, ai1, ai2, ai3, ai4, ai5 = visuals(sess)
    ai_texts = ai_status_texts(sess)
    return (sess.dump_log(), state_text, result_html,
            gr.update(interactive=can_fold),
            gr.update(interactive=can_cc),
            gr.update(interactive=can_r_half),
            gr.update(interactive=can_r_full),
            gr.update(interactive=can_r_allin),
            board_pil, my_status_text(sess), my_pil,
            ai_texts[0], ai1, ai_texts[1], ai2,
            ai_texts[2], ai3, ai_texts[3], ai4,
            ai_texts[4], ai5, session_stats_md(sess))

def on_raise_full_pot_amount(sess: Session):
    sess.log('>>>raise full pot amount')
    s = sess.current_state #state
    if s and s.current_player == sess.player_pos:
        legal = list(s.legal_actions)
        if pkrs.ActionEnum.Raise in legal:
            pid = sess.player_pos
            pstate = s.players_state[pid]
            current_bet = pstate.bet_chips
            available = pstate.stake
            call_amount = max(0.0, s.min_bet - current_bet)
            full_pot_amount = 1.0 * s.pot
            remain_after_call = available - call_amount
            if remain_after_call <= 1e-6:
                sess.log("⚠ 跟注后已无多余筹码，无法满池加注。")
            elif full_pot_amount > remain_after_call + 1e-6:
                sess.log(f"⚠ 筹码不足以支持满池加注(需要≈{full_pot_amount:.2f}，剩余≈{remain_after_call:.2f})，操作已忽略。")
            else:
                sess.log(f"你选择加注，加注额度为满池 [{full_pot_amount:.2f}]")
                act = pkrs.Action(pkrs.ActionEnum.Raise, full_pot_amount)
                _apply_and_progress(sess, act)

    s = sess.current_state
    if s and not s.final_state:
        result_html = ""
        you = s.players_state[sess.player_pos]
        your_hand = " ".join(card_to_str(c) for c in getattr(you, "hand", []))
        state_text = f"<span style='color:red;'>轮到你了.</span>. Hand: {your_hand}. Pot=${s.pot:.2f}."
        can_fold, can_cc, can_r_half, can_r_full, can_r_allin = compute_action_enables(sess)
    else:
        result_html = sess.last_result_html
        state_text = "<span style='color:red;'>这把结束了.</span>"
        can_fold = can_cc = can_r_half = can_r_full = can_r_allin = False

    board_pil, my_pil, ai1, ai2, ai3, ai4, ai5 = visuals(sess)
    ai_texts = ai_status_texts(sess)
    return (sess.dump_log(), state_text, result_html,
            gr.update(interactive=can_fold),
            gr.update(interactive=can_cc),
            gr.update(interactive=can_r_half),
            gr.update(interactive=can_r_full),
            gr.update(interactive=can_r_allin),
            board_pil, my_status_text(sess), my_pil,
            ai_texts[0], ai1, ai_texts[1], ai2,
            ai_texts[2], ai3, ai_texts[3], ai4,
            ai_texts[4], ai5, session_stats_md(sess))

def on_raise_all_in_amount(sess: Session):
    sess.log('>>>raise all in amount')
    s = sess.current_state #state
    if s and s.current_player == sess.player_pos:
        legal = list(s.legal_actions)
        if pkrs.ActionEnum.Raise in legal:
            pid = sess.player_pos
            pstate = s.players_state[pid]
            current_bet = pstate.bet_chips
            available = pstate.stake
            call_amount = max(0.0, s.min_bet - current_bet)
            remain_after_call = available - call_amount
            all_in_amount = max(0.0, remain_after_call)
            if all_in_amount <= 1e-6:
                sess.log("⚠ 已无可加注筹码，无法 all-in。")
            else:
                sess.log(f"你选择加注，加注额度为all-in [{all_in_amount:.2f}]")
                act = pkrs.Action(pkrs.ActionEnum.Raise, all_in_amount)
                _apply_and_progress(sess, act)
    s = sess.current_state
    if s and not s.final_state:
        result_html = ""
        you = s.players_state[sess.player_pos]
        your_hand = " ".join(card_to_str(c) for c in getattr(you, "hand", []))
        state_text = f"<span style='color:red;'>轮到你了.</span>. Hand: {your_hand}. Pot=${s.pot:.2f}."
        can_fold, can_cc, can_r_half, can_r_full, can_r_allin = compute_action_enables(sess)
    else:
        result_html = sess.last_result_html
        state_text = "<span style='color:red;'>这把结束了.</span>"
        can_fold = can_cc = can_r_half = can_r_full = can_r_allin = False

    board_pil, my_pil, ai1, ai2, ai3, ai4, ai5 = visuals(sess)
    ai_texts = ai_status_texts(sess)
    return (sess.dump_log(), state_text, result_html,
            gr.update(interactive=can_fold),
            gr.update(interactive=can_cc),
            gr.update(interactive=can_r_half),
            gr.update(interactive=can_r_full),
            gr.update(interactive=can_r_allin),
            board_pil, my_status_text(sess), my_pil,
            ai_texts[0], ai1, ai_texts[1], ai2,
            ai_texts[2], ai3, ai_texts[3], ai4,
            ai_texts[4], ai5, session_stats_md(sess))
css = """
/* ===== 总布局：左(Logs+Stats) / 右(对局区) ===== */
#layout_row{
  max-width: 1800px;
  margin: 0 auto;
  align-items: flex-start;   /* 关键：不要 stretch，避免高度计算怪异 */
  gap: 12px;
}
/* ===== 字体放大 ===== */
[id^="ai_txt_"] .prose,
[id^="ai_txt_"] .gr-prose,
[id^="ai_txt_"] .prose *,
#me_txt .prose,
#me_txt .gr-prose,
#me_txt .prose * {
  font-size: 20px !important;
  line-height: 1.2;
}

/* ===== 左列：日志+统计 ===== */
#left_col{
  min-width: 420px;
  max-width: 520px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* logs：固定高度 + 滚动 */
#logs_box textarea{
  height: 420px !important;
  overflow-y: auto !important;
}

/* stats：固定最大高度 + 滚动 */
#stats_md{
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 8px;
}
/* ===== 右侧对局区 ===== */
#game_col{
  flex: 1;
  display:flex;
  flex-direction:column;
  gap: 12px;
}

/* 顶/底两排座位 */
#top_seats, #bottom_seats{
  display:flex;
  justify-content: space-around;
  gap: 12px;
}

/* 中间：我 / 桌面 / P3 */
#mid_row{
  display:flex;
  justify-content: space-around;
  align-items: stretch;
  gap: 12px;
}

/* 桌面图片占满列高 */
#board_col .gr-image,
#board_col .gr-image .image-container{
  height: 100%;
}
#board_col img{
  height:100%;
  width:100%;
  object-fit:contain;
}
/* result_md 放大 */
#result_md,
#result_md .prose,
#result_md .gr-prose,
#result_md .prose * {
  font-size: 24px !important;
  text-align: center !important;
}
"""
# ===================== UI =====================
with gr.Blocks(title="Texas Hold'em", css=css) as demo:
    gr.Markdown("# No-limited Texas Hold'em-Base Demo(锦标赛模式)")
    gr.Markdown("暂时训练竞标赛模式，起始筹码是固定的，若不固定比较难学")
    sess = gr.State(Session())

    with gr.Row():
        btn_new = gr.Button("开始新一局/重置游戏环境", variant="primary", elem_id="btn_new")

    # ===== 外层：左(Logs+Stats) / 右(对局区) =====
    with gr.Row(elem_id="layout_row"):

        # ---- 左列：日志 + 统计 ----
        with gr.Column(elem_id="left_col"):
            logs = gr.Textbox(label="日志记录", value="", lines=22, interactive=False, elem_id="logs_box")
            stats_md = gr.Markdown("", elem_id="stats_md")

        # ---- 右列：对局窗口 ----
        with gr.Column(elem_id="game_col"):

            # 顶部：P1 / P2
            with gr.Row(elem_id="top_seats"):
                with gr.Column(min_width=220):
                    ai_txt_1 = gr.Markdown("", elem_id="ai_txt_1")
                    ai_img_1 = gr.Image(label="P1", interactive=False, height=150, width=200)
                with gr.Column(min_width=220):
                    ai_txt_2 = gr.Markdown("", elem_id="ai_txt_2")
                    ai_img_2 = gr.Image(label="P2", interactive=False, height=150, width=200)

            # 中部：我 / 桌面 / P3
            with gr.Row(elem_id="mid_row"):
                with gr.Column(min_width=220):
                    me_txt = gr.Markdown("", elem_id="me_txt")
                    me_img = gr.Image(label=f"P{PLAYER_POS}", interactive=False, height=150, width=200)

                with gr.Column(scale=2, elem_id="board_col"):
                    board_img = gr.Image(label="桌面", interactive=False, height=420)

                with gr.Column(min_width=220):
                    ai_txt_3 = gr.Markdown("", elem_id="ai_txt_3")
                    ai_img_3 = gr.Image(label="P3", interactive=False, height=150, width=200)

            # 底部：P5 / P4
            with gr.Row(elem_id="bottom_seats"):
                with gr.Column(min_width=220):
                    ai_txt_5 = gr.Markdown("", elem_id="ai_txt_5")
                    ai_img_5 = gr.Image(label="P5", interactive=False, height=150, width=200)
                with gr.Column(min_width=220):
                    ai_txt_4 = gr.Markdown("", elem_id="ai_txt_4")
                    ai_img_4 = gr.Image(label="P4", interactive=False, height=150, width=200)

    # 状态提示（建议也放到右侧对局区下面，或单独一行）
    state_text = gr.Markdown("尚未开始")
    result_md = gr.Markdown("", elem_id="result_md")

    with gr.Row():
        btn_fold = gr.Button("弃牌 Fold", interactive=False)
        btn_cc = gr.Button("让牌/跟注 Check/Call", interactive=False)
    with gr.Row():
        btn_raise_half_pot = gr.Button('加注 半池', interactive=False)
        btn_raise_full_pot = gr.Button('加注 全池', interactive=False)
        btn_raise_all_in_pot = gr.Button('加注 all-in', interactive=False)

    gr.Markdown("此版本为Base版本，五个AI都是不同iteration阶段保存的AI，因此会出现能力不一致的原因，一般来说Iteration越高，AI能力越强，敬请期待推出更强力的AI吧！有问题请联系微信：<mark>18656008163</mark>")
    gr.Markdown("模式为锦标赛模式，一局输的上限是200，赢的上限为5*200=1000，请及时关注左侧文本框以及赌桌上的信息提示。本游戏raise为<mark>raise by</mark>，并非raise to。")
    with gr.Row():
        with gr.Column(scale=1):  # 左侧占满空间
            pass
        with gr.Column(scale=0, min_width=150):  # 右侧固定宽度
            gr.Markdown("Created by Vincent", elem_id="signature")
    outputs_common = [logs, state_text, result_md, btn_fold, btn_cc,
                      btn_raise_half_pot, btn_raise_full_pot, btn_raise_all_in_pot,
                      board_img,
                      me_txt, me_img,  # << 新增：我的文本 + 我的手牌图片
                      ai_txt_1, ai_img_1,
                      ai_txt_2, ai_img_2,
                      ai_txt_3, ai_img_3,
                      ai_txt_4, ai_img_4,
                      ai_txt_5, ai_img_5,
                      stats_md]

    btn_new.click(start_new_game, inputs=[sess], outputs=outputs_common)
    btn_fold.click(on_fold, inputs=[sess], outputs=outputs_common)
    btn_cc.click(on_check_call, inputs=[sess], outputs=outputs_common)
    btn_raise_half_pot.click(on_raise_half_pot_amount, inputs = [sess], outputs = outputs_common)
    btn_raise_full_pot.click(on_raise_full_pot_amount, inputs = [sess], outputs = outputs_common)
    btn_raise_all_in_pot.click(on_raise_all_in_amount, inputs = [sess], outputs = outputs_common)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="启动Gradio应用")
    parser.add_argument("--port", type=int, default=8800, help="服务器端口号")

    args = parser.parse_args()

    demo.queue(max_size=32)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)
