import gradio as gr
import joblib
import numpy as np
from poker_ai.games.short_deck.state import new_game, ShortDeckPokerState
from pathlib import Path

n_players = 3
positions = ["left", "middle", "right"]
names = {"left": "BOT 1", "middle": "BOT 2", "right": "HUMAN"}

# 共享缓存

# 共享缓存立即加载（替代启动按钮的加载逻辑）

shared_data = {
    "lut": None,
    "strategy": None
}


# 精简 20 张牌直接映射
card_str_to_short = {
    "10 of spades ♠": "SX", "jack of spades ♠": "SJ", "queen of spades ♠": "SQ", "king of spades ♠": "SK", "ace of spades ♠": "SA",
    "10 of hearts ♥": "HX", "jack of hearts ♥": "HJ", "queen of hearts ♥": "HQ", "king of hearts ♥": "HK", "ace of hearts ♥": "HA",
    "10 of clubs ♣": "CX", "jack of clubs ♣": "CJ", "queen of clubs ♣": "CQ", "king of clubs ♣": "CK", "ace of clubs ♣": "CA",
    "10 of diamonds ♦": "DX", "jack of diamonds ♦": "DJ", "queen of diamonds ♦": "DQ", "king of diamonds ♦": "DK", "ace of diamonds ♦": "DA",
}

def card_to_short(card_obj):
    card_str = f"{card_obj.rank} of {card_obj.suit} {card_obj._suit_to_icon(card_obj.suit)}"
    return card_str_to_short.get(card_str, "??")

def card_to_short_code(card_obj):
    """将 Card 对象转为短码，如 CA, SX"""
    rank_map = {10: "X", 11: "J", 12: "Q", 13: "K", 14: "A"}
    suit_map = {"spades": "S", "hearts": "H", "clubs": "C", "diamonds": "D"}

    rank_char = rank_map.get(card_obj.rank_int, str(card_obj.rank_int))
    suit_char = suit_map[card_obj.suit.lower()]

    return f"{suit_char}{rank_char}"


UI_IMAGE_DIR = Path("/home/jml/poker_ai/poker_ai/terminal/ui")
CARD_BACK_IMAGE = str(UI_IMAGE_DIR / "BACK.png")  # 建议准备一张背面图用于 BOT

def create_poker_strip_image(image_paths, output_path, max_per_row=10):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np

    images = [mpimg.imread(p) for p in image_paths if Path(p).exists()]
    if len(images) == 0:
        # 若为空，创建一个空白 strip 图覆盖旧图
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return output_path

    rows = 1
    cols = len(images)
    fig, axs = plt.subplots(rows, cols, figsize=(cols, 3))

    if cols == 1:
        axs = [axs]

    for ax, img in zip(axs, images):
        ax.imshow(img)
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return output_path
def cards_to_image_paths(cards, reveal=True):
    paths = []
    for c in cards:
        code = card_to_short_code(c)
        img_path = UI_IMAGE_DIR / f"{code}.png"
        if reveal and img_path.exists():
            paths.append(str(img_path))
        else:
            paths.append(CARD_BACK_IMAGE)
    return paths

'''
10 of spades ♠: SX
jack of spades ♠: SJ
queen of spades ♠: SQ
king of spades ♠: SK
ace of spades ♠: SA

10 of hearts ♥: HX
jack of hearts ♥: HJ
queen of hearts ♥: HQ
king of hearts ♥: HK
ace of hearts ♥: HA

10 of clubs ♣: CX
jack of clubs ♣: CJ
queen of clubs ♣: CQ
king of clubs ♣: CK
ace of clubs ♣: CA

10 of diamonds ♦: DX
jack of diamonds ♦: DJ
queen of diamonds ♦: DQ
king of diamonds ♦: DK
ace of diamonds ♦: DA
'''
def render_state(state):
    log = []

    # 公共牌
    public_cards_short = " ".join([card_to_short(c) for c in state.community_cards])
    log.append(f"公共牌: {public_cards_short}\n")

    for i, player in enumerate(state.players):
        pos = positions[i]
        is_human = names[pos] == "HUMAN"
        log.append(f"[{names[pos]}]{' (HUMAN)' if is_human else ''}")

        if is_human or state.is_terminal:
            hand_cards_short = " ".join([card_to_short(c) for c in player.cards])
        else:
            hand_cards_short = "Hidden"

        log.append(f"手牌: {hand_cards_short}")
        log.append(f"筹码: {player.n_chips}, 已下注: {player.n_bet_chips}, {'弃牌' if not player.is_active else '在局中'}\n")

        # 调试打印真实牌名用于排查映射不匹配原因
        for c in state.community_cards + player.cards:
            card_str_dbg = f"{c.rank} of {c.suit} {c._suit_to_icon(c.suit)}"
            print(f"|{card_str_dbg}| --> {card_to_short(c)}")

    log.append(f"底池: {state._table.pot.total}\n")

    if state.is_terminal:
        log.append("状态: 游戏已结束，请点击【重置】重新开始。")
    else:
        log.append(f"当前行动玩家: {names[positions[state.player_i]]}")
        log.append(f"可执行动作: {state.legal_actions}")

    return "\n".join(log)
def render_state_with_images(state):
    """
    返回：
    - 文本信息（str）
    - 公共牌图片路径列表
    - HUMAN 手牌图片路径列表
    - BOT1 手牌图片路径列表
    - BOT2 手牌图片路径列表
    """

    log = []

    # 公共牌
    public_card_imgs = cards_to_image_paths(state.community_cards, reveal=True)
    public_cards_str = " ".join([card_to_short_code(c) for c in state.community_cards])
    log.append(f"公共牌: {public_cards_str}\n")

    human_hand_imgs = []
    bot1_hand_imgs = []
    bot2_hand_imgs = []
    bot1_info = ""
    bot2_info = ""
    human_info = ''
    for i, player in enumerate(state.players):
        pos = positions[i]
        is_human = names[pos] == "HUMAN"

        #log.append(f"[{names[pos]}]{' (HUMAN)' if is_human else ''}")
        if pos == "left":
            bot1_info = f"手牌: {' '.join([card_to_short(c) for c in player.cards]) if state.is_terminal else 'Hidden'}\n" \
                        f"筹码: {player.n_chips}\n已下注: {player.n_bet_chips}\n" \
                        f"{'弃牌' if not player.is_active else '在局中'}"
        elif pos == "middle":
            bot2_info = f"手牌: {' '.join([card_to_short(c) for c in player.cards]) if state.is_terminal else 'Hidden'}\n" \
                        f"筹码: {player.n_chips}\n已下注: {player.n_bet_chips}\n" \
                        f"{'弃牌' if not player.is_active else '在局中'}"
        else:
            human_info = f"手牌: {' '.join([card_to_short(c) for c in player.cards])}\n" \
                        f"筹码: {player.n_chips}\n已下注: {player.n_bet_chips}\n" \
                        f"{'弃牌' if not player.is_active else '在局中'}"
        if is_human or state.is_terminal:
            hand_cards_str = " ".join([card_to_short_code(c) for c in player.cards])
            hand_imgs = cards_to_image_paths(player.cards, reveal=True)
        else:
            hand_cards_str = "Hidden"
            hand_imgs = cards_to_image_paths(player.cards, reveal=False)

        if is_human:
            human_hand_imgs = hand_imgs
        elif pos == "left":
            bot1_hand_imgs = hand_imgs
        elif pos == "middle":
            bot2_hand_imgs = hand_imgs

        # log.append(f"手牌: {hand_cards_str}")
        # log.append(f"筹码: {player.n_chips}, 已下注: {player.n_bet_chips}, {'弃牌' if not player.is_active else '在局中'}\n")

    log.append(f"底池: {state._table.pot.total}\n")

    if state.is_terminal:
        log.append("状态: 游戏已结束，请点击【重置】重新开始。")
    else:
        log.append(f"当前行动玩家: {names[positions[state.player_i]]}")
        log.append(f"可执行动作: {state.legal_actions}")
    bot1_image_path = "/tmp/bot1_strip.png"
    bot2_image_path = "/tmp/bot2_strip.png"
    public_image_path = "/tmp/public_strip.png"
    human_image_path = "/tmp/human_strip.png"

    create_poker_strip_image(bot1_hand_imgs, bot1_image_path)
    create_poker_strip_image(bot2_hand_imgs, bot2_image_path)
    create_poker_strip_image(public_card_imgs, public_image_path)
    create_poker_strip_image(human_hand_imgs, human_image_path)


    return "\n".join(log), public_image_path, human_image_path, bot1_image_path, bot2_image_path, bot1_info, bot2_info, human_info





def auto_bot(state):
    while not state.is_terminal and names[positions[state.player_i]] != "HUMAN":
        offline_strategy = shared_data["strategy"]
        default_strategy = {a: 1/len(state.legal_actions) for a in state.legal_actions}
        this_state_strategy = offline_strategy.get(state.info_set, default_strategy)
        total = sum(this_state_strategy.values())
        this_state_strategy = {k: v/total for k, v in this_state_strategy.items()}
        actions = list(this_state_strategy.keys())
        probs = list(this_state_strategy.values())
        action = np.random.choice(actions, p=probs)
        state = state.apply_action(action)
    return state
def perform_action(action_name, state):
    if state.is_terminal:
        log = "游戏已结束，请点击【重置】或【启动】重新开始。"
        return log, state

    if action_name not in state.legal_actions:
        log = f"当前无法执行 {action_name}，可执行动作: {state.legal_actions}"
        return log, state

    state = state.apply_action(action_name)
    state = auto_bot(state)
    log = render_state(state)
    return log, state

with gr.Blocks() as demo:
    gr.Markdown("# ♠️ Poker CFR Gradio 对战 Demo（可视化牌面）")

    # 顶部三列布局： BOT1 | 公共牌 | BOT2
    with gr.Row():
        with gr.Column(scale=1):
            bot1_image = gr.Image(label="BOT1 手牌")
            bot1_info_output = gr.Textbox(label="BOT1 信息", lines=4, interactive=False)

        with gr.Column(scale=2):
            public_image = gr.Image(label="公共牌")
            human_info_output = gr.Textbox(label="Human 信息", lines=4, interactive=False)

        with gr.Column(scale=1):
            bot2_image = gr.Image(label="BOT2 手牌")
            bot2_info_output = gr.Textbox(label="BOT2 信息", lines=4, interactive=False)

    # 下方 HUMAN 手牌
    human_image = gr.Image(label="HUMAN 手牌")

    # 牌局日志信息展示
    log_output = gr.Textbox(label="牌局状态", lines=10, interactive=False)

    # 状态缓存
    state = gr.State()

    # 按钮区
    with gr.Row():
        start_btn = gr.Button("▶️ 启动")
        reset_btn = gr.Button("🔄 重置")
        fold_btn = gr.Button("🪃 Fold")
        call_btn = gr.Button("☎️ Call")
        raise_btn = gr.Button("🚀 Raise")

    def start_game_gr():
        shared_data["lut"] = '/home/jml/poker_ai/research/blueprint_algo'
        offline_strategy_dict = joblib.load('/home/jml/poker_ai/research/blueprint_algo/agent.joblib')
        shared_data["strategy"] = offline_strategy_dict["strategy"]

        state = new_game(
            n_players,
            lut_path=shared_data["lut"],
            pickle_dir=False
        )
        state = auto_bot(state)
        return get_render_outputs(state)


    def reset_game_gr(state):
        if state is None:
            return "请先点击【启动】加载环境后再使用【重置】。", None, None, None, None, "", "", state
        state: ShortDeckPokerState = new_game(n_players, state.card_info_lut)
        state = auto_bot(state)
        return get_render_outputs(state)


    def perform_action_gr(action_name, state):
        state = perform_action(action_name, state)[1]
        return get_render_outputs(state)

    def get_render_outputs(state):
        log, pub_imgs, human_imgs, bot1_imgs, bot2_imgs, bot1_info, bot2_info, human_info = render_state_with_images(state)
        return log, pub_imgs, human_imgs, bot1_imgs, bot2_imgs, bot1_info, bot2_info, human_info,state

    start_btn.click(
        fn=start_game_gr, inputs=[],
        outputs=[
            log_output, public_image, human_image,
            bot1_image, bot2_image,
            bot1_info_output, bot2_info_output, human_info_output,
            state
        ]
    )

    reset_btn.click(
        fn=reset_game_gr, inputs=state,
        outputs=[
            log_output, public_image, human_image,
            bot1_image, bot2_image,
            bot1_info_output, bot2_info_output, human_info_output,
            state
        ]
    )

    fold_btn.click(
        fn=lambda s: perform_action_gr("fold", s), inputs=state,
        outputs=[
            log_output, public_image, human_image,
            bot1_image, bot2_image,
            bot1_info_output, bot2_info_output, human_info_output,
            state
        ]
    )
    call_btn.click(
        fn=lambda s: perform_action_gr("call", s), inputs=state,
        outputs=[
            log_output, public_image, human_image,
            bot1_image, bot2_image,
            bot1_info_output, bot2_info_output, human_info_output,
            state
        ]
    )
    raise_btn.click(
        fn=lambda s: perform_action_gr("raise", s), inputs=state,
        outputs=[
            log_output, public_image, human_image,
            bot1_image, bot2_image,
            bot1_info_output, bot2_info_output, human_info_output,
            state
        ]
    )

demo.launch(server_name="0.0.0.0", server_port=8821, share=True)


