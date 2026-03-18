import random
import time
import joblib
import numpy as np

from poker_ai.games.short_deck.state import new_game, ShortDeckPokerState
from poker_ai.utils.algos import rotate_list

def run_pure_cli(
        lut_path='/home/jml/poker_ai/research/blueprint_algo',
        pickle_dir=False,
        agent: str = "offline",
        strategy_path: str = '/home/jml/poker_ai/research/blueprint_algo/agent.joblib',
        debug_quick_start: bool = False
):
    n_players = 3
    state: ShortDeckPokerState = new_game(
        n_players,
        lut_path=lut_path,
        pickle_dir=pickle_dir
    )
    positions = ["left", "middle", "right"]
    names = {"left": "BOT 1", "middle": "BOT 2", "right": "HUMAN"}
    offline_strategy_dict = joblib.load(strategy_path)
    offline_strategy = offline_strategy_dict['strategy']
    del offline_strategy_dict["pre_flop_strategy"]
    del offline_strategy_dict["regret"]

    n_table_rotations = 0
    selected_action_i = 0

    while True:
        state_players = rotate_list(state.players[::-1], n_table_rotations)
        og_name_to_position = {}
        og_name_to_name = {}

        print("\n==========================")
        print(f"公共牌: {' '.join([str(c) for c in state.community_cards])}")
        for player_i, player in enumerate(state_players):
            position = positions[player_i]
            is_human = names[position].lower() == "human"
            og_name_to_position[player.name] = position
            og_name_to_name[player.name] = names[position]
            print(f"\n[{names[position]}] {'(HUMAN)' if is_human else ''}")
            print(f"手牌: {' '.join([str(c) for c in player.cards]) if (is_human or state.is_terminal) else 'Hidden'}")
            print(f"筹码: {player.n_chips} | 已下注: {player.n_bet_chips} | {'已弃牌' if not player.is_active else '正在游戏'}")
            if player.is_turn:
                current_player_name = names[position]

        print(f"\n底池: {state._table.pot.total}")
        print("==========================")

        if state.is_terminal:
            print("当前状态: 牌局已结束")
            legal_actions = ["quit", "new game"]
            human_should_interact = True
        else:
            og_current_name = state.current_player.name
            human_should_interact = og_name_to_position[og_current_name] == "right"
            if human_should_interact:
                legal_actions = state.legal_actions
                print(f"当前行动: {current_player_name} (HUMAN)")
                print(f"可选动作: {legal_actions}")
            else:
                legal_actions = []

        if human_should_interact:
            while True:
                action = input("请输入你的动作（可选: " + ", ".join(legal_actions) + "）：").strip()
                if action not in legal_actions:
                    print("无效动作，请重新输入。")
                else:
                    break
            if action == "quit":
                print("已退出游戏。")
                break
            elif action == "new game":
                print("开始新游戏...")
                if debug_quick_start:
                    state = new_game(n_players, state.card_info_lut, load_card_lut=False)
                else:
                    state = new_game(n_players, state.card_info_lut)
                n_table_rotations = (n_table_rotations - 1) % n_players
            else:
                state = state.apply_action(action)
        else:
            print(f"{current_player_name} (BOT) 思考中...")
            time.sleep(0.5)
            if agent == "random":
                action = random.choice(state.legal_actions)
            elif agent == "offline":
                default_strategy = {action: 1 / len(state.legal_actions) for action in state.legal_actions}
                this_state_strategy = offline_strategy.get(state.info_set, default_strategy)
                total = sum(this_state_strategy.values())
                this_state_strategy = {k: v / total for k, v in this_state_strategy.items()}
                actions = list(this_state_strategy.keys())
                probabilities = list(this_state_strategy.values())
                action = np.random.choice(actions, p=probabilities)
            print(f"{current_player_name} (BOT) 选择动作: {action}")
            state = state.apply_action(action)

if __name__ == "__main__":
    run_pure_cli()
