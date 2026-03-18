import random
import time
from typing import Dict

import click
import joblib
import numpy as np
from blessed import Terminal

from poker_ai.games.short_deck.state import new_game, ShortDeckPokerState
from poker_ai.terminal.ascii_objects.card_collection import AsciiCardCollection
from poker_ai.terminal.ascii_objects.player import AsciiPlayer
from poker_ai.terminal.ascii_objects.logger import AsciiLogger
from poker_ai.terminal.render import print_footer, print_header, print_log, print_table
from poker_ai.terminal.results import UserResults
from poker_ai.utils.algos import rotate_list


def run_terminal_app(
        lut_path='/home/jml/poker_ai/research/blueprint_algo',
        pickle_dir=False,
        agent: str = "offline",
        strategy_path: str = '/home/jml/poker_ai/research/blueprint_algo/agent.joblib',
        debug_quick_start: bool = False
):
    """Start up terminal app to play against your poker AI.

    Example
    -------

    Usually you would call this from the `poker_ai` CLI. Alternatively you can
    call this method from this module directly from python.

    ```bash
    python -m poker_ai.terminal.runner                                       \
        --lut_path ./research/blueprint_algo                               \
        --agent offline                                                      \
        --pickle_dir ./research/blueprint_algo                               \
        --strategy_path ./agent.joblib                                       \
        --no_debug_quick_start
    ```
    """
    term = Terminal()
    log = AsciiLogger(term)
    n_players: int = 3

    state: ShortDeckPokerState = new_game(  # shortdeckpokerstate：就是RL里面的环境了
        n_players,
        lut_path='/home/jml/poker_ai/research/blueprint_algo',
        pickle_dir=False
    )
    n_table_rotations: int = 0
    selected_action_i: int = 0
    positions = ["left", "middle", "right"]
    names = {"left": "BOT 1", "middle": "BOT 2", "right": "HUMAN"}
    offline_strategy_dict = joblib.load('/home/jml/poker_ai/research/blueprint_algo/agent.joblib')  # 类似与json pickle这种
    offline_strategy = offline_strategy_dict['strategy']  # 主要策略表？
    # Using the more fine grained preflop strategy would be a good idea
    # for a future improvement
    del offline_strategy_dict["pre_flop_strategy"]  # 翻牌前的特殊策略
    del offline_strategy_dict["regret"]  # 后悔值表(用于cfr 收敛)
    # 但是都删除了 是因为什么呢？为了节省内存吗？

    user_results: UserResults = UserResults()
    with term.cbreak(), term.hidden_cursor():
        while True:
            # Construct ascii objects to be rendered later.
            ascii_players: Dict[str, AsciiPlayer] = {}
            state_players = rotate_list(state.players[::-1], n_table_rotations)
            og_name_to_position = {}
            og_name_to_name = {}
            for player_i, player in enumerate(state_players):
                position = positions[player_i]
                is_human = names[position].lower() == "human"
                ascii_players[position] = AsciiPlayer(
                    *player.cards,
                    term=term,
                    name=names[position],
                    og_name=player.name,
                    hide_cards=not is_human and not state.is_terminal,
                    folded=not player.is_active,
                    is_turn=player.is_turn,
                    chips_in_pot=player.n_bet_chips,
                    chips_in_bank=player.n_chips,
                    is_small_blind=player.is_small_blind,
                    is_big_blind=player.is_big_blind,
                    is_dealer=player.is_dealer,
                )
                print('hand_cards:', *player.cards)
                print('name', names[position])
                print('og_name', player.name)
                print('folded', not player.is_active)
                print('is_turn', player.is_turn)
                print('chips_in_pot', player.n_bet_chips)
                print('chips_in_bank', player.n_chips)
                print('is_small_blind', player.is_small_blind)
                print('is_big_blind', player.is_big_blind)
                print('is_dealer', player.is_dealer)  # 庄

                og_name_to_position[player.name] = position
                og_name_to_name[player.name] = names[position]
                if player.is_turn:
                    current_player_name = names[position]
            public_cards = AsciiCardCollection(*state.community_cards)
            print('public_cards:', *state.community_cards)
            if state.is_terminal:
                legal_actions = ["quit", "new game"]
                human_should_interact = True
            else:
                og_current_name = state.current_player.name
                human_should_interact = og_name_to_position[og_current_name] == "right"
                if human_should_interact:
                    legal_actions = state.legal_actions
                else:
                    legal_actions = []
            # Render game.
            print(term.home + term.white + term.clear)
            print_header(term, state, og_name_to_name)
            print_table(
                term,
                ascii_players,
                public_cards,
                n_table_rotations,
                n_chips_in_pot=state._table.pot.total,
            )
            print_footer(term, selected_action_i, legal_actions)
            print_log(term, log)
            # Make action of some kind.
            print('小盲注：', state.small_blind)
            print('大盲注：', state.big_blind)
            print('玩家列表:', state.players)
            if human_should_interact:
                # Incase the legal_actions went from length 3 to 2 and we had
                # previously picked the last one.
                selected_action_i %= len(legal_actions)
                key = term.inkey(timeout=None)
                if key.name == "q":
                    log.info(term.pink("quit"))
                    break
                elif key.name == "KEY_LEFT":
                    selected_action_i -= 1
                    if selected_action_i < 0:
                        selected_action_i = len(legal_actions) - 1
                elif key.name == "KEY_RIGHT":
                    selected_action_i = (selected_action_i + 1) % len(legal_actions)
                elif key.name == "KEY_ENTER":
                    action = legal_actions[selected_action_i]
                    if action == "quit":
                        user_results.add_result(strategy_path, agent, state, og_name_to_name)
                        log.info(term.pink("quit"))
                        break
                    elif action == "new game":
                        user_results.add_result(strategy_path, agent, state, og_name_to_name)
                        log.clear()
                        log.info(term.green("new game"))
                        if debug_quick_start:
                            state: ShortDeckPokerState = new_game(
                                n_players, state.card_info_lut, load_card_lut=False,
                            )
                        else:
                            state: ShortDeckPokerState = new_game(
                                n_players, state.card_info_lut,
                            )
                        n_table_rotations -= 1
                        if n_table_rotations < 0:
                            n_table_rotations = n_players - 1
                    else:
                        log.info(term.green(f"{current_player_name} chose {action}"))
                        state: ShortDeckPokerState = state.apply_action(action)
            else:
                if agent == "random":
                    action = random.choice(state.legal_actions)
                    time.sleep(0.8)
                elif agent == "offline":
                    default_strategy = {
                        action: 1 / len(state.legal_actions)
                        for action in state.legal_actions
                    }
                    this_state_strategy = offline_strategy.get(
                        state.info_set, default_strategy
                    )
                    # Normalizing strategy.
                    total = sum(this_state_strategy.values())
                    this_state_strategy = {
                        k: v / total for k, v in this_state_strategy.items()
                    }
                    actions = list(this_state_strategy.keys())
                    probabilties = list(this_state_strategy.values())
                    action = np.random.choice(actions, p=probabilties)
                    time.sleep(0.8)
                log.info(f"{current_player_name} chose {action}")
                state: ShortDeckPokerState = state.apply_action(action)


if __name__ == "__main__":
    run_terminal_app()
