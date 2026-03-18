import copy
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np

from poker_ai.ai.agent import Agent
from poker_ai.games.short_deck.state import ShortDeckPokerState


log = logging.getLogger("sync.ai")


def calculate_strategy(this_info_sets_regret: Dict[str, float]) -> Dict[str, float]: #这个函数的意义就是 将 某个信息集合里面的正遗憾 转换成 一组 行动概率
    # 这一步就是 从遗憾 得到 概率的关键一步
    # 仍然是 我们只看 正遗憾值，不看负遗憾值
    """
    Calculate the strategy based on the current information sets regret.

    ...

    Parameters
    ----------
    this_info_sets_regret : Dict[str, float]
        Regret for each action at this info set.

    Returns
    -------
    strategy : Dict[str, float]
        Strategy as a probability over actions.
    """
    # TODO: Could we instanciate a state object from an info set?
    actions = this_info_sets_regret.keys()
    regret_sum = sum([max(regret, 0) for regret in this_info_sets_regret.values()]) #这个字典---动作：该动作在这个信息集合上的累计遗憾
    if regret_sum > 0: #只统计 正向的
        strategy: Dict[str, float] = {
            action: max(this_info_sets_regret[action], 0) / regret_sum
            for action in actions
        }
    else:
        default_probability = 1 / len(actions) #如果是 负的 就均匀分布 说明 没有哪个动作更好
        strategy: Dict[str, float] = {action: default_probability for action in actions}
    return strategy


def update_strategy(  #这是个递归 遍历器
    agent: Agent,
    state: ShortDeckPokerState,
    i: int,
    t: int,
    locks: Dict[str, mp.synchronize.Lock] = {}, #我方： 进行概率随机采样 而对手 则进行 所有合法动作 展开
        # 但是 这个只实现了 preflop阶段的遍历器 只跑了 最容易的一段
):
    """
    Update pre flop strategy using a more theoretically sound approach.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    i : int
        The Player.
    t : int
        The iteration.
    locks : Dict[str, mp.synchronize.Lock]
        The locks for multiprocessing
    """
    ph = state.player_i  # this is always the case no matter what i is

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand or state.betting_round > 0: #终局 / 玩家已经弃牌 / 已经进入翻盘圈 ---只统计 preflop的平均策略？
        # 先做一个固定的翻牌策略？
        # 所以 要开放 flop及以后的 是要把 state.betting_round > 0 取消掉的吗？
        return

    # NOTE(fedden): According to Algorithm 1 in the supplementary material,
    #               we would add in the following bit of logic. However we
    #               already have the game logic embedded in the state class,
    #               and this accounts for the chance samplings. In other words,
    #               it makes sure that chance actions such as dealing cards
    #               happen at the appropriate times.
    # elif h is chance_node:
    #   sample action from strategy for h
    #   update_strategy(rs, h + a, i, t)

    elif ph == i: #轮到我们了
        # calculate regret
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret) #返回 当前的动作后悔值的 概率分布
        log.debug(f"Calculated Strategy for {state.info_set}: {sigma}")
        # choose an action based of sigma
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: np.ndarray = np.array(list(sigma.values()))
        action: str = np.random.choice(available_actions, p=action_probabilities) #按照这个 概率 随机采样 一个动作
        log.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")
        # Increment the action counter.
        if locks:
            locks["strategy"].acquire()
        this_states_strategy = agent.strategy.get(
            state.info_set, state.initial_strategy
        )
        this_states_strategy[action] += 1 #把改动作 计数 加一
        # Update the master strategy by assigning.
        agent.strategy[state.info_set] = this_states_strategy
        if locks:
            locks["strategy"].release()
        new_state: ShortDeckPokerState = state.apply_action(action) #进入 下一个状态 进行递归
        update_strategy(agent, new_state, i, t, locks)
    else:
        # Traverse each action.
        for action in state.legal_actions: #把对手的 所有合法动作 逐个展开，不采样，进行全分支遍历
            log.debug(f"Going to Traverse {action} for opponent")
            new_state: ShortDeckPokerState = state.apply_action(action)
            update_strategy(agent, new_state, i, t, locks)


def cfr(
    agent: Agent,
    state: ShortDeckPokerState,
    i: int,
    t: int,
    locks: Dict[str, mp.synchronize.Lock] = {},
) -> float:
    """
    Regular counter factual regret minimization algorithm.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    i : int
        The Player.
    t : int
        The iteration.
    locks : Dict[str, mp.synchronize.Lock]
        The locks for multiprocessing
    """
    log.debug("CFR")
    log.debug("########")
    log.debug(f"Iteration: {t}")
    log.debug(f"Player Set to Update Regret: {i}")
    log.debug(f"P(h): {state.player_i}")
    log.debug(f"P(h) Updating Regret? {state.player_i == i}")
    log.debug(f"Betting Round {state._betting_stage}")
    log.debug(f"Community Cards {state._table.community_cards}")
    for i, player in enumerate(state.players):
        log.debug(f"Player {i} hole cards: {player.cards}")
    try:
        log.debug(f"I(h): {state.info_set}")
    except KeyError:
        pass
    log.debug(f"Betting Action Correct?: {state.players}")

    ph = state.player_i  # 当前轮到谁行动了

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]  #终局 或者 弃牌 就返回这个玩家的钱 然后开始回溯  所以这个函数最后的返回值 永远是 一条路径 从这里往下走到终局，玩家 i 的收益

    # NOTE(fedden): The logic in Algorithm 1 in the supplementary material
    #               instructs the following lines of logic, but state class
    #               will already skip to the next in-hand player.
    # elif p_i not in hand:
    #   cfr()
    # NOTE(fedden): According to Algorithm 1 in the supplementary material,
    #               we would add in the following bit of logic. However we
    #               already have the game logic embedded in the state class,
    #               and this accounts for the chance samplings. In other words,
    #               it makes sure that chance actions such as dealing cards
    #               happen at the appropriate times.
    # elif h is chance_node:
    #   sample action from strategy for h
    #   cfr()

    elif ph == i: #轮到我了
        # calculate strategy
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret) #拿到这个信息集合上累计的regret
        sigma = calculate_strategy(this_info_sets_regret) #用regret - matching 计算出 当前要使用的 策略
        log.debug(f"Calculated Strategy for {state.info_set}: {sigma}")

        vo = 0.0  #当前 信息集 在当前策略下的期望
        voa: Dict[str, float] = {} # 每个动作单独走一遍得到的子树价值
        for action in state.legal_actions:  #对所有节点都要尝试一遍
            log.debug(
                f"ACTION TRAVERSED FOR REGRET: ph {state.player_i} ACTION: {action}"
            )
            new_state: ShortDeckPokerState = state.apply_action(action)
            voa[action] = cfr(agent, new_state, i, t, locks) # 递归计算这个动作的价值
            log.debug(f"Got EV for {action}: {voa[action]}") #
            vo += sigma[action] * voa[action] #累加成节点期望
            log.debug(
                f"Added to Node EV for ACTION: {action} INFOSET: {state.info_set}\n"
                f"STRATEGY: {sigma[action]}: {sigma[action] * voa[action]}"
            )
        log.debug(f"Updated EV at {state.info_set}: {vo}")
        if locks:
            locks["regret"].acquire()
            # 有了每个动作的价值 voa[action],也就有了 当前策略的期望 vo  就能算 regret了
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        for action in state.legal_actions:
            this_info_sets_regret[action] += voa[action] - vo  #更新遗憾值  但是这里是 累积的cfr，每迭代一次 就把这一局 的 遗憾 加上去 然后再用这张大表跑 regret_matching 就会越来越偏向那些过去 后悔得多 的动作
        # Assign regret back to the shared memory.
        agent.regret[state.info_set] = this_info_sets_regret
        if locks:
            locks["regret"].release()
        return vo
    else: #轮到对手 / 不是我更新的玩家
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)
        log.debug(f"Calculated Strategy for {state.info_set}: {sigma}")
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)
        log.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")  #对手 就不计算遗憾值了 直接按照他们的策略 采样一次，不需要计算遗憾值
        new_state: ShortDeckPokerState = state.apply_action(action)
        return cfr(agent, new_state, i, t, locks)
'''
cfr() 是 计算遗憾的 regret表
update_regret 是把按照当前regret算出的策略 累积成 平均策略的 stragety 表
'''

def cfrp(
    agent: Agent,
    state: ShortDeckPokerState,
    i: int,
    t: int,
    c: int,
    locks: Dict[str, mp.synchronize.Lock] = {},
):
    # 带剪枝的cfr
    # 如果某个动作的累计遗憾不够大 也就是说 看起来一直都不好，那么我这次迭代 就先不往 这个动作下面展开了 省时间
    # 等哪天遗憾值又变大了再反剪枝回来
    # 但是这种 cfrp 也是有风险的，
    # 1. 可能出现一个动作都没有展开的情况： 如果这条信息集合上的所有动作遗憾都＜= c 那么 vo 会一直是0，然后你也不会更新任何动作的regret，这一层就等于白跑 所以至少展开一个(遗憾值最大的那个？要么把c 换成 随迭代设计变化的门槛)
    # 2. 反剪枝的奖励： 重新展开一个之前被剪过的动作时候，把他的遗憾往上托一点，不然很难被重新选中
    # 3. c是固定的？c理应随着迭代次数t的增大 而 慢慢放宽吗？
    # 4. 还是没有把对手的采样概率 带进来 做 importance correction，这一点可以认为是 todo吧
    """
    Counter factual regret minimazation with pruning.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    i : int
        The Player.
    t : int
        The iteration.
    locks : Dict[str, mp.synchronize.Lock]
        The locks for multiprocessing
    """
    ph = state.player_i

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]
    # NOTE(fedden): The logic in Algorithm 1 in the supplementary material
    #               instructs the following lines of logic, but state class
    #               will already skip to the next in-hand player.
    # elif p_i not in hand:
    #   cfr()
    # NOTE(fedden): According to Algorithm 1 in the supplementary material,
    #               we would add in the following bit of logic. However we
    #               already have the game logic embedded in the state class,
    #               and this accounts for the chance samplings. In other words,
    #               it makes sure that chance actions such as dealing cards
    #               happen at the appropriate times.
    # elif h is chance_node:
    #   sample action from strategy for h
    #   cfr()
    elif ph == i:
        # calculate strategy
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)
        # TODO: Does updating sigma here (as opposed to after regret) miss out
        #       on any updates? If so, is there any benefit to having it up
        #       here?
        vo = 0.0
        voa: Dict[str, float] = dict()
        # Explored dictionary to keep track of regret updates that can be
        # skipped.
        explored: Dict[str, bool] = {action: False for action in state.legal_actions}
        # Get the regret for this state.
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        for action in state.legal_actions:
            if this_info_sets_regret[action] > c: #累计遗憾 大于 某个阈值 c 的时候 才会去真的展开他 否则就当他不存在 这次递归就不走他 这就是 剪枝
                new_state: ShortDeckPokerState = state.apply_action(action)
                voa[action] = cfrp(agent, new_state, i, t, c, locks)
                explored[action] = True
                vo += sigma[action] * voa[action]
        if locks:
            locks["regret"].acquire()
        # Get the regret for this state again, incase any other process updated
        # it whilst we were doing `cfrp`.
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        for action in state.legal_actions: #最后更新遗憾的时候 如果没有评估这个动作 那么就别更新他的遗憾
            if explored[action]:
                this_info_sets_regret[action] += voa[action] - vo
        # Update the master copy of the regret.
        agent.regret[state.info_set] = this_info_sets_regret
        if locks:
            locks["regret"].release()
        return vo
    else:
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)
        new_state: ShortDeckPokerState = state.apply_action(action)
        return cfrp(agent, new_state, i, t, c, locks)


def serialise(
    agent: Agent,
    save_path: Path,
    t: int,
    server_state: Dict[str, Union[str, float, int, None]],
    locks: Dict[str, mp.synchronize.Lock] = {},
): # 训练中断的时候 的 保存 和 恢复功能
    """
    Write progress of optimising agent (and server state) to file.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    save_path : ShortDeckPokerState
        Current game state.
    t : int
        The iteration.
    server_state : Dict[str, Union[str, float, int, None]]
        All the variables required to resume training.
    locks : Dict[str, mp.synchronize.Lock]
        The locks for multiprocessing
    """
    # Load the shared strategy that we accumulate into.
    agent_path = os.path.abspath(str(save_path / f"agent.joblib"))
    if os.path.isfile(agent_path):
        offline_agent = joblib.load(agent_path)  #首先尝试 加载 如果没有那就新建
    else:
        offline_agent = {
            "regret": {},
            "timestep": t,
            "strategy": {},
            "pre_flop_strategy": {}
        }
    # Lock shared dicts so no other process modifies it whilst writing to
    # file.
    # Calculate the strategy for each info sets regret, and accumulate in
    # the offline agent's strategy.
    for info_set, this_info_sets_regret in sorted(agent.regret.items()):
        if locks:
            locks["regret"].acquire()
        strategy = calculate_strategy(this_info_sets_regret)
        if locks:
            locks["regret"].release()
        if info_set not in offline_agent["strategy"]:
            offline_agent["strategy"][info_set] = {
                action: probability for action, probability in strategy.items()
            }
        else:
            for action, probability in strategy.items():
                offline_agent["strategy"][info_set][action] += probability
    if locks:
        locks["regret"].acquire()
    offline_agent["regret"] = copy.deepcopy(agent.regret)
    if locks:
        locks["regret"].release()
    if locks:
        locks["pre_flop_strategy"].acquire()
    offline_agent["pre_flop_strategy"] = copy.deepcopy(agent.strategy)
    if locks:
        locks["pre_flop_strategy"].release()
    joblib.dump(offline_agent, agent_path)
    # Dump the server state to file too, but first update a few bits of the
    # state so when we load it next time, we start from the right place in
    # the optimisation process.
    server_path = save_path / f"server.gz"  #更新 并 保存 服务器状态
    server_state["agent_path"] = agent_path
    server_state["start_timestep"] = t + 1
    joblib.dump(server_state, server_path)
