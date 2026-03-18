import os,re,glob,logging
from typing import List
from src.core.deep_cfr_clean_model_input import DeepCFRAgent
from src.agents.random_agent_modified import RandomAgent
from src.agents.random_agent_hard import RandomAgent_hard
from src.agents.random_agent_hard_ez_raise import RandomAgent_fixed50

import torch
_iter_pat = re.compile(r"iter_(\d+)\.pt$")
logger = logging.getLogger(__name__)

def _iter_from_name(path: str) -> int:
    """
    从文件名中解析 iteration，比如 checkpoint_iter_0900.pt -> 900
    解析失败则返回 -1（保证这种文件排在最后）
    """
    m = _iter_pat.search(os.path.basename(path))
    return int(m.group(1)) if m else -1

def list_latest_checkpoints(models_dir: str, pattern: str = "checkpoint_iter_*.pt", k: int = 5):
    files = glob.glob(os.path.join(models_dir, pattern))
    # 以“迭代号”排序（文件名形如 checkpoint_iter_250.pt）
    def _iter_num(p):
        try:
            s = os.path.basename(p).split("_")[-1].split(".")[0]
            return int(s)
        except Exception:
            return -1
    files = sorted(files, key=_iter_num, reverse=True)
    return files[:k]

def load_checkpoint_as_agent(checkpoint_path: str, player_id: int, device: str):
    agent = DeepCFRAgent(player_id=player_id, num_players=6, device=device)
    # 仅加载权重（安全 & 未来兼容）
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent.advantage_net.load_state_dict(state['advantage_net'])
    agent.strategy_net.load_state_dict(state['strategy_net'])
    agent.advantage_net.eval()  #梯度冻结
    agent.strategy_net.eval()
    for p in agent.advantage_net.parameters():
        p.requires_grad_(False)
    for p in agent.strategy_net.parameters():
        p.requires_grad_(False)
    return agent

def build_selfplay_opponents(models_dir: str, device: str):
    """
    返回长度为6的列表：索引0放 None（学习者自己），1~5 放 5 个对手 Agent。
    规则：
      - 如果 models_dir 里 <5 个 checkpoint：用 5 个 FrozenSelfPlayAgent（从当前 learning_agent 冻结克隆）
      - 否则：用最新的 5 个 checkpoint 依次加载到座位 1..5
    """
    opponents = [None] * 6  # seat 0 是学习者
    latest = list_latest_checkpoints(models_dir, "checkpoint_iter_*.pt", k=5)  # 记得这个 k 也要进行修改 不然下面这条判断永远为真！搞得他妈的永远是进行自博弈！
    if len(latest) < 5:  #random_play 第一阶段 这边记得换成10！！！！！！！！！！ 不然就把5个ckpt加载进来了！！！！
        logger.info(f"[Random-PLAY] <5 checkpoints in {models_dir}; using 5 RandomAgents as opponents.")
        for seat in range(1, 6):
            opponents[seat] = RandomAgent_hard(seat)
    else:
        logging.info(f"[SELF-PLAY] Using latest 5 checkpoints: {[os.path.basename(x) for x in latest]}")
        for seat, ckpt in zip(range(1, 6), latest):
            opponents[seat] = load_checkpoint_as_agent(ckpt, player_id=seat, device=device)
    return opponents
