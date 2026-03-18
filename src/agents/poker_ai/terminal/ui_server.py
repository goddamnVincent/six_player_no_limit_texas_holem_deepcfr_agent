import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, List, Optional
import random
import joblib
import numpy as np
from poker_ai.games.short_deck.state import new_game, ShortDeckPokerState


# 德州扑克服务器
class PokerServer:
    def __init__(self):
        self.games: Dict[str, ShortDeckPokerState] = {}  # 存储多个游戏实例
        self.ai_strategy = self.load_ai_strategy()
        self.n_players = 3  # 默认3个玩家(2个AI和1个人类)

    def load_ai_strategy(self):
        """加载AI策略"""
        strategy_path = '/home/jml/poker_ai/research/blueprint_algo/agent.joblib'
        offline_strategy_dict = joblib.load(strategy_path)
        offline_strategy = offline_strategy_dict['strategy']
        return offline_strategy

    def create_game(self, game_id: str) -> ShortDeckPokerState:
        """创建新游戏"""
        lut_path = '/home/jml/poker_ai/research/blueprint_algo'
        state = new_game(self.n_players, lut_path=lut_path, pickle_dir=False)
        self.games[game_id] = state
        return state

    def get_game(self, game_id: str) -> Optional[ShortDeckPokerState]:
        """获取游戏状态"""
        return self.games.get(game_id)

    def human_action(self, game_id: str, action: str) -> Dict:
        """处理人类玩家动作"""
        state = self.get_game(game_id)
        if not state:
            return {"error": "Game not found", "valid": False}

        if action not in state.legal_actions:
            return {"error": "Invalid action", "valid": False, "legal_actions": state.legal_actions}

        state = state.apply_action(action)
        self.games[game_id] = state

        # 检查游戏是否结束
        if state.is_terminal:
            return {
                "valid": True,
                "game_over": True,
                "winners": self.get_winners(state),
                "next_player": None
            }

        # 如果游戏继续，处理AI动作
        return self.ai_action(game_id)

    def ai_action(self, game_id: str) -> Dict:
        """处理AI玩家动作"""
        state = self.get_game(game_id)
        if not state:
            return {"error": "Game not found", "valid": False}

        response = {"valid": True, "game_over": False}

        # 处理所有AI玩家的动作
        while not state.is_terminal and not state.current_player.name.lower() == "human":
            default_strategy = {
                action: 1 / len(state.legal_actions)
                for action in state.legal_actions
            }
            this_state_strategy = self.ai_strategy.get(
                state.info_set, default_strategy
            )
            # 归一化策略
            total = sum(this_state_strategy.values())
            this_state_strategy = {
                k: v / total for k, v in this_state_strategy.items()
            }
            actions = list(this_state_strategy.keys())
            probabilities = list(this_state_strategy.values())
            action = np.random.choice(actions, p=probabilities)

            # 应用AI动作
            state = state.apply_action(action)
            self.games[game_id] = state

            # 记录AI动作
            response.setdefault("ai_actions", []).append({"player": state.current_player.name,"action": action})

        # 更新游戏状态
        if state.is_terminal:
            response.update({
                "game_over": True,
                "winners": self.get_winners(state)
            })
        else:
            response.update({
                "next_player": state.current_player.name,
                "legal_actions": state.legal_actions
            })

        return response

    def get_winners(self, state: ShortDeckPokerState) -> List[str]:
        """获取赢家"""
        winners = []
        for player in state.players:
            if player.n_chips > 0:  # 简化判断，实际应该根据牌型等
                winners.append(player.name)
        return winners


# HTTP请求处理器
class PokerRequestHandler(SimpleHTTPRequestHandler):
    poker_server = PokerServer()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")
        SimpleHTTPRequestHandler.end_headers(self)

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

    def do_POST(self):
        if self.path == '/create_game':
            self.handle_create_game()
        elif self.path == '/action':
            self.handle_player_action()
        elif self.path == '/reset':
            self.handle_reset_game()
        else:
            self.send_error(404, "Not Found")

    def handle_create_game(self):
        """处理创建新游戏请求"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        game_id = data.get('game_id', str(random.randint(1000, 9999)))
        state = self.poker_server.create_game(game_id)

        # 初始游戏状态
        initial_state = {
            "game_id": game_id,
            "players": [p.name for p in state.players],
            "current_player": state.current_player.name,
            "legal_actions": state.legal_actions if state.current_player.name.lower() == "human" else [],
            "community_cards": state.community_cards,
            "pot": state._table.pot.total
        }

        # 如果是AI先行动，处理AI动作
        if state.current_player.name.lower() != "human":
            ai_response = self.poker_server.ai_action(game_id)
            initial_state.update(ai_response)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(initial_state).encode('utf-8'))

    def handle_player_action(self):
        """处理玩家动作请求"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        game_id = data.get('game_id')
        action = data.get('action')

        if not game_id or not action:
            self.send_error(400, "Missing game_id or action")
            return

        response = self.poker_server.human_action(game_id, action)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def handle_reset_game(self):
        """重置游戏"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        game_id = data.get('game_id')
        if not game_id:
            self.send_error(400, "Missing game_id")
            return

        if game_id in self.poker_server.games:
            del self.poker_server.games[game_id]

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            "message": "Game reset successfully",
            "valid": True
        }).encode('utf-8'))


def run_server(port=8822):
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, PokerRequestHandler)
    print(f"Starting poker server on port {port}...")
    httpd.serve_forever()


if __name__ == '__main__':
    run_server()