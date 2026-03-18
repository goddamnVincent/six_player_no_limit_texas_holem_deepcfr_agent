document.addEventListener('DOMContentLoaded', function() {
    // 游戏状态变量
    let currentGameId = null;
    let currentPlayer = null;
    let legalActions = [];

    // DOM元素
    const newGameBtn = document.getElementById('newGameBtn');
    const resetGameBtn = document.getElementById('resetGameBtn');
    const gameStatus = document.getElementById('gameStatus');
    const currentPlayerDisplay = document.getElementById('currentPlayer');
    const potSizeDisplay = document.getElementById('potSize');
    const communityCardsContainer = document.querySelector('#communityCards .cards-container');
    const actionButtons = document.getElementById('actionButtons');
    const raiseControls = document.getElementById('raiseControls');
    const raiseAmountInput = document.getElementById('raiseAmount');
    const confirmRaiseBtn = document.getElementById('confirmRaise');
    const gameLog = document.querySelector('.log-entries');

    // 玩家DOM元素
    const players = {
        player1: {
            element: document.getElementById('player1'),
            cardsContainer: document.querySelector('#player1 .cards-container'),
            chipsDisplay: document.querySelector('#player1 .player-chips span'),
            betDisplay: document.querySelector('#player1 .player-bet span')
        },
        player2: {
            element: document.getElementById('player2'),
            cardsContainer: document.querySelector('#player2 .cards-container'),
            chipsDisplay: document.querySelector('#player2 .player-chips span'),
            betDisplay: document.querySelector('#player2 .player-bet span')
        },
        player3: {
            element: document.getElementById('player3'),
            cardsContainer: document.querySelector('#player3 .cards-container'),
            chipsDisplay: document.querySelector('#player3 .player-chips span'),
            betDisplay: document.querySelector('#player3 .player-bet span')
        }
    };

    // 事件监听器
    newGameBtn.addEventListener('click', startNewGame);
    resetGameBtn.addEventListener('click', resetGame);

    // 动作按钮事件
    document.querySelectorAll('.action-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const action = this.dataset.action;
            if (action === 'raise') {
                showRaiseControls();
            } else {
                sendPlayerAction(action);
            }
        });
    });

    confirmRaiseBtn.addEventListener('click', function() {
        const amount = parseInt(raiseAmountInput.value);
        if (amount > 0) {
            sendPlayerAction('raise', amount);
            hideRaiseControls();
        }
    });

    // 辅助函数：显示/隐藏加注控件
    function showRaiseControls() {
        raiseControls.style.display = 'flex';
    }

    function hideRaiseControls() {
        raiseControls.style.display = 'none';
    }

    // 开始新游戏
    function startNewGame() {
        const gameId = 'game_' + Date.now();
        currentGameId = gameId;

        fetch('http://192.168.105.228:8822/create_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ game_id: gameId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                logMessage('创建游戏失败: ' + data.error);
                return;
            }

            updateGameState(data);
            logMessage('新游戏已创建! 游戏ID: ' + gameId);
        })
        .catch(error => {
            logMessage('创建游戏时出错: ' + error);
        });
    }

    // 重置游戏
    function resetGame() {
        if (!currentGameId) {
            logMessage('没有正在进行的游戏可以重置');
            return;
        }

        fetch('http://192.168.105.228:8822/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ game_id: currentGameId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.valid) {
                logMessage('游戏已重置');
                clearGameBoard();
                gameStatus.textContent = '游戏已重置，点击"新游戏"开始';
            } else {
                logMessage('重置游戏失败');
            }
        })
        .catch(error => {
            logMessage('重置游戏时出错: ' + error);
        });
    }

    // 发送玩家动作
    function sendPlayerAction(action, amount = null) {
        if (!currentGameId) {
            logMessage('没有正在进行的游戏');
            return;
        }

        const actionData = {
            game_id: currentGameId,
            action: action
        };

        if (action === 'raise' && amount) {
            actionData.amount = amount;
        }

        fetch('http://192.168.105.228:8822/action', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(actionData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                logMessage('动作无效: ' + data.error);
                return;
            }

            updateGameState(data);

            if (data.ai_actions) {
                data.ai_actions.forEach(aiAction => {
                    logMessage(`玩家 ${aiAction.player} ${aiAction.action}`);
                });
            }

            if (data.game_over) {
                logMessage(`游戏结束! 赢家: ${data.winners.join(', ')}`, true);
            }
        })
        .catch(error => {
            logMessage('执行动作时出错: ' + error);
        });
    }

    // 更新游戏状态
    function updateGameState(data) {
        // 更新游戏状态显示
        if (data.game_over) {
            gameStatus.textContent = `游戏结束! 赢家: ${data.winners.join(', ')}`;
            gameStatus.classList.add('highlight');
        } else {
            gameStatus.textContent = '游戏进行中';
            gameStatus.classList.remove('highlight');
        }

        // 更新当前玩家
        if (data.next_player) {
            currentPlayer = data.next_player;
            currentPlayerDisplay.textContent = `当前玩家: ${currentPlayer}`;

            // 高亮显示当前玩家
            Object.values(players).forEach(player => {
                player.element.classList.remove('highlight');
            });

            if (currentPlayer.toLowerCase() === 'human') {
                players.player1.element.classList.add('highlight');
            } else if (currentPlayer.includes('2')) {
                players.player2.element.classList.add('highlight');
            } else if (currentPlayer.includes('3')) {
                players.player3.element.classList.add('highlight');
            }
        }

        // 更新合法动作
        if (data.legal_actions) {
            legalActions = data.legal_actions;
            updateActionButtons(legalActions);
        }

        // 更新底池大小
        if (data.pot) {
            potSizeDisplay.textContent = data.pot;
        }

        // 更新公共牌
        if (data.community_cards) {
            renderCards(communityCardsContainer, data.community_cards);
        }

        // TODO: 更新玩家手牌和筹码信息
        // 在实际应用中，您需要从后端获取这些信息并更新UI
    }

    // 更新动作按钮
    function updateActionButtons(actions) {
        document.querySelectorAll('.action-btn').forEach(btn => {
            const action = btn.dataset.action;
            btn.disabled = !actions.includes(action);
        });
    }

    // 渲染牌
    function renderCards(container, cards) {
        container.innerHTML = '';

        if (!cards || cards.length === 0) {
            const placeholder = document.createElement('div');
            placeholder.className = 'card';
            placeholder.textContent = '?';
            container.appendChild(placeholder);
            return;
        }

        cards.forEach(card => {
            const cardElement = document.createElement('div');
            cardElement.className = 'card';

            // 简化显示，实际应该根据牌值显示对应的花色和点数
            cardElement.textContent = card || '?';
            container.appendChild(cardElement);
        });
    }

    // 清空游戏板
    function clearGameBoard() {
        communityCardsContainer.innerHTML = '';

        Object.values(players).forEach(player => {
            player.cardsContainer.innerHTML = '';
            player.chipsDisplay.textContent = '1000';
            player.betDisplay.textContent = '0';
        });

        gameStatus.textContent = '等待游戏开始...';
        currentPlayerDisplay.textContent = '';
        potSizeDisplay.textContent = '0';
    }

    // 记录游戏消息
    function logMessage(message, isImportant = false) {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        if (isImportant) {
            entry.classList.add('highlight');
        }
        entry.textContent = message;
        gameLog.appendChild(entry);
        gameLog.scrollTop = gameLog.scrollHeight;
    }

    // 初始化
    clearGameBoard();
    hideRaiseControls();
});