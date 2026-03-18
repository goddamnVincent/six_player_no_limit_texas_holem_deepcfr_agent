# no_limit_6_players_texas_holdem_deepcfr_AI_agent
a six-player Texas Hold'em poker agent based on DeepCFR
# 德州扑克AI
- 这是一个基于deepcfr框架魔改的六人桌德州扑克agent
- 一切都是为了落地!请记住本项目的宗旨!一切都是为了方便落地!
# 项目介绍以及主要改进点
- 六人桌无限注，起始筹码是200，小盲1，大盲2。暂时不支持改，除非重新训练。
- 合法动作:fold/ check/call/ raise '0.5pot' / raise '1.0pot' /raise 'all-in' .为什么不做成连续的呢?连续值比较难学，因为是一个回归问题，分类比较简单学。而且从实际落地来讲，这三挡位完全够用了。
- 训练细节:使用的仍然是deepcfr这套框架，整体分为两个阶段，第一个阶段是从scratch出发，和一个我自己写的逻辑机器人进行对战，这个逻辑机器人还是很复杂的。第一阶段的目标是为了建立对手牌强度的大致认知，也就是知道什么是大牌，什么是小牌。第二阶段训练就是自博弈阶段了，也就是强化学习。第二阶段不断地和不同iteration阶段保存的agent进行对战，始终读取最新的五个agent进行对战。
