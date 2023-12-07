import numpy as np
import random
from math import erf, exp, sqrt, tau
from functools import lru_cache

BOARD_SIZE = 6
N_ROWS = 1200
COLORS = ['R', 'B', 'G', 'Y', 'P']
MAX_TURNS = 100

actions = []
for i in range(BOARD_SIZE - 1):
    for j in range(BOARD_SIZE):
        actions.extend((((i + 1, j), (i, j)), ((j, i), (j, i + 1))))


@lru_cache
def pile(*tuples):
    if len(tuples) == 1:
        return tuples[0]
    if len(tuples) == 2:
        (m1, s1), (m2, s2) = tuples
        t = sqrt(s1*s1+s2*s2)
        dev = erf((m1-m2)/(sqrt(2)*t))
        phi = t*exp(-(m1-m2)*(m1-m2)/(2*t*t))/sqrt(tau)
        Ex = m1*(1+dev)/2+m2*(1-dev)/2+phi
        Esq = (m1*m1+s1*s1)*(1+dev)/2+(m2*m2+s2*s2)*(1-dev)/2+(m1+m2)*phi
        ans = (Ex, sqrt(Esq-Ex*Ex))
        return ans
    return pile(pile(*tuples[:len(tuples)//2]), pile(*tuples[len(tuples)//2:]))


class MyBoard:
    def __init__(self, board: np.ndarray):
        self.size = board.shape[0]
        self.board = board

    def change(self, loc1, loc2):
        x1, y1 = loc1
        x2, y2 = loc2
        self.board[x1, y1], self.board[x2, y2] = \
            self.board[x2, y2], self.board[x1, y1]

    @staticmethod
    def check(arr):
        repeats = set()
        for i in range(0, BOARD_SIZE - 2):
            for j in range(BOARD_SIZE):
                if arr[i, j] != 'nan' and (arr[i+1:i+3, j] == arr[i, j]).all():
                    repeats.add((i+1, j))
                if arr[j, i] != 'nan' and (arr[j, i+1:i+3] == arr[j, i]).all():
                    repeats.add((j, i+1))
        return repeats

    def eliminate(self, func=lambda x: (x - 2) ** 2):
        arr = self.board
        to_eliminate = np.zeros((self.size, self.size), dtype=int)
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        to_visit = self.check(arr)
        score = 0

        for coord in to_visit:
            if to_eliminate[coord[0], coord[1]] == 1:
                continue
            head = 0
            connected = [coord, ]
            while head < len(connected):
                current = connected[head]
                to_eliminate[current[0], current[1]] = 1
                for d in directions:
                    neighbor = current + d
                    if (neighbor < 0).any() or (neighbor >= self.size).any():
                        continue
                    if (arr[neighbor[0], neighbor[1]] == arr[current[0], current[1]]
                            and to_eliminate[neighbor[0], neighbor[1]] == 0):
                        connected.append(neighbor)
                head += 1
            score += func(len(connected))

        col_eliminated = np.sum(to_eliminate, axis=1)
        col_remained = self.size - col_eliminated
        for i in range(self.size):
            if col_eliminated[i] == 0:
                continue
            col = self.board[i]
            self.board[i, :col_remained[i]
                       ] = col[:self.size][to_eliminate[i] == 0]
            self.board[i, col_remained[i]:self.board.shape[1] -
                       col_eliminated[i]] = col[self.size:]
            self.board[i, self.board.shape[1] - col_eliminated[i]:] = np.nan

        # Return the total score and the number of columns eliminated
        return score, col_eliminated

    def action(self, action):
        """测试一个操作后经历完连锁反应后的效果，其中有拷贝不会影响原来的棋盘"""
        new_board = MyBoard(self.board.copy())
        new_board.change(*action)
        total_score, columns_eliminated = new_board.eliminate()
        while columns_eliminated.sum() and not (new_board.board[:BOARD_SIZE, :BOARD_SIZE] == 'nan').any():
            score, columns_eliminated = new_board.eliminate()
            total_score += score
        return new_board.board, total_score

    @property
    def operations(self):
        """查找所有可以消除的操作"""
        arr = self.board
        arrt = self.board.T
        operations = []
        for i in range(self.size - 1):
            for j in range(self.size):
                if (
                    (i <= self.size-4 and (arr[i, j] == arr[i+2:i+4, j]).all()) or
                    (i >= 2 and (arr[i+1, j] == arr[i-2:i, j]).all()) or
                    (j >= 2 and ((arr[i, j] == arr[i+1, j-2:j]).all() |
                                 (arr[i+1, j] == arr[i, j-2:j]).all())) or
                    (j <= self.size-3 and ((arr[i, j] == arr[i+1, j+1:j+3]).all() |
                                           (arr[i+1, j] == arr[i, j+1:j+3]).all())) or
                    (j >= 1 and j <= self.size-2 and ((arr[i, j] == arr[i+1, [j-1, j+1]]).all() |
                                                      (arr[i+1, j] == arr[i, [j-1, j+1]]).all()))
                ):
                    operations.append(((i, j), (i+1, j)))
                if (
                    (i <= self.size-4 and (arrt[i, j] == arrt[i+2:i+4, j]).all()) or
                    (i >= 2 and (arrt[i+1, j] == arrt[i-2:i, j]).all()) or
                    (j >= 2 and ((arrt[i, j] == arrt[i+1, j-2:j]).all() |
                                 (arrt[i+1, j] == arrt[i, j-2:j]).all())) or
                    (j <= self.size-3 and ((arrt[i, j] == arrt[i+1, j+1:j+3]).all() |
                                           (arrt[i+1, j] == arrt[i, j+1:j+3]).all())) or
                    (j >= 1 and j <= self.size-2 and ((arrt[i, j] == arrt[i+1, [j-1, j+1]]).all() |
                                                      (arrt[i+1, j] == arrt[i, [j-1, j+1]]).all()))
                ):
                    operations.append(((j, i), (j, i+1)))
        return operations


class Plaser:

    def __init__(self, prior=True):
        self.value = 0
        self.board = None
        self.scores = (-1, -1)
        self.lastscores = (-1, -1)  # 上次的双方分数
        self.prior = prior

    def greedy(self, depth):
        """贪心算法，depth为深度"""

        choice = None

        def alphabeta(board, d, alpha, beta, is_self, profit):
            nonlocal choice

            if d == 0:
                return profit
            myboard = MyBoard(board)
            operations = self.operations if d == depth else myboard.operations

            if not operations:
                # 特殊情况，没有可以消除的
                if is_self:
                    flag = (self.scores[0] + profit) > self.scores[1]
                else:
                    flag = (self.scores[1] + profit) > self.scores[0]
                return 10000 if flag else -10000

            for action in operations:
                new_board, score = myboard.action(action)
                new_profit = profit + score
                value = - alphabeta(new_board, d - 1,
                                    -beta, -alpha, not is_self, -new_profit)
                if value >= beta:
                    return beta
                if value > alpha:
                    alpha = value
                    if d == depth:  # 回溯到最表层的时候，记得记录选择的那个操作
                        choice = action
            return alpha

        value = alphabeta(self.board, depth, -10000, 10000, True, 0)
        return choice, value

    def expectation_search(self):
        """来自sneak in的期望最大化搜索，目前用于劣势情形"""

        value, choice = -10000, None
        if not self.operations:
            return random.choice(actions)  # 必输

        mb = MyBoard(self.board)
        for action in self.operations:
            new_board, total_score = mb.action(action)
            losslist = []
            ob = MyBoard(new_board)
            for nop in actions:
                ngain = ob.action(nop)[1]
                losslist.append((ngain, 5))
            if total_score >= self.scores[1] - self.scores[0] and max(map(lambda x: x[0], losslist)) == 0:
                return action
            Eloss, Sloss = pile(*losslist)
            if total_score - Eloss + Sloss > value:
                choice = action
                value = total_score - Eloss + Sloss
        return choice

    def conservative(self):
        """保守策略，交换一样的棋子，让一先"""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return ((i, j), (i, j + 1))
        for i in range(BOARD_SIZE - 1):
            for j in range(BOARD_SIZE):
                if self.board[i, j] == self.board[i + 1, j]:
                    return ((i, j), (i + 1, j))
        # 一对一样颜色的邻居都没有啊？也够非酋的，这种概率非常非常低
        for action in actions:
            if actions not in self.operations:
                return action  # 尽量别产生消除出乱子吧

    def end_game(self):
        """优势下，企图结束游戏，没有成功就返回None"""
        mb = MyBoard(self.board)
        for action in actions:
            if action not in self.operations:
                mb.change(*action)
                if mb.operations == []:  # 逮到
                    return action
                mb.change(*action)  # 交换回来

    def last_round(self):
        """最后一回合的冲刺"""
        if not self.prior:  # 最后一手，当然无脑贪心了
            return self.greedy(1)[0]
        # 先手，观察局势
        choice, value = self.greedy(2)
        if self.scores[0] + value > self.scores[1]:
            return choice  # 稳赢了
        elif self.scores[0] > self.scores[1]:
            return self.conservative()  # 赌一下对面忘了贪心这回事吧
        else:
            return choice  # 彻底没救了

    def select(self):
        """直接挑选最优解"""
        if self.turn_number == MAX_TURNS:
            return self.last_round()

        if self.scores[0] > self.scores[1]:  # 优势局面
            if self.scores[1] == self.lastscores[1]:
                # 对方上回合没有消除
                return self.move_history[-1]
            action = self.end_game()
            if action is not None:
                # 成功带领游戏结束
                return action

            # 不能卡死，老实双层贪心：
            action, value = self.greedy(2)
            tolerance = self._get_tolerance()
            if self.scores[0] - self.scores[1] > 30:
                if value > - tolerance:
                    return action
                return self.conservative()  # 优势大了就也没必要去牺牲自己了
            else:
                return action

        else:  # 劣势局面
            return self.expectation_search()

    def _get_tolerance(self):
        """得到容忍度"""
        if self.turn_number < MAX_TURNS * 0.3:
            return 15
        else:
            rate = self.turn_number / MAX_TURNS
            res = 10 + int(rate * (self.scores[1] - self.scores[0]) / 3)
            return max(res, 3)

    def move(self,
             board,
             operations=None,
             scores=None,
             turn_number: int = None
             ):
        """评测中调用的主函数
        board: 当前棋盘
        operations: 可用操作（当然可以不给，但是要自己去试）
        """
        self.board = np.array(board)[:, :BOARD_SIZE * 20]
        self.operations = operations
        self.scores = scores  # (我方，对方)
        self.turn_number = turn_number
        res = self.select()

        if res is None:  # 防止超级尴尬的情况发生导致出事...
            res = random.choice(actions)

        # 保留上局
        self.lastscores = scores.copy()
        return res
