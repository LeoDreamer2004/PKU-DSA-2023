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
        self.board = board.copy()

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

    @property
    def mainboard(self):
        return self.board[:BOARD_SIZE, :BOARD_SIZE]


class Plaser:

    def __init__(self, prior=True):
        self.value = 0
        self.board = None
        self.lastin = None  # 上次我收到的棋盘
        self.lastout = None  # 上次我传出的棋盘
        self.scores = (-1, -1)
        self.lastscores = (-1, -1)  # 上次的双方分数
        self.prior = prior

    def change(self, action):
        """在棋盘上进行一次操作"""
        (x1, y1), (x2, y2) = action
        new_board = MyBoard(self.board.copy())
        new_board.change((x1, y1), (x2, y2))
        total_score, columns_eliminated = new_board.eliminate()
        while columns_eliminated.sum() and not (new_board.mainboard == 'nan').any():
            score, columns_eliminated = new_board.eliminate()
            total_score += score
        return new_board.board, total_score

    @staticmethod
    def greedy(board: np.ndarray):
        """给定棋盘的贪心...在不知道operations的前提下"""
        temp = Plaser()
        temp.board = board
        value, choice = -1, None
        for action in actions:
            new_board, score = temp.change(action)
            if score > value:
                value = score
                choice = action
        return choice, value

    @staticmethod
    def check_connect(board: np.ndarray):
        """检查棋盘是否有能消除的地方，即为检查死局做准备"""
        arr = board[:BOARD_SIZE, :BOARD_SIZE]
        arrt = arr.T
        for i in range(BOARD_SIZE - 1):
            for j in range(BOARD_SIZE):
                if i <= BOARD_SIZE - 4 and (arr[i, j] == arr[i+2:i+4, j]).all():
                    return True
                if i >= 2 and (arr[i+1, j] == arr[i-2:i, j]).all():
                    return True
                if j >= 2 and ((arr[i, j] == arr[i+1, j-2:j]).all() |
                               (arr[i+1, j] == arr[i, j-2:j]).all()):
                    return True
                if j <= BOARD_SIZE - 3 and ((arr[i, j] == arr[i+1, j+1:j+3]).all() |
                                            (arr[i+1, j] == arr[i, j+1:j+3]).all()):
                    return True
                if j >= 1 and j <= BOARD_SIZE - 2 and ((arr[i, j] == arr[i+1, [j-1, j+1]]).all() |
                                                       (arr[i+1, j] == arr[i, [j-1, j+1]]).all()):
                    return True
        for i in range(BOARD_SIZE - 1):
            for j in range(BOARD_SIZE):
                if i <= BOARD_SIZE - 4 and (arrt[i, j] == arrt[i+2:i+4, j]).all():
                    return True
                if i >= 2 and (arrt[i+1, j] == arrt[i-2:i, j]).all():
                    return True
                if j >= 2 and ((arrt[i, j] == arrt[i+1, j-2:j]).all() |
                               (arrt[i+1, j] == arrt[i, j-2:j]).all()):
                    return True
                if j <= BOARD_SIZE - 3 and ((arrt[i, j] == arrt[i+1, j+1:j+3]).all() |
                                            (arrt[i+1, j] == arrt[i, j+1:j+3]).all()):
                    return True
                if j >= 1 and j <= BOARD_SIZE - 2 and ((arrt[i, j] == arrt[i+1, [j-1, j+1]]).all() |
                                                       (arrt[i+1, j] == arrt[i, [j-1, j+1]]).all()):
                    return True
        return False

    def conservative(self):
        """保守策略"""
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

    def suppress(self):
        """优势，对方上回合没得分...镇压对手！"""
        if (self.lastout[:, :BOARD_SIZE] == self.board[:, :BOARD_SIZE]).all():
            return self.conservative()
        res = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i, j] != self.lastout[i, j]:
                    res.append((i, j))
        return tuple(res)

    def greedy_double(self):
        """两层贪心，返回所选的那个和得到的分数"""
        value, choice = -10000, None
        for action in self.operations:
            new_board, total_score = self.change(action)
            value_oppo = Plaser.greedy(new_board)[1]
            if value_oppo == 0:
                if self.scores[0] + total_score > self.scores[1]:
                    return action, 10000
                elif self.scores[0] + total_score < self.scores[1]:
                    continue
            if total_score - value_oppo > value:
                value = total_score - value_oppo
                choice = action
        return choice, value
    
    """
    def breakout(self):
        # 突破劣势对峙境地
        self.flag = True
        value, choice = 10000, None
        for action in self.operations:
            new_board, total_score = self.change(action)
            value_oppo = Plaser.greedy(new_board)[1]
            if value_oppo == 0:
                if self.scores[0] + total_score > self.scores[1]:
                    return action, 10000  # 必胜！
                continue  # 防止GG
            if value_oppo - total_score < value:
                value = value_oppo - total_score
                choice = action
        if value <= self.TOLERANCE:
            return choice  # 可以容忍

        # 不能容忍
        value, choice = 10000, None
        for action in actions[:10]:
            if action in self.operations or action in self.avoidDeath:
                continue
            if self.board[action[0]] == self.board[action[1]]:
                 continue
            new_board, zero_score = self.change(action)
            value_oppo = Plaser.greedy(new_board)[1]
            if value_oppo < self.TOLERANCE:
                self.avoidDeath.add(action)
                return action  # 就这样勉强满意了，小心超时！
            if value_oppo < value:
                value = value_oppo
                choice = action
        return choice
    """

    def select(self):
        """直接挑选最优解"""
        if self.turn_number == MAX_TURNS:
            return self.last_round()
        if self.scores[0] > self.scores[1]:
            return self.advantageous()
        else:
            return self.disadvantageous(self.scores[1] - self.scores[0])

    def advantageous(self):
        """优势局面"""
        # STEP1: 是否能卡死对面？
        if self.scores[1] == self.lastscores[1]:
            # 1.1 对方上回合没有消除
            return self.suppress()

        if not self.operations:
            # 1.2 没有可消除的东西
            return self.conservative()

        # 1.3 检查移动后有没有能导致无消除的——低复杂度？
        # STEP2: 不能卡死，老实双层贪心：
        action, value = self.greedy_double()
        tolerance = self._get_tolerance()

        if value > - tolerance:
            return action
        return self.conservative()
    
    def disadvantageous(self, gap):
        """劣势局面"""
        value, choice = -10000, None
        if not self.operations:
            # 必输
            return random.choice(actions)
        for action in self.operations:
            new_board, total_score = self.change(action)
            losslist = []
            for nop in actions:
                oppo_board = Plaser()
                oppo_board.board = new_board.copy()
                ngain = oppo_board.change(nop)[1]
                losslist.append((ngain, 5))
            if total_score >= gap and max(map(lambda x:x[0], losslist)) == 0:
                return action
            Eloss, Sloss = pile(*losslist)
            if total_score - Eloss + Sloss > value:
                choice = action
                value = total_score - Eloss + Sloss
        return choice

    def last_round(self):
        """最后一回合的冲刺"""
        if not self.prior:  # 最后一手，当然无脑贪心了
            return Plaser.greedy(self.board)[0]
        # 先手，观察局势
        choice, value = self.greedy_double()
        if self.scores[0] + value > self.scores[1]:
            return choice  # 稳赢了
        elif self.scores[0] > self.scores[1]:
            return self.conservative()  # 赌一下对面忘了贪心这回事吧
        else:
            return choice  # 彻底没救了

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
             turn_number: int = None):
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
        self.lastin = self.board
        self.lastout, score = self.change(res)

        return res
