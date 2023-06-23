import numpy as np
import random

BOARD_SIZE = 6
N_ROWS = 1200
COLORS = ['R', 'B', 'G', 'Y', 'P']
MAX_TURNS = 100
actions = [((i + 1, j), (i, j)) for i in range(BOARD_SIZE - 1)
           for j in range(BOARD_SIZE)] + \
    [((i, j), (i, j + 1)) for i in range(BOARD_SIZE)
     for j in range(BOARD_SIZE - 1)]


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


actions = [((i + 1, j), (i, j)) for i in range(BOARD_SIZE - 1)
           for j in range(BOARD_SIZE)] + \
    [((i, j), (i, j + 1)) for i in range(BOARD_SIZE)
     for j in range(BOARD_SIZE - 1)]


class Plaser:

    def __init__(self):
        self.value = 0
        self.board = None
        self.scores = None
        self.lastscores = None  # 上次的双方分数
        self.avoidDeath = set()
        self.flag = False
        self.tolerant = 0
        self.TOLERANCE = 10  # 最多容忍的让步，决定是否采用激进的策略
        self.oppocount = 0  # 数一下对面几回合不动了

    def change(self, action):
        """在棋盘上进行一次操作"""
        (x1, y1), (x2, y2) = action
        new_board = MyBoard(self.board[:, :BOARD_SIZE * 10].copy())
        new_board.change((x1, y1), (x2, y2))
        total_score, columns_eliminated = new_board.eliminate()
        while columns_eliminated.sum() and not (new_board.mainboard == 'nan').any():
            score, columns_eliminated = new_board.eliminate()
            total_score += score
        return new_board, total_score

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

    def greedy_double(self):
        """两层贪心，返回所选的那个和得到的分数"""
        value, choice = -10000, None
        for action in self.operations:
            new_board, total_score = self.change(action)
            value_oppo = Plaser.greedy(new_board.board)[1]
            if total_score - value_oppo > value:
                value = total_score - value_oppo
                choice = action
        if choice is None:  # operation没东西可能是
            for action in actions: 
                new_board, zero_score = self.change(action)
                value_oppo = Plaser.greedy(new_board.board)[1]
                if - value_oppo > value:
                    value = - value_oppo
                    choice = action
        return choice, value

    def breakout(self):
        """突破劣势对峙境地"""
        self.flag = True
        value, choice = 10000, None
        for action in self.operations:
            new_board, total_score = self.change(action)
            loss = Plaser.greedy(new_board.board)[1] - total_score
            if loss < value:
                value = loss
                choice = action
        if value <= self.TOLERANCE:
            return choice  # 可以容忍

        # 不能容忍
        value, choice = 10000, None
        for action in actions[:40]:
            if action in self.operations or action in self.avoidDeath:
                continue
            if self.board[action[0]] == self.board[action[1]]:
                continue
            new_board, zero_score = self.change(action)
            value_oppo = Plaser.greedy(new_board.board)[1]
            if value_oppo < self.TOLERANCE:
                self.avoidDeath.add(action)
                return action  # 就这样勉强满意了，小心超时！
            if value_oppo < value:
                value = value_oppo
                choice = action
        return choice

    def select(self):
        """直接挑选最优解"""

        # A. 对峙情形/持续让步情形
        a, b = self.scores
        self.flag = False
        if self.prior:
            b += 20  # 假设对方多拿了30分
        if self.lastscores is not None:

            if self.scores == self.lastscores or self.tolerant == 1:
                if a > b + 50:  # 大优势当然保守了
                    return self.conservative()
                elif a > b:  # 小优势就争取一下，尽量扩大优势
                    choice, value = self.greedy_double()
                    if value < - self.TOLERANCE // 2:
                        return self.conservative()
                    return choice

                self.tolerant = 0
                return self.breakout()

            if self.scores[1] == self.lastscores[1]:
                self.oppocount += 1
                if self.oppocount == 2:
                    self.oppocount = 0
                    if a > b:  # 优势当然保守了
                        return self.conservative()
                    choice, value = self.greedy_double()
                    if value < -self.TOLERANCE // 2:
                        return self.conservative()
                    return choice

        # B. 正常情形
        choice, value = self.greedy_double()
        if value >= 0:
            return choice
        # 扫描亏损
        if value >= -(self.scores[0] - self.scores[1]) // 2:
            return choice
        if self.scores[0] < self.scores[1]:
            self.tolerant += 1
        return self.conservative()

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

    def check_side(self):
        """判断自己是先手还是后手"""
        if hasattr(self, "prior"):
            return
        if self.scores[0] or self.scores[1]:
            self.prior = False
        else:
            self.prior = True

    def move(self, board, operations=None, scores=None, turn_number=None):
        """评测中调用的主函数
        board: 当前棋盘
        operations: 可用操作（当然可以不给，但是要自己去试）
        """
        self.board = np.array(board)
        self.operations = operations
        self.scores = scores  # (我方，对方)
        self.check_side()
        if turn_number == MAX_TURNS:
            return self.last_round()
        if turn_number > MAX_TURNS * 0.6:
            self.TOLERANCE = 6
        if turn_number > MAX_TURNS - 5:
            self.TOLERANCE = 3
        res = self.select()
        # 保留上局
        self.lastscores = scores
        if not self.flag:
            self.avoidDeath = set()
        if res is None:  # 防止超级尴尬的情况发生导致出事...
            return random.choice(actions)
        return res
