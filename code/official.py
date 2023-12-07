import numpy as np
from math import erf, exp, sqrt, tau
from functools import lru_cache
from typing import List, Tuple

Pos = Tuple[int, int]
Pair = Tuple[Pos, Pos]

BOARD_SIZE = 6
N_ROWS = 1200
COLORS = ['R', 'B', 'G', 'Y', 'P']
MAX_TURNS = 100

actions: List[Pair] = []
for i in range(BOARD_SIZE - 1):
    for j in range(BOARD_SIZE):
        actions.extend((((i + 1, j), (i, j)), ((j, i), (j, i + 1))))


@lru_cache
def pile(*tuples) -> Tuple[float, float]:
    """
    A function to get the `E` and `std` of a list of tuples.

    Parameters:  
    - tuples: The tuples to analysis.

    Returns:
    tuple: (E, std).
    """

    if len(tuples) == 1:
        return tuples[0]
    if len(tuples) == 2:
        (m1, s1), (m2, s2) = tuples
        t = sqrt(s1 * s1 + s2 * s2)
        dev = erf((m1 - m2) / (sqrt(2) * t))
        phi = t * exp(-(m1 - m2) * (m1 - m2) / (2 * t * t)) / sqrt(tau)
        Ex = m1 * (1 + dev) / 2 + m2 * (1 - dev) / 2 + phi
        Esq = (m1 * m1 + s1 * s1) * (1 + dev) / 2 + \
            (m2 * m2 + s2 * s2) * (1 - dev) / 2 + (m1 + m2) * phi
        ans = (Ex, sqrt(Esq - Ex * Ex))
        return ans
    return pile(pile(*tuples[:len(tuples) // 2]), pile(*tuples[len(tuples) // 2:]))


class MyBoard:
    """A class with basic method to simulate the board in the game."""

    def __init__(self, board: np.ndarray):
        """
        Parameters:
        - board (array): The current board in the game.
        """

        self.size = board.shape[0]
        self.board = board

    def change(self, loc1: Pos, loc2: Pos):
        """
        Swap two chesses of the board.

        Parameters:
        - loc1, loc2: Two chesses to swap.
        """

        self.board[loc1], self.board[loc2] = \
            self.board[loc2], self.board[loc1]

    @property
    def mainboard(self) -> np.ndarray:
        """
        The mainboard of the board.

        Returns:
        array: A view of a subarray of the board, which is the mainboard (within shape of (size, size))
        """

        return self.board[:self.size, :self.size]

    def eliminate(self, func=lambda x: (x - 2) ** 2) -> Tuple[int, np.ndarray]:
        '''
        Eliminates connected elements from the mainboard and calculates the score.

        Parameters:
        - func (function): A function that takes in a group of connected elements and returns a score.

        Returns:
        tuple: A tuple that contains the total score (int) and the number of columns eliminated (array).
        '''

        # Scan the board for connected elements
        arr = self.mainboard
        to_eliminate = np.zeros((self.size, self.size), dtype=int)
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        score = 0

        # Find the repeats of the board
        to_visit = set()
        for i in range(0, BOARD_SIZE - 2):
            for j in range(BOARD_SIZE):
                if (arr[i + 1: i + 3, j] == arr[i, j]).all():
                    to_visit.add((i + 1, j))
                if (arr[j, i + 1: i + 3] == arr[j, i]).all():
                    to_visit.add((j, i + 1))

        # Check if it belongs to connected elements
        for coord in to_visit:
            if to_eliminate[coord[0], coord[1]] == 1:
                continue
            head = 0
            connected = np.array([coord])
            while head < len(connected):
                current = connected[head]
                to_eliminate[current[0], current[1]] = 1
                for d in directions:
                    neighbor = current + d
                    if (neighbor < 0).any() or (neighbor >= self.size).any():
                        continue
                    if (arr[neighbor[0], neighbor[1]] == arr[current[0], current[1]]
                            and to_eliminate[neighbor[0], neighbor[1]] == 0) and not (connected == [neighbor]).all(1).any():
                        connected = np.concatenate((connected, [neighbor]))
                head += 1
            score += func(len(connected))

        # Eliminate the columns with connected elements
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

        # Return the total score and the number of columns eliminated
        return score, col_eliminated

    def action(self, action: Pair) -> Tuple['MyBoard', int]:
        """
        Take an action on the chessboard, and get the score after the chain elimination.

        Parameters:
        - action: The action to take.

        Returns:
        tuple: A tuple contains the new board (MyBoard) and the score (int) after the elimination.
        """

        new_board = MyBoard(self.board.copy())
        new_board.change(*action)
        total_score, columns_eliminated = new_board.eliminate()
        while columns_eliminated.sum():
            score, columns_eliminated = new_board.eliminate()
            total_score += score
        return new_board, total_score

    @property
    def operations(self) -> List[Pair]:
        """
        Get all the valid operations on the scoreboard. It has a better performance than using `action`.

        Returns:
        list: A list contains all the valid operations on the board.
        """

        arr = self.board
        arrt = self.board.T
        operations = []
        for i in range(self.size - 1):
            for j in range(self.size):
                if (
                    (i <= self.size - 4 and (arr[i, j] == arr[i + 2: i + 4, j]).all()) or
                    (i >= 2 and (arr[i + 1, j] == arr[i - 2: i, j]).all()) or
                    (j >= 2 and ((arr[i, j] == arr[i + 1, j - 2: j]).all() |
                                 (arr[i + 1, j] == arr[i, j - 2: j]).all())) or
                    (j <= self.size-3 and ((arr[i, j] == arr[i + 1, j + 1: j + 3]).all() |
                                           (arr[i + 1, j] == arr[i, j + 1: j + 3]).all())) or
                    (j >= 1 and j <= self.size - 2 and ((arr[i, j] == arr[i + 1, [j - 1, j + 1]]).all() |
                                                        (arr[i + 1, j] == arr[i, [j - 1, j + 1]]).all()))
                ):
                    operations.append(((i, j), (i + 1, j)))
                if (
                    (i <= self.size - 4 and (arrt[i, j] == arrt[i + 2: i + 4, j]).all()) or
                    (i >= 2 and (arrt[i + 1, j] == arrt[i - 2: i, j]).all()) or
                    (j >= 2 and ((arrt[i, j] == arrt[i + 1, j - 2: j]).all() |
                                 (arrt[i + 1, j] == arrt[i, j - 2: j]).all())) or
                    (j <= self.size - 3 and ((arrt[i, j] == arrt[i + 1, j + 1: j + 3]).all() |
                                             (arrt[i + 1, j] == arrt[i, j + 1: j + 3]).all())) or
                    (j >= 1 and j <= self.size - 2 and ((arrt[i, j] == arrt[i + 1, [j - 1, j + 1]]).all() |
                                                        (arrt[i + 1, j] == arrt[i, [j - 1, j + 1]]).all()))
                ):
                    operations.append(((j, i), (j, i + 1)))
        return operations


class Plaser:
    """The class which will be called in the match."""

    def __init__(self, prior: bool):
        """
        Parameters:
        - prior (bool): If I am the first or not.
        """

        self.value = 0
        self.board = None
        self.scores = [-1, -1]
        self.lastscores = [-1, -1]  # the scores of the last turn
        self.prior = prior
        self.move_history = []
        self.used_time = [0, 0]

    def greedy(self, depth: int, complete: bool = False) -> Tuple[Pair, int]:
        """
        Greedy algorithm with the given depth using alpha-beta.

        Parameters:
        - depth (int): the depth for the greedy algorithm. 
            Due to the limited time, It is recommanded to be <= 2.
        - complete (bool, optional): Whether to scan all the operations (include the invalid ones).
            Default to be False. 

        Returns:
        tuple: A tuple contains the choice and the profit (maybe negative).
        """

        choice = None
        left_time = (60 - self.used_time[0]) / (101 - self.turn_number)
        select_num = 60
        if left_time < 0.3:
            select_num = 20
        elif left_time < 0.5:
            select_num = 40

        def alphabeta(board: MyBoard, d: int, alpha: int, beta: int, is_self: bool, profit: int):
            nonlocal choice, select_num

            if d == 0:
                return profit

            if complete and is_self:
                operations = (self.operations +
                              list(set(actions) - set(self.operations)))[:select_num]
            else:
                operations = self.operations if d == depth else board.operations

            if not operations:
                # There's no valid movement.
                if is_self:
                    flag = (self.scores[0] + profit) > self.scores[1]
                else:
                    flag = (self.scores[1] + profit) > self.scores[0]
                return 10000 if flag else -10000

            for action in operations:
                if d == 2 and action not in self.operations:
                    new_board = MyBoard(board.board.copy())
                    new_board.change(*action)
                    score = 0
                else:
                    new_board, score = board.action(action)
                new_profit = profit + score
                if d == 2 and new_profit <= alpha:
                    if new_profit == 0:
                        return alpha
                    continue
                value = -alphabeta(new_board, d - 1,
                                   -beta, -alpha, not is_self, -new_profit)
                if value >= beta:
                    return beta
                if value > alpha:
                    alpha = value
                    if d == depth:
                        choice = action
            return alpha

        value = alphabeta(MyBoard(self.board), depth, -10000, 10000, True, 0)
        return choice, value

    def expectation_search(self) -> Pair:
        """
        Adapted from an idea from the code `sneak in`.

        It can search for an elimination using the expectation.
        Use it when you HAVE TO eliminate, like when disadvantageous.
        As "TLE" will cause a large loss (0 vs 500), a check to avoid ending game is deleted.

        Returns:
        Pair: The movement to do by the expectation search.
        """

        value, choice = -10000, None

        mb = MyBoard(self.board)
        for action in self.operations:
            new_board, total_score = mb.action(action)
            losslist = []
            for nop in actions:
                ngain = new_board.action(nop)[1]
                losslist.append((ngain, 5))
            if max(map(lambda x: x[0], losslist)) == 0:
                if total_score >= self.scores[1] - self.scores[0]:
                    return action
                else:
                    continue
            Eloss, Sloss = pile(*losslist)
            if total_score - Eloss + Sloss > value:
                value = total_score - Eloss + Sloss
                choice = action

        return choice

    def conservative(self) -> Pair:
        """
        A very conservative strategy.
        It will swap two same chesses, which means do nothing on the board.

        Returns:
        Pair: A conservative movement.
        """

        for action in actions:
            if self.board[action[0]] == self.board[action[1]]:
                return action

        # No same neighbor (low possibility), search for a safe movement
        for action in actions:
            if actions not in self.operations:
                return action

    def destroy(self) -> Pair | None:
        """
        Try to end the game and win.

        Returns:
        Pair | None: If there is a movement can end the game, return it. Otherwise, return None. 
        """

        choice = None
        mb = MyBoard(self.board)
        for action in actions:
            if action not in self.operations:
                mb.change(*action)
                newop = mb.operations
                if newop == []:  # Get it, and return
                    return action
                mb.change(*action)
        return choice

    def last_round(self) -> Pair:
        """
        Special movement for the last round, usually with greedy algorithm.

        Returns:
        Pair: The movement for the last round.
        """

        if not self.prior:
            return self.greedy(1)[0]
        choice, value = self.greedy(2)
        if self.scores[0] + value > self.scores[1]:
            return choice  # win
        elif self.scores[0] > self.scores[1]:
            return self.conservative()  # lose
        else:
            return choice  # lose

    def select(self) -> Pair:
        """
        The core method to find the movement.

        Returns:
        Pair: The movement for this turn.
        """

        if self.turn_number == MAX_TURNS:
            return self.last_round()

        if self.turn_number == 1 and self.prior:
            action, value = self.greedy(2)
            if value < -7:
                return self.conservative()
            return action

        if self.scores[0] > self.scores[1]:  # advantagous
            if self.scores[1] == self.lastscores[1]:
                # Undo the movement by the opponent
                return self.move_history[-1]

            if len(self.operations) < 3:
                # Try to end the game
                choice = self.destroy()
                if choice is not None:
                    return choice

            return self.greedy(2, complete=True)[0]

        else:  # disadvantageous
            return self.expectation_search()

    def move(self, board: List, operations: List[Pair],
             scores: List[int, int], turn_number: int) -> Pair:
        """
        The main function when playing the game.

        Parameters:
        - board (list): The current board, including those hidden chesses.
        - operations (list): All valid operations on the current board.
        - scores (list): A list contains the scores of both sides.
        - turn_number (int): The current turn number.

        Returns:
        Pair: The movement decided in `select`.
        """

        self.board = np.array(board)[:, :BOARD_SIZE * 10]
        self.operations = operations
        self.scores = scores
        self.turn_number = turn_number
        res = self.select()
        self.lastscores = scores.copy()
        return res
