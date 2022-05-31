import random as R

import numpy as np


# Naama Omer 207644014, Moshe Zeev Hefter


def raffle(p):
    """
    this function chooses 1 (true) given the probability P.
    :param p: - probability
    :return: either 1 or 0 based on chance.
    """
    return np.random.choice((0, 1), p=[1 - p, p])


class Board:

    def __init__(self, file):
        """
        this function reads the input file in the specified format, and initiates the staring values for the game.
        :param: file: txt file.
        """
        with open(file) as fp:
            # size of the Futoshiki map.
            self.board_size = int(fp.readline())
            # empty map->for immutable values
            self.starting_values_matrix = np.zeros((self.board_size, self.board_size))
            # the number of immutable placed values.
            num_start_values = int(fp.readline())
            # putting starting values int map.
            for value in range(num_start_values):
                line_start_val = np.fromstring(fp.readline(), dtype=int, sep=' ')
                self.starting_values_matrix[line_start_val[0] - 1][line_start_val[1] - 1] = line_start_val[2]
            num_conditions = int(fp.readline())
            # matrix of all the conditions number of conditions x 4.
            self.conditions = np.zeros((num_conditions, 4))
            for sign in range(num_conditions):
                line = np.fromstring(fp.readline(), dtype=int, sep=' ') - 1
                for i in range(4):
                    self.conditions[sign][i] = line[i]

    def copy(self):
        """
        this function returns a deep copy of the numpy part of the map, with the immutable variables.
        :return: np array.
        """
        return np.copy(self.starting_values_matrix)


class Solution:
    def __init__(self, board=None, matrix=None, condition=None, size=None):
        """
        this function initiates a new solution, either from scratch, given only the Board object, or from a numpy array
        with the size and conditions for the Board construction.
        :param: board: Board object.
        :param: matrix: np array with correct values to be cultivated into a fully defined solution.
        :param: condition: the list of conditions.
        :param: size: the size of a single dimension of the numpy array.
        """
        # init from values.
        if board is None:
            self.board = matrix
            self.board_size = size
            self.conditions = condition
            self.evaluation = 0

        # init from scratch.
        else:
            self.board = board.copy()  # the starting map.
            self.board_size = board.board_size
            self.evaluation = 0
            self.conditions = board.conditions.astype(int)

    def set_new_solution(self):
        """
        this function will start a new board with legal lines, and correct number
        of each number.
        :return: none
        """
        for i in range(self.board_size):
            for var in range(1, self.board_size + 1):
                if var in self.board[i]:
                    continue
                spot = R.randint(0, self.board_size - 1)
                while not self.board[i][spot] == 0:
                    spot = R.randint(0, self.board_size - 1)
                self.board[i][spot] = var

    def evaluate(self):
        """
        this function evaluates how close a solution is to being correct using a score of the sum of errors.
        errors or calculated by adding the output of on 2 test:
        1. the sum amount of missing values in each column
        2. the sum amount of conditions that weren't upheld.
        -> we insured that each row holds all distinct values, such that the amount missing would be zero, and
        unnecessary to count.
        :return: none.
        """
        self.evaluation = 0
        # the sum amount of missing values in each column
        for i in range(self.board_size):
            self.evaluation += self.board_size - np.unique(self.board[:, i]).size
        for con in self.conditions[:, ]:
            # The coordinates of the two cells with the > sign between them, so that we add the apposite case
            self.evaluation += (self.board[con[0]][con[1]] < self.board[con[2]][con[3]])

    def optimiser(self, starting_matrix, board_size):
        """
        :return:
        """
        # go over on half on the conditions - random choices
        # for each choice switch two values within a line if legal
        # 1. Random
        temp_arr = np.array(range(len(self.conditions)))
        np.random.shuffle(temp_arr)
        idx = 0
        try_long = False
        flag = True
        cond_num = 0
        while flag and idx != len(temp_arr):
            current_condition = self.conditions[idx]
            row_cell_left = current_condition[0]
            row_cell_right = current_condition[2]
            column_cell_left = current_condition[1]
            column_cell_right = current_condition[3]

            val_left = self.board[row_cell_left][column_cell_left]
            val_right = self.board[row_cell_right][column_cell_right]
            left_is_mutable = starting_matrix[row_cell_left][column_cell_left] == 0
            right_is_mutable = starting_matrix[row_cell_right][column_cell_right] == 0
            # if not >:
            if val_left < val_right:
                # same row:
                if row_cell_left == row_cell_right:
                    # and can to be moved:
                    if left_is_mutable and right_is_mutable:
                        # swap
                        self.board[row_cell_left][column_cell_left] = val_right
                        self.board[row_cell_right][column_cell_right] = val_left
                        # successes:
                        cond_num += 1
                    # if only left:
                    elif left_is_mutable:
                        # same row
                        index_switch = np.argwhere(self.board[row_cell_left] >= val_right + 1)
                        no_other = index_switch.shape[0] == 0  # not found, pass
                        if no_other:
                            try_long = True
                        else:
                            for column in index_switch[0]:
                                if starting_matrix[row_cell_left][column] == 0:
                                    temp = self.board[row_cell_left][column]
                                    self.board[row_cell_left][column] = val_left
                                    self.board[row_cell_left][column_cell_left] = temp
                                    # successes:
                                    cond_num += 1
                                    break
                                else:
                                    pass
                    else:
                        # same row
                        index_switch = np.argwhere(self.board[row_cell_right] <= val_left - 1)
                        no_other = index_switch.shape[0] == 0  # not found
                        if no_other:
                            try_long = True
                        else:
                            for column in index_switch[0]:
                                if starting_matrix[row_cell_right][column] == 0:
                                    temp = self.board[row_cell_right][column]
                                    self.board[row_cell_left][column] = val_right
                                    self.board[row_cell_right][column_cell_right] = temp
                                    # successes:
                                    cond_num += 1
                                    break
                                else:
                                    pass
                else:
                    # try to fix condition in columns
                    if try_long:
                        # Arranged longitudinally, in the same column but not in the same row
                        # switch_rows
                        if left_is_mutable and right_is_mutable:
                            valid = True
                            for i in range(self.board_size):
                                if starting_matrix[row_cell_left][i] == 0:
                                    valid = False
                            for i in range(self.board_size):
                                if starting_matrix[row_cell_right][i] == 0:
                                    valid = False
                            if valid:
                                self.board[[row_cell_left, row_cell_right]] = self.board[
                                    [row_cell_right, row_cell_left]]
                                # successes:
                                cond_num += 1
                            else:
                                pass
            else:
                pass
            if cond_num == board_size:
                flag = False
            idx += 1


class Futoshiki_Solver:
    def __init__(self, board):
        """
        initiates the game with the relevant data. is responsible to create new solutions.
        :param board: -board object
        """
        self.empty_board = board.starting_values_matrix
        self.solutions = []
        self.board_size = board.board_size
        evaluations = []
        # create 100 new solutions.
        for i in range(100):
            solution_baby = Solution(board=board)
            solution_baby.set_new_solution()
            solution_baby.evaluate()
            self.solutions.append(solution_baby)
            evaluations.append(solution_baby.evaluation)
        self.evaluations = np.array(evaluations)

        # order the solutions by best grade.
        zipped = list(zip(self.solutions, self.evaluations))
        res = sorted(zipped, key=lambda x: x[1])
        self.solutions, self.evaluations = zip(*res)

    def selection(self):
        """
        this function selects the best solutions, sends them for cross over, and sets new solutions.
        :return:
        """
        new_solutions = [None] * 100
        new_evaluations = [None] * 100
        # get rid of the worse half of the solutions, and evaluations.
        self.solutions = self.solutions[:50]
        self.evaluations = self.evaluations[:50]
        # select the best 10 examples unchanged and for the next selection.
        new_solutions[:7] = self.solutions[:7]
        new_evaluations[:7] = self.evaluations[:7]
        # create a cross-over from the best solutions, to fill back up to 100 solutions.

        for idx in range(7, 100):
            # biased selection of indices of the solutions chosen for cross-over.
            fifty_arr = np.arange(0, 50)
            p = np.arange(0.0298, 0.010, -0.0004)
            a = np.random.choice(a=fifty_arr, p=p, size=2)
            first, second = a[0], a[1]
            # single solution matrix returned.
            solution_matrix = self.crossover(first, second)
            condition = self.solutions[0].conditions
            # creating a new solution object.
            sol = Solution(matrix=solution_matrix, condition=condition, size=self.board_size)
            sol.evaluate()
            new_solutions[idx] = sol
            new_evaluations[idx] = sol.evaluation

        # integrate the new solutions to the game, in an ordered fashion.
        zipped = list(zip(new_solutions, new_evaluations))
        res = sorted(zipped, key=lambda x: x[1])
        self.solutions, self.evaluations = zip(*res)
        return 0

    def selection_Darwin(self):

        """
        this function selects the best solutions, sends them for cross over, and sets new solutions.
        :return:
        """
        new_solutions = [None] * 100
        new_evaluations = [None] * 100
        # get rid of the worse half of the solutions, and evaluations.
        self.solutions = self.solutions[:50]
        self.evaluations = self.evaluations[:50]
        size = 5
        # select the best 10 examples unchanged and for the next selection.
        new_solutions[:size] = self.solutions[:size]
        new_evaluations[:size] = self.evaluations[:size]
        condition = self.solutions[0].conditions
        # create a cross-over from the best solutions, to fill back up to 100 solutions.
        for i in range(size, 100):
            # biased selection of indices of the solutions chosen for cross-over.
            fifty_arr = np.arange(0, 50)
            p = np.arange(0.0298, 0.010, -0.0004)
            a = np.random.choice(a=fifty_arr, p=p, size=2)
            first, second = a[0], a[1]
            # single solution matrix returned.
            solution_matrix = self.crossover(first, second)
            # creating a new solution object.
            solution = Solution(matrix=solution_matrix, condition=condition, size=self.board_size)
            solution.optimiser(self.empty_board, self.board_size)
            solution.evaluate()
            new_solutions[i] = solution
            new_evaluations[i] = solution.evaluation

        if self.board_size > 5:
            for j in range(0, size):
                self.solutions[j].optimiser(self.empty_board, self.board_size)
                self.solutions[j].evaluate()
                new_solutions[j] = self.solutions[j]
                new_evaluations[j] = self.solutions[j].evaluation

        # integrate the new solutions to the game, in an ordered fashion.
        zipped = list(zip(new_solutions, new_evaluations))
        res = sorted(zipped, key=lambda x: x[1])
        self.solutions, self.evaluations = zip(*res)
        return 1

    def selection_Lamarck(self):
        """
        this function selects the best solutions, sends them for cross over, and sets new solutions.
        :return:
        """
        # integrate the new solutions to the game, in an ordered fashion.
        new_solutions = [None] * 100
        new_evaluations = [None] * 100
        size = 5
        for j in range(size, len(self.solutions)):
            self.solutions[j].optimiser(self.empty_board, self.board_size)
            self.solutions[j].evaluate()
        # order the solutions by best grade.
        zipped = list(zip(self.solutions, self.evaluations))
        res = sorted(zipped, key=lambda x: x[1])
        self.solutions, self.evaluations = zip(*res)

        # get rid of the worse half of the solutions, and evaluations.
        self.solutions = self.solutions[:50]
        self.evaluations = self.evaluations[:50]
        # select the best 10 examples unchanged and for the next selection.
        new_solutions[:size] = self.solutions[0:size]
        new_evaluations[:size] = self.evaluations[:size]
        # create a cross-over from the best solutions, to fill back up to 100 solutions.
        for i in range(size, 100):
            # biased selection of indices of the solutions chosen for cross-over.
            fifty_arr = np.arange(0, 50)
            p = np.arange(0.0298, 0.010, -0.0004)
            a = np.random.choice(a=fifty_arr, p=p, size=2)
            first, second = a[0], a[1]
            # single solution matrix returned.
            solution_matrix = self.crossover(first, second)
            condition = self.solutions[0].conditions
            # creating a new solution object.
            solution = Solution(matrix=solution_matrix, condition=condition, size=self.board_size)
            solution.evaluate()
            new_solutions[i] = solution
            new_evaluations[i] = solution.evaluation
        # integrate the new solutions to the game, in an ordered fashion.
        zipped = list(zip(new_solutions, new_evaluations))
        res = sorted(zipped, key=lambda x: x[1])
        self.solutions, self.evaluations = zip(*res)
        return 2

    def crossover(self, solution_index1, solution_index2):
        """
        this function creates a cross-over between two solutions, chooses random lines from each solution.
        :param: solution_index1: - the index of the first solution, within the solutions array.
        :param: solution_index2:- the index of the second solution, within the solutions array.
        :return: a new solution board.
        """
        new_solution = np.zeros((self.board_size, self.board_size))
        for i in range(self.board_size):
            if R.randint(0, 1):
                for j in range(self.board_size):
                    new_solution[i][j] = self.solutions[solution_index1].board[i][j]
            else:
                for j in range(self.board_size):
                    new_solution[i][j] = self.solutions[solution_index2].board[i][j]
        return new_solution

    def mutation(self, p=0.05):
        """
        this function creates a mutation within each line.
        :param: p: the probability of each variable to be switched with another variable in the line.
        :return: none
        """
        for solution in self.solutions:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if raffle(p):
                        if self.empty_board[i][j] == 0:  # able to transform
                            a = R.randint(0, self.board_size - 1)
                            while not self.empty_board[i][a] == 0:
                                a = R.randint(0, self.board_size - 1)
                            temp = solution.board[i][j]
                            solution.board[i][j] = solution.board[i][a]
                            solution.board[i][a] = temp
