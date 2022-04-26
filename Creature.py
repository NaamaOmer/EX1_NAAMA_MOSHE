
"""This is a class that represent a creature"""

"""
Authors: Moshe Zeev Hefter 205381379 and Naama Omer 207644014
"""


class Creature:
    """
    this class represents a single creature.
    """

    def __init__(self, index):
        """
        init creature - done in steps.
        :param: index: the ID which is the index in the array.
        """
        self.index = index
        self.state = 0
        self.is_fast = False
        self.x = -1
        self.y = -1
        self.generations = 1

    def set_x_y(self, x, y):
        """
        x and y coordinates for spot
        :param x: x coordinate
        :param y: y coordinate
        :return: none
        """
        self.x = x
        self.y = y

    def set_gen(self, x):
        """
        :param: X: the account of generations where the creature will be sick
        :return:
        """
        self.generations = x

    def get_state(self):
        """
        state is heath situation.
        :return: health state.
        """
        return self.state

    def get_x_y(self):
        """
        gets the coordinates.
        :return: x, y coordinates.
        """
        return self.x, self.y

