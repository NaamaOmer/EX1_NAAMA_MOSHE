import sys
import Creature
import numpy as np
import pygame as pg
import random as rand

from matplotlib import pyplot as plt

"""
Authors: Moshe Zeev Hefter and Naama Omer
"""

SIZE = 200
global threshold, p1, p2
global SIMULATION, number_of_creatures, sick_start_percent, is_fast_percent, number_of_generations


# public functions
def raffle(p):
    """
    this function chooses 1 (true) given the probability P.
    :param p: - probability
    :return: either 1 or 0 based on chance.
    """
    return np.random.choice((0, 1), p=[1 - p, p])


def menu():
    global SIMULATION, number_of_creatures, sick_start_percent, is_fast_percent, number_of_generations
    global threshold, p1, p2

    is_legal: bool = False
    while not is_legal:
        # printing to the user the menu to console:
        print("~~~~~~~~~~~~~~~~~~~~~ HELLO ~~~~~~~~~~~~~~~~~~~~~~~")
        print("Do you want to use default values? Y/N ")
        print("The default values are: 9200 creatures, 0.08 percent start sick, "
              " 0.1 percent are fast and the creatures live for 13 generations")
        # get the input answer and save it in parameter answer:
        answer = input()
        # checking the answer of user:
        # if YES:
        if answer == 'Y' or answer == 'y' or answer == "yes":
            # Simulation initialized with the default parameters:
            SIMULATION = Simulation(9200, 0.08, 0.1, 13)
            print("Initialising Simulation...")
            # and the rest from params are defined in the class level
            return SIMULATION
        # if NO:
        elif answer == 'N' or answer == 'n' or answer == 'no':
            # FIRST INPUT FROM USER AND CHECK:
            is_legal_values = False
            while not is_legal_values:
                is_legal_values = True
                print("Please enter the following details:\nPlease enter (N) number of creatures. ")
                number_of_creatures = input()
                if not number_of_creatures.isnumeric():
                    is_legal_values = False
                elif 0 > int(number_of_creatures) or int(number_of_creatures) > 40000:
                    is_legal_values = False

            # SECOND INPUT FROM USER AND CHECK:
            is_legal_values = False
            while not is_legal_values:
                is_legal_values = True
                print("Please enter (D) primary percentage of patients.")
                sick_start_percent = input()
                if not sick_start_percent.replace('.', '', 1).isdigit():
                    is_legal_values = False
                elif 100 < float(sick_start_percent) or float(sick_start_percent) < 0:
                    is_legal_values = False

            # THIRD INPUT FROM USER AND CHECK:
            is_legal_values = False
            while not is_legal_values:
                is_legal_values = True
                print("Please enter (R) percentage of creatures that move fast (move fast = moves 10 cells per"
                      " generation).")

                is_fast_percent = input()
                if not is_fast_percent.replace('.', '', 1).isdigit():
                    is_legal_values = False
                elif 100 < float(is_fast_percent) or float(is_fast_percent) < 0:
                    is_legal_values = False

            # FOURTH INPUT FROM USER AND CHECK:
            is_legal_values = False
            while not is_legal_values:
                is_legal_values = True
                print("Please enter (X) number of generations until recovery.")
                number_of_generations = input()
                if not number_of_generations.isnumeric():
                    is_legal_values = False
                elif int(number_of_generations) <= 0:
                    is_legal_values = False

            number_of_creatures = int(number_of_creatures)
            sick_start_percent = float(sick_start_percent)
            is_fast_percent = float(is_fast_percent)
            number_of_generations = int(number_of_generations)
            SIMULATION = Simulation(number_of_creatures, sick_start_percent, is_fast_percent,
                                    number_of_generations)

            print(
                "Would you like to set other values: Y/y for yes and any letter if not.\nDefault threshold value = 0.5,"
                " p_pre_threshold = 0.12, p_post_threshold = 0.10")
            answer = input()
            if answer == 'Y' or answer == 'y' or answer == "yes":
                # FIRST INPUT AND CHECK:
                is_legal_values = False
                while not is_legal_values:
                    is_legal_values: bool = True
                    print(
                        "Please enter (T) the threshold value for change of P as a function of morbidity (ratio 0-1).")
                    threshold = input()
                    if not threshold.replace('.', '', 1).isdigit():
                        is_legal_values = False
                    elif 1 <= float(threshold) or float(threshold) < 0:
                        is_legal_values = False
                SIMULATION.set_threshold(float(threshold))

                # SECOND INPUT AND CHECK:
                is_legal_values = False
                while not is_legal_values:
                    is_legal_values: bool = True
                    print("Please enter (P1) the chance of infection pre threshold (proportion 0-1).")
                    p1 = input()
                    if not p1.replace('.', '', 1).isdigit():
                        is_legal_values = False
                    elif 1 <= float(p1) or float(p1) < 0:
                        is_legal_values = False
                SIMULATION.set_p_pre_threshold(float(p1))
                # THIRD INPUT AND CHECK:
                is_legal_values = False
                while not is_legal_values:
                    is_legal_values: bool = True
                    print("Please enter (P2) the chance of infection post threshold (proportion 0-1).")
                    p2 = input()
                    if not p2.replace('.', '', 1).isdigit():
                        is_legal_values = False
                    elif 1 <= float(p2) or float(p2) < 0:
                        is_legal_values = False
                SIMULATION.set_p_post_threshold(float(p2))
                print("Initialising Simulation...")
                return SIMULATION
            else:
                print("Initialising Simulation...")
                return SIMULATION
        else:
            print("Sorry, didnt understand you... lets try again")
            is_legal = False


"""---------------------------Simulation-class----------------------------------------------------------------"""


class Simulation:
    def __init__(self, size: int, D: float, R: float, X: int):
        """
        initiate all the Simulation parameters.
        most of the first set of parameters are not kept.
        :param: size: the amount of creatures on the board.
        :param: D:  the stating percentage to be sick.
        :param: R: the percentage of fast creatures.
        :param: X:
        ---- self set parameters:
        :param: threshold - the threshold of sick creatures were probability to get sick will go down
        :param: p_pre_threshold - the change to get sick, under the threshold.
        :param: p_post_threshold - the chance to get sick above the threshold.
        :param: sum_sick- the number of starting sick creatures.
        :param: sum_healed - the number of healed creatures.
        :param: sum_iterations - the number of iterations.
        ---- the data structures.
        :param: creatures - aray of creatures.
        :param: infection_percent - array for tracking of the infection percent.
        :param: board - 200X200 board where the creatures reside.
        """
        # un initialized
        self.threshold = 0.01
        self.p_pre_threshold = 0.12
        self.p_post_threshold = 0.1
        # immutable parameters
        self.size = size
        self.init_infect = D
        self.fast_pres = R
        self.generations = X

        self.how_fast = 10

        # the board. 200*200
        # full list of creatures.
        self.sum_sick = 0
        self.sum_healed = 0
        self.sum_iterations = 0

        self.creatures = []
        self.infection_percent = []
        self.board = np.asarray(np.zeros((SIZE, SIZE)) - 1, dtype=np.int32)
        # create basic list.

        num_sick = int(self.size * (D / 100))
        num_fast = int(self.size * (R / 100))
        self.sum_sick = num_sick
        for i in range(self.size):
            creature = Creature.Creature(i)
            self.creatures.append(creature)

        for fast in range(num_fast):
            rand_index_fast = rand.randint(0, self.size - 1)
            # THIS LOOP INSURES THAT WE DON'T ASSIGN THE SAME CREATURE AS BEING FAST TWICE.
            while self.creatures[rand_index_fast].is_fast:
                rand_index_fast = rand.randint(0, self.size - 1)
            self.creatures[rand_index_fast].is_fast = True

        for sick in range(num_sick):
            rand_index_sick = rand.randint(-1, self.size - 1)
            # THIS WHILE LOOP MAKES SURE A CREATURE ISN'T ASSIGNED SICK TWICE, BY RE-RAFFLE TO A DIFFERENT CREATURE.
            while self.creatures[rand_index_sick].state == 1:
                rand_index_sick = rand.randint(0, self.size - 1)
            self.creatures[rand_index_sick].state = 1

        # update starting point of each creature.
        for creature_index in range(size):
            x_rand = rand.randint(0, SIZE - 1)
            y_rand = rand.randint(0, SIZE - 1)
            # THIS LOOP INSURES THAT EACH CREATURE IS ASSIGNED A UNIQUE SLOT.
            while not self.board[x_rand, y_rand] == -1:  # until we get an "empty" cell
                x_rand = rand.randint(0, SIZE - 1)
                y_rand = rand.randint(0, SIZE - 1)
            self.creatures[creature_index].set_x_y(x_rand, y_rand)
            self.board[x_rand][y_rand] = creature_index
            if self.creatures[creature_index].get_state() == 1:
                self.creatures[creature_index].set_gen(self.generations)

    def set_p_pre_threshold(self, p):
        self.p_pre_threshold = p

    def set_p_post_threshold(self, p):
        self.p_post_threshold = p

    def set_threshold(self, t):
        self.threshold = t

    def get_creature(self, index):
        return self.creatures[index]

    def move_creature(self, creature: Creature):
        """
        this function moves a single creature on the board.
        :param creature: - the single creature.
        :return: none.
        """
        didnt_move = True
        # THIS LOOK INSURES THAT THE CREATURE MOVES TO AN EMPTY SPOT, OR STAYS IN PLACE, BY CHECKING IF THE RAFFLED NEW
        # SPOT IS EMPTY BEFORE MOVING
        while didnt_move:
            if creature.is_fast:
                i = rand.randint(-10, 10)
                j = rand.randint(-10, 10)
            else:
                i = rand.randint(-1, 1)
                j = rand.randint(-1, 1)
            new_x = (creature.x + i) % SIZE
            new_y = (creature.y + j) % SIZE
            if self.board[new_x][new_y] == -1:
                self.board[creature.x][creature.y] = -1
                self.board[new_x][new_y] = creature.index
                creature.set_x_y(new_x, new_y)
                didnt_move = False
            elif i == 0 and j == 0:
                didnt_move = False

    def infection_step(self, creature: Creature.Creature, p: float):
        """
        this function checks if the current creature gets infected, and updates the health state.
        :param: creature: the current creature.
        :param: p: the probability to get sick
        :return: none
        """
        if creature.state == 2:  # if current creature has recovered
            return
        if creature.state == 1:
            creature.generations -= 1  # if current creature is sick
            if creature.generations == 0:
                creature.state = 2
                self.sum_sick -= 1
                self.sum_healed += 1
                return
        else:  # creature is healthy now
            x, y = creature.get_x_y()
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbor_creature = self.board[(x + i) % SIZE][(y + j) % SIZE]
                    if neighbor_creature != -1 and neighbor_creature != creature.index:  # if creature has a neighbor
                        if self.creatures[neighbor_creature].state == 1:  # if neighbor is sick
                            if raffle(p):  # if creature just got sick
                                creature.state = 1
                                creature.generations = self.generations
                                self.sum_sick += 1
                                return

    def simulation_iteration(self):
        """
        this function is responsible for a single iteration, which includes updating the infection chance
        and moving and infection each creature.
        :return:  none.
        """
        percent_sick = float(self.sum_sick / self.size)
        self.infection_percent.append(percent_sick * 100)
        if percent_sick >= self.threshold:
            p = self.p_post_threshold
        else:
            p = self.p_pre_threshold
        for creature in self.creatures:
            self.move_creature(creature)
        for creature in self.creatures:
            self.infection_step(creature, p)
        self.sum_iterations += 1

    def visual_creature_output(self):
        """
        this function creates a numpy array that is ready for display.
        :return: the new board.
        """
        display_board = np.asarray(np.zeros((SIZE, SIZE)) - 1, dtype=np.int32)
        for i in range(SIZE):
            for j in range(SIZE):
                if self.board[i][j] != -1:
                    c = self.board[i][j]
                    display_board[i][j] = self.creatures[c].state
        return display_board

    def show_graph(self, title, simulation):
        """
        this function displays the graph at the end of the run.
        :param: title: the title.ma
        :param: simulation: the simulation, with its parameters.
        :return: none
        """
        param_title1 = "N= " + str(simulation.size) + " , D= " + str(simulation.init_infect) + "%  , R=" + str(
            simulation.fast_pres) + ",%  X= " + str(simulation.generations)
        param_title2 = "T= " + str(simulation.threshold) + ",  P1= " + str(simulation.p_pre_threshold) + ",  P2=" + str(
            simulation.p_post_threshold)
        iter_ = np.arange(1, self.sum_iterations + 1)
        infected_np = np.array(self.infection_percent)
        plt.title(title, fontsize=8)
        plt.suptitle('Percentage of infection per generations', fontsize=12)
        # plt.text(param_title1, fontsize=10, y=-0.01)
        plt.plot(iter_, infected_np)
        plt.ylabel('Present infected')
        plt.xlabel('Generation')
        plt.text(float(simulation.sum_iterations) * 0.55, np.amax(simulation.infection_percent), param_title1,
                 fontsize=8)
        plt.text(float(simulation.sum_iterations) * 0.55, np.amax(simulation.infection_percent) * 0.95, param_title2,
                 fontsize=8)
        plt.show()

    def render(self, win):
        """
        gui setup.
        :param: win: gui used.
        :return: none.
        """
        pg.draw.rect(win, 'grey', (0, 0, 700, 700))
        colors = ['dark grey', 'teal', 'crimson', 'navy']
        display_board = self.visual_creature_output()
        size = 600 // SIZE
        for row in range(SIZE):
            for col in range(SIZE):
                pg.draw.rect(win, colors[1 + display_board[row][col]], (50 + col * size, 50 + row * size, size, size))
        #
        font = pg.font.SysFont('freesansbold.ttf', 24)

        text = font.render("Corona-virus Cell Automator", True, 'navy', 'grey')
        text_rect = text.get_rect()
        text_rect.center = (200, 30)
        win.blit(text, text_rect)

        font = pg.font.SysFont('freesansbold.ttf', 18)

        iterations_str = "Iterations: " + str(self.sum_iterations)
        text_iterations = font.render(iterations_str, True, 'black', 'grey')
        text_rect1 = text_iterations.get_rect()
        text_rect1.center = (600, 30)
        win.blit(text_iterations, text_rect1)

        num_sick_str = "Sick: " + str(self.sum_sick)
        text_sick = font.render(num_sick_str, True, 'crimson', 'grey')
        text_rect2 = text_sick.get_rect()
        text_rect2.center = (150, 680)
        win.blit(text_sick, text_rect2)

        num_healthy_str = "healthy: " + str(self.size - self.sum_sick)
        text_healthy = font.render(num_healthy_str, True, 'teal', 'grey')
        text_rect3 = text_healthy.get_rect()
        text_rect3.center = (350, 680)
        win.blit(text_healthy, text_rect3)

        num_recovered_str = "recovered: " + str(self.sum_healed)
        text_recovered = font.render(num_recovered_str, True, 'navy', 'grey')
        text_rect4 = text_recovered.get_rect()
        text_rect4.center = (550, 680)
        win.blit(text_recovered, text_rect4)
        pg.display.update()


"""-------------------------------main-------------------------------------------------------------------------------"""
if __name__ == '__main__':
    newSim = menu()
    print('\n')
    pg.init()
    win_ = pg.display.set_mode((700, 700))
    pg.display.set_caption("Cell Automata")
    icon = pg.image.load('icon.png')
    pg.display.set_icon(icon)

    while newSim.sum_sick != 0:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        newSim.simulation_iteration()
        newSim.render(win_)

    sumZ = ((newSim.sum_healed + newSim.sum_sick) / newSim.size) * 100
    strZ = "{:.2f}".format(sumZ)
    title_ = "Final infected: " + str(strZ) + "%"
    newSim.show_graph(title_, newSim)
