from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
import math
import copy
from tetris import Tetris

# Deep Q Learning Agent + Maximin
#
# This version only provides only value per input,
# that indicates the score expected in that state.
# This is because the algorithm will try to find the
# best final state for the combinations of possible states,
# in constrast to the traditional way of finding the best
# action for a particular state.

class Node:
    def __init__(self, state):
        #List of children, since it will be varying size
        self.state = state
        self.children = []
        self.visits = 0
        self.wins = 0
        self.parent = None

    def uct(self, n):
        return (self.wins / self.visits) + math.sqrt(2) * math.sqrt(np.log(n) / self.visits)

"""
Algorithm: 

repeat n times:
    selection:
        choose node to expand from
    expansion:
        do random move from selected node
    simulation
        random playouts from expanded node
    backpropagation
        calculate the average and backpropagate win/loss

"""


class MCAgent:

    '''Monte Carlo Tree Search Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        n_neurons (list(int)): List with the number of neurons in each inner layer
        activations (list): List with the activations used in each inner layer, as well as the output
        loss (obj): Loss function
        optimizer (obj): Otimizer used
        replay_start_size: Minimum size needed to train
    '''

    def __init__(self, state, playouts):
        self.root = Node(state)
        self.playouts = playouts

    def select(self):
        _, node = self.recursive_select(self.root, self.root)
        return node

    def recursive_select(self, current, parent):
        bestuct = current.uct(parent.visits)
        bestnode = current
        if len(current.children) > 0:
            for child in current.children:
                childuct, childbest = self.recursive_select(child, current)
                if bestuct < childuct:
                    bestuct = childuct
                    bestnode = childbest
        return bestuct, bestnode

    def expand_and_simulate(self, node):
        # get all actions
        next_states = node.state.get_next_states()
        action = random.choice(list(next_states.items()))
        # expand
        child = copy.deepcopy(node)
        child.parent = node
        node.children.append(child)
        child.play(action[0], action[1])
        # simulate
        value = self.do_playouts(child)
        # TODO: determine if win or loss (value = 0: loss, value = 1: win)
        return child, value

    def do_playouts(self, node):
        # create copy to preserve state of node
        curnode = copy.deepcopy(node)
        n = 0
        done = False
        
        #continuously do playouts for each action for set time or game is finished
        while n < self.playouts and not done:
            next_states = curnode.state.get_next_states()
            random_action = random.choice(list(next_states.items()))
            score, done = curnode.state.play(random_action[0], random_action[1])
            n += 1

        return curnode.get_reachability_score()
    
    def backpropagate(self, child, value):
        child.wins += value
        child.visits += 1

        while child.parent:
            child = child.parent
            child.wins += value
            child.visits += 1