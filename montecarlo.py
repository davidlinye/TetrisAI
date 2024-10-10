import time
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
import math
import copy
import joblib
from reward_regressors.base_regressor import BaseRegressor
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
    def __init__(self, state, model):
        #List of children, since it will be varying size
        self.t = Tetris()
        self.t.board = state
        self.children = []
        self.visits = 0
        self.wins = 0
        self.parent = None
        self.model = model
        self.reg = BaseRegressor(self.import_model(model))
        self.leaf = not self.t.has_next_states()


    def import_model(self, model):
        try:
            return joblib.load(f"models/{model}.pkl")
        except:
            print(f"Model {model} not found, using random forest")
            return joblib.load(f"models/rf.pkl")

    def uct(self, n):
        if self.leaf:
            #do not visit if leaf
            return -1000
        elif self.visits != 0:
            return (self.wins / self.visits) + math.sqrt(2) * math.sqrt(np.log(n) / self.visits)
        else:
            #force exploration when node has not been visited
            return 1000
        
        
    
    def get_reachability_score(self):
        # print(self.t.board)
        # print("predict")
        return self.reg.predict([np.array(self.t.board).flatten()])

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

    def __init__(self, state, playouts, model):
        self.root = Node(state, model)
        self.playouts = playouts
        # self.model = model
        self.final_sequence = []

    def select(self, curnode, parent):
        # print("select")
        # print()
        _, node = self.recursive_select(curnode, parent)
        # print()
        # print("select end")
        return node

    def recursive_select(self, current, parent):
        # print("R")
        bestuct = current.uct(parent.visits)
        bestnode = current
        if len(current.children) > 0:
            for child in current.children:
                childuct, childbest = self.recursive_select(child, current)
                if bestuct < childuct and not child.leaf:
                    # print("best: ")
                    # print(np.array(childbest.t.board))
                    bestuct = childuct
                    bestnode = childbest
        # print(bestnode.leaf)
        # print("R end")
        return bestuct, bestnode
    
    def copy_node(self, node):
        copied_node = Node(node.t.board, node.model)
        return copied_node

    def expand_and_simulate(self, node):
        random_pieces = list(range(7))
        random.shuffle(random_pieces)
        # print(np.array(node.t.board))
        for piece in random_pieces:
            # get all actions
            node.t.current_piece = piece
            next_states = node.t.get_next_states(1)
            if len(next_states) > 0:
                # print(next_states)
                action = random.choice(next_states)
                # print("action")
                # print(action)
                # expand
                child = self.copy_node(node)
                child.parent = node
                node.children.append(child)
                child.t.board = child.t._add_piece_to_board(action[0], action[1])
                child.leaf = not child.t.has_next_states()
                # print("new child")
                # print(np.array(child.t.board))
                # print(child.leaf)
                # simulate
                # print("a")
                value = self.do_playouts(child)

                # less moves = higher value
                if value == 0:
                    value = 1
                converted_value = 1/value
                # TODO: determine if win or loss (value = 0: loss, value = 1: win)
                # print("r")
                return child, converted_value
        # print(np.array(node.t.board))
        raise Exception("Leaf node should not be selected for expansion")

    def do_playouts(self, node):
        # create copy to preserve state of node
        curnode = self.copy_node(node)
        n = 0
        done = False
        
        # print("Expand playouts")
        # continuously do playouts for each action for set time or game is finished
        # only do the actual playouts if the node is not a leaf
        while n < self.playouts and not done and not curnode.leaf:
            # if n % (self.playouts / 10) == 0:
            #     print(n)
            
            random_pieces = list(range(7))
            random.shuffle(random_pieces)
            action_selected = False
            for piece in random_pieces:
                # get all actions
                curnode.t.current_piece = piece
                next_states = curnode.t.get_next_states(1)
                if len(next_states) > 0:
                    action_selected = True
                    random_action = random.choice(next_states)
                    curnode.t.board = curnode.t._add_piece_to_board(random_action[0], random_action[1])
                    curnode.leaf = not curnode.t.has_next_states()
                    n += 1
                    break
            # if no more actions, end of game reached
            if not action_selected:
                n = self.playouts
        # print("b")
        return curnode.get_reachability_score()
    
    def backpropagate(self, child, value):
        child.wins += value
        child.visits += 1

        while child.parent:
            child = child.parent
            child.wins += value
            child.visits += 1

    # def get_optimal_path(self, n_simulations_done):
    #     best_option = None
    #     best_score = 0
    #     for option in self.root.children:
    #         current_score = option.uct(n_simulations_done)
    #         if option.visits > 0:
    #             if current_score > best_score or best_option == None:
    #                 best_option = option
    #                 best_score = current_score
    #     piece, position, rotation = self.decode_option(best_option)
    #     self.final_sequence.append((best_option, piece, position, rotation))

    # def decode_option(self, best_option):
        

