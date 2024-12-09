from dqn_agent import DQNAgent
from montecarlo import MCAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
# from logs import CustomTensorBoard
from tqdm import tqdm
import argparse
import numpy as np
import os
from PIL import Image
import joblib
from graphviz import Digraph

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--option", type=int, default=0, help="Select mode to play Tetris in: 0 (dqn), 1 (MCTS). Default: 0")
parser.add_argument("-t", "--target_image", type=str, default="./hi.png", help="Select 2x2 binary image to use as target board. Default: ./hi.png")
parser.add_argument("-m", "--reg_model", type=str, default="rf", help="Determine the model used for reward function of MCTS. Options: rf (random forest), linear, knn (k-nearest neighbour), cnn (convolutional neural network). Default: rf")
parser.add_argument("-v", "--verbose", type=int, default=0, help="Output: 0 (no output), or 1 (print MCTS tree to file). Default: 0")
args = parser.parse_args()

# Run dqn with Tetris
def dqn():
    # set image for reachability target
    # set to None for default DQN
    target_image = './hi.png'
    env = Tetris(target_image)
    # episodes = 2000
    episodes = 200
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 50
    log_every = 1
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    #toggle for reachability
    reachability = True

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay, reachability=reachability)
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])
            print(avg_score)
            print(min_score)
            print(max_score)

            # log.log(episode, avg_score=avg_score, min_score=min_score,
                    # max_score=max_score)


def display_tree(node, level=0):
    print('  ' * level + str(node.leaf))
    for child in node.children:
        display_tree(child, level + 1)

# Run MC with Tetris
def MCTS():
    target_image = args.target_image
    env = Tetris(target_image)
    iterations = 50
    max_steps = 25

    #number of playouts per MCTS iteration
    playouts = 100
    
    #unused
    mem_size = 20000
    batch_size = 512
    epochs = 1
    n_neurons = [32, 32]
    episodes = 2000
    epsilon_stop_episode = 1500
    discount = 0.95
    render_every = 50
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    # log_dir = f'logs/tetris-mc={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    
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
    env.reset()
    current_state = env.board

    rootagent = MCAgent(current_state, playouts, args.reg_model)
    agent = rootagent

    n_moves = 0

    curnode = agent.root
    parent = agent.root

    best_sequence = []


    while curnode.t.board != env.target and n_moves < max_steps:
        # print(np.array(curnode.t.board))
        print(f"Move {n_moves}")
        # for i in range(iterations):
        for i in tqdm(range(iterations)):
            # print("select")
            node = agent.select(curnode, parent)
            if node:
                # print("expand")
                # print(node.leaf)
                # print(node.t.has_next_states())
                # print(np.array(node.t.board))
                child, value = agent.expand_and_simulate(node)
                # print("back")
                agent.backpropagate(child, value)
                # display_tree(curnode)
                # print(np.array(curnode.t.board))
        # print(np.array(curnode.t.board))
        # print("end")
        # display_tree(curnode)
        parent = curnode
        bestnode = None
        bestscore = 0
        for child in curnode.children:
            if child.visits > 0:
                current_score = child.uct(iterations) / child.visits
                if current_score > bestscore or bestnode is None:
                    bestnode = child
                    bestscore = current_score
        if bestnode is None:
            break
        best_sequence.append(bestnode)
        curnode = bestnode
        agent = MCAgent(curnode.t.board, playouts, args.reg_model)
        n_moves += 1
        # if curnode is not None:
        #     agent = MCAgent(curnode.t.board, playouts, args.reg_model)
        #     n_moves += 1
        # else:
        #     break
    try:
        os.makedirs(f"mcts_best_sequence_{args.reg_model}")
    except:
        pass

    # display_tree(rootagent.root)

    def save_tree_graph_recursive(root, filename="mcts_tree"):
        """
        Save a tree graph to a file using Graphviz, starting from a root node.

        :param root: The root node of the tree.
        :param filename: The filename for the graph output (without extension).
        """
        dot = Digraph()

        def traverse_and_add(node):
            # Add the current node to the graph
            dot.node(str(id(node)), label=f"{node.wins}/{node.visits}")
            
            # Recursively add child nodes and edges
            for child in node.children:
                dot.edge(str(id(node)), str(id(child)))  # Add edge from parent to child
                traverse_and_add(child)  # Recursive call for child node

        traverse_and_add(root)  # Start the traversal with the root node

        # Save the graph to a file
        dot.render("graphs/"+filename, format="png", cleanup=True)
        print(f"Tree graph saved as {filename}.png")

    if args.verbose != 0:
        save_tree_graph_recursive(rootagent.root)



    for i, node in enumerate(best_sequence):
        array = np.array(node.t.board, dtype=np.uint8)
        image_array = 255 - array * 255
        image = Image.fromarray(image_array)

        file_path = os.path.join(f"mcts_best_sequence_{args.reg_model}", f"{i}.png")
        image.save(file_path)


    # for episode in tqdm(range(episodes)):
    #     current_state = env.reset()
    #     done = False
    #     steps = 0

    #     if render_every and episode % render_every == 0:
    #         render = True
    #     else:
    #         render = False

    #     # Game
    #     while not done and (not max_steps or steps < max_steps):
    #         next_states = env.get_next_states()
    #         best_state = agent.best_state(next_states.values())
            
    #         best_action = None
    #         for action, state in next_states.items():
    #             if state == best_state:
    #                 best_action = action
    #                 break

    #         reward, done = env.play(best_action[0], best_action[1], render=render,
    #                                 render_delay=render_delay)
            
    #         agent.add_to_memory(current_state, next_states[best_action], reward, done)
    #         current_state = next_states[best_action]
    #         steps += 1

    #     scores.append(env.get_game_score())

    #     # Train
    #     if episode % train_every == 0:
    #         agent.train(batch_size=batch_size, epochs=epochs)

    #     # Logs
    #     if log_every and episode and episode % log_every == 0:
    #         avg_score = mean(scores[-log_every:])
    #         min_score = min(scores[-log_every:])
    #         max_score = max(scores[-log_every:])

    #         # log.log(episode, avg_score=avg_score, min_score=min_score,
    #                 # max_score=max_score)

if __name__ == "__main__":
    if args.option == 0:
        print("Running Tetris with DQN")
        dqn()
    elif args.option == 1:
        print("Running Tetris with MCTS")
        MCTS()
