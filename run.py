from dqn_agent import DQNAgent
from montecarlo import MCAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
# from logs import CustomTensorBoard
from tqdm import tqdm
        

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



# Run MC with Tetris
def MCTS():
    env = Tetris()
    iterations = 1000
    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 50
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    log_dir = f'logs/tetris-mc={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    #number of playouts per MCTS iteration
    playouts = 100
    
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

    current_state = env.reset()

    agent = MCAgent(current_state, playouts)

    for _ in tqdm(range(iterations)):
        node = MCAgent.select()
        child, value = MCAgent.expand_and_simulate(node)
        MCAgent.backpropagate(child, value)



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
    dqn()
    # MCTS()
