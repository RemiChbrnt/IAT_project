import sys
import time

from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.dqn_agent import DQNAgent
from controller.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile

from networks.networks import MLP, CNN


# Tests phase
def test_space(agent: DQNAgent, max_steps: int, nepisodes: int = 1, speed: float = 0.001, display: bool = True):
    n_steps = max_steps
    env = SpaceInvaders(display=display)
    print("Let's gooooooooooooooooooooooooooooooooooo the tests phase")
    sum_rewards = 0.  # Reward initialise to 0
    for _ in range(nepisodes):
        state = env.reset()

        for step in range(max_steps):
            action = agent.select_greedy_action(state)  # The agent select one of the 4 possible actions
            next_state, reward, terminal = env.step(action)  # Values update

            time.sleep(speed)

            sum_rewards += reward  # Update global reward
            if terminal:
                n_steps = step + 1  # Number of steps taken
                break
            state = next_state
    return n_steps, sum_rewards


def main(nn):
    """ INITIALISE THE GAME """
    env = SpaceInvaders(display=False)  # To display or not the game during learning
    # env.mode = "nn" 
    model = None
    gamma = 0.01

    """ INITIALISE LEARNING PARAMETERS"""
    # Basic hyperparameters
    n_episodes = 100
    max_steps = 2000
    alpha = 1  # Set to 1 when deterministic problem
    eps_profile = EpsilonProfile(1.0, 0.1)  # The influence of the random choice, from 100% to 10%
    final_exploration_episode = 100

    # DQN hyperparameters
    batch_size = 300
    replay_memory_size = 2000
    target_update_frequency = 100
    tau = 1.0

    """ INSTANTIATE THE NEURAL NETWORK """
    # To chose the kind of network
    if nn == "mlp":  # MultiLayer Perceptron
        model = MLP(env.get_nstate(), env.na)  # Number of states max for each parameter
    elif nn == "cnn":  # Convolutional Neural Network
        model = CNN(env.get_nstate(), env.na)
    else:
        print("Error : Unknown neural network (" + nn + ").")

    print('--- neural network ---')
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)  # Number of parameters
    print('number of parameters:', num_params)
    print(model)

    """ LEARNING PARAMETERS """
    agent = DQNAgent(model, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau,
                     final_exploration_episode)  # Creation of the agent with his parameters
    agent.learn(env, n_episodes, max_steps)
    test_space(agent, max_steps=15000, nepisodes=10, speed=0.001, display=True)  # Tests phase


# Main class, beginning of the program
if __name__ == '__main__':
    """ Usage : python main.py [ARG]
    - First argument (str) : the name of the neural network (i.e. 'mlp', 'cnn')
    """

    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Error : Use is python3 run_game.py [nn]")

    # game = SpaceInvaders(display=True)
    # #controller = KeyboardController()
    # controller = RandomAgent(game.na)

    # state = game.reset()
    # while True:
    #     action = controller.select_action(state)
    #     state, reward, is_done = game.step(action)
    #     sleep(0.0001)
