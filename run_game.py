import sys
import time

from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.dqn_agent import DQNAgent
from controller.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile


from networks.networks import MLP, CNN


def test_space(env: SpaceInvaders, agent: DQNAgent, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    for _ in range(nepisodes):
        state = env.reset() if (same) else env.reset()
        if display:
            env.render()

        for step in range(max_steps):
            action = agent.select_greedy_action(state)
            next_state, reward, terminal = env.step(action)

            if display:
                time.sleep(speed)
                env.render()

            sum_rewards += reward
            if terminal:
                n_steps = step+1  # number of steps taken
                break
            state = next_state
    return n_steps, sum_rewards


def main(nn, opt):
 
    """ INSTANCIE LE JEU """ 
    env = SpaceInvaders(display = False)
    # env.mode = "nn" 
    model = None
    gamma = 1.

    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    # Hyperparamètres basiques
    n_episodes = 2000
    max_steps = 50
    alpha = 0.001
    eps_profile = EpsilonProfile(1.0, 0.1)
    final_exploration_episode = 500

    # Hyperparamètres de DQN
    batch_size = 32
    replay_memory_size = 1000
    target_update_frequency = 100
    tau = 1.0

    """ INSTANCIE LE RESEAU DE NEURONES """
    # Hyperparamètres 
    if (nn == "mlp"):
        model = MLP(env.get_state(), env.na)   #Max possible de states pour chaque param
    elif (nn == "cnn"):
        model = CNN(env.get_state(), env.na)
    else:
        print("Error : Unknown neural network (" + nn + ").")
    

    print('--- neural network ---')
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('number of parameters:', num_params)
    print(model)

    """  LEARNING PARAMETERS"""
    agent = DQNAgent(model, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode)
    agent.learn(env, n_episodes, max_steps)
    test_space(env, agent, max_steps=15, nepisodes=10, speed=0.1, display=True)

if __name__ == '__main__':
    """ Usage : python main.py [ARGS]
    - First argument (str) : the name of the agent (i.e. 'random', 'vi', 'qlearning', 'dqn')
    - Second argument (int) : the maze hight
    - Third argument (int) : the maze width
    """
    if (len(sys.argv) > 2):
        print("Error : Use is python3 run_game.py [nn]")
    if (len(sys.argv) > 1):
        main(sys.argv[1], [])
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

