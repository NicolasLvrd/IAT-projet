from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from epsilon_profile import EpsilonProfile
import numpy as np

def main():

    game = SpaceInvaders(display=True)
    eps_profile = EpsilonProfile(0.3, 0., 0.3, 0)
    # controller = KeyboardController()
    # controller = RandomAgent(game.na)
    controller = QAgent(game, eps_profile, 1.0, 0.75)

    #> ENTRAINEMENT

    #controller.learn(game, 50, 15000)


    #> INFERENCE
    
    # Charge les Q-values
    controller.Q = np.load("res/qvalues_epi=50_steps=15000_gamma=1_alpha=0.75.npy")

    state = game.reset()
    game.display = True
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        #sleep(0.0001)
    

if __name__ == '__main__' :
    main()
