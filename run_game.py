from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from epsilon_profile import EpsilonProfile

def main():

    game = SpaceInvaders(display=True)
    eps_profile = EpsilonProfile(0.7, 0.05, 1, 0)
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    controller = QAgent(game, eps_profile, 0.3, 0.75)
    controller.learn(game, 50, 20000)

    '''
    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        #sleep(0.0001)
    '''

if __name__ == '__main__' :
    main()
