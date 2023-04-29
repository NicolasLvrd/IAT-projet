from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from epsilon_profile import EpsilonProfile

def main():

    game = SpaceInvaders(display=False)
    eps_profile = EpsilonProfile(1., 0.1, 1, 0)
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    controller = QAgent(game, eps_profile, 0.9, 0.1)
    controller.learn(game, 500, 50000)

    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)

if __name__ == '__main__' :
    main()
