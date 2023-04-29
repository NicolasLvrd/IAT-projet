import numpy as np
from epsilon_profile import EpsilonProfile
from time import sleep

class QAgent():

    def __init__(self, game, eps_profile: EpsilonProfile, gamma: float, alpha: float):
        # Initialise la fonction de valeur Q
        #self.Q = np.zeros([ round(800/game.reducing_factor + 1), round(800/game.reducing_factor + 1), round(600/game.reducing_factor + 1), game.na])
        #self.Q = np.zeros([ 81, 61, 81, 2, game.na])
        self.Q = np.zeros([ 11, 8, 11, 8, 2, game.na])

        self.game = game
        self.na = game.na

        # Paramètres de l'algorithme
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        # Visualisation des données (vous n'avez pas besoin de comprendre cette partie)
        # self.qvalues = pd.DataFrame(data={'episode': [], 'value': []})
        # self.values = pd.DataFrame(data={'nx': [maze.nx], 'ny': [maze.ny]})

    def learn(self, env, n_episodes, max_steps):       
        # Execute N episodes 
        for episode in range(n_episodes):

            # Reinitialise l'environnement
            state = env.reset()

            for step in range(max_steps):
                #print("\r#> Ep. {}/{} | Step {}/{} |  | Epsilon={}".format(episode, n_episodes, step, max_steps, self.epsilon), end =" ")
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)
                
                if terminal:
                    break

                state = next_state

            # Mets à jour la valeur du epsilon
            # self.epsilon = max(self.epsilon - self.eps_profile.dec_step, self.eps_profile.final)

            # Sauvegarde es données d'apprentissage
            score = env.get_score()
            state = env.reset()
            print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes, self.Q[state][self.select_greedy_action(state)]), end =" ")
            #print("\r#> Ep. {}/{} | Value={} | score={}".format(episode, n_episodes, self.Q[state][self.select_greedy_action(state)], score), end =" ")
            #p.save(f"res/qvalues_epi={n_episodes}_steps={max_steps}_gamma={self.gamma}_alpha={self.alpha}.npy", self.Q)

    def updateQ(self, state, action, reward, next_state):
        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))

    def select_action(self, state : 'Tuple[int, int, int ,int, int]'):
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.na)      # random action
        else:
            a = self.select_greedy_action(state)
        #print("Action : ", a)
        return a

    def select_greedy_action(self, state : 'Tuple[int, int, int ,int, int]'):
        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])

    '''
    def save_log(self, env, episode):
        state = env.reset_using_existing_maze()
        # Construit la fonction de valeur d'état associée à Q
        V = np.zeros((int(self.maze.ny), int(self.maze.nx)))
        for state in self.maze.getStates():
            val = self.Q[state][self.select_action(state)]
            V[state] = val

        self.qvalues = self.qvalues.append({'episode': episode, 'value': self.Q[state][self.select_greedy_action(state)]}, ignore_index=True)
        self.values = self.values.append({'episode': episode, 'value': np.reshape(V,(1, self.maze.ny*self.maze.nx))[0]},ignore_index=True)
    '''