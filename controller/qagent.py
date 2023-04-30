import numpy as np
from epsilon_profile import EpsilonProfile

class QAgent():

    def __init__(self, game, eps_profile: EpsilonProfile, gamma: float, alpha: float):
        # Initialise la fonction de valeur Q
        self.Q = np.zeros([ 41, 13, 3, 2, game.na])

        # Initialise l'environnement
        self.env = game
        self.na = game.na

        # Paramètres de l'algorithme
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

    def learn(self, env, n_episodes, max_steps):     
        # Tableau des scores
        scores_arr = np.zeros(n_episodes)

        # Tableau des Q-values cumulées
        Q_sum_arr = np.zeros(n_episodes)

        # Execute N episodes 
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset()
            # Execute K steps 
            for step in range(max_steps):
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
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)

            # Print
            score = env.get_score()
            print("\r#> Ep. {}/{} | Value={} | Score={}".format(episode+1, n_episodes, self.Q[state][self.select_greedy_action(state)], score), end =" ")

            # Ajoute le score à la liste
            scores_arr[episode] = score

            # Ajoute la somme des Q-values à la liste
            Q_sum_arr[episode] = np.sum(self.Q)

        np.save(f"res/qvalues_epi={n_episodes}_steps={max_steps}_gamma={self.gamma}_alpha={self.alpha}.npy", self.Q)

        np.save(f"res/scores_epi={n_episodes}_steps={max_steps}_gamma={self.gamma}_alpha={self.alpha}.npy", scores_arr)

        np.save(f"res/Q_sum_epi={n_episodes}_steps={max_steps}_gamma={self.gamma}_alpha={self.alpha}.npy", Q_sum_arr)

        print("\nDone !")

    def updateQ(self, state, action, reward, next_state):
        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))

    def select_action(self, state : 'Tuple[int, int]'):
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.na) # random action
        else:
            a = self.select_greedy_action(state) # greedy action
        return a

    def select_greedy_action(self, state : 'Tuple[int, int]'):
        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])