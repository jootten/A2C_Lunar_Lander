class Memory:
    def __init__(self, num_steps):
        self.states = np.empty(shape=(num_steps, 8))
        self.actions = np.empty(shape=(num_steps, 2))
        self.rewards = np.zeros(shape=(num_steps, 1))
        self.terminals = []
    
    def store(self, state, action, reward, done, t):
        self.states[t] = state
        self.actions[t] = action
        self.rewards[t] = reward
        self.terminals.append(done)