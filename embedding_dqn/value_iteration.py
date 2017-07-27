import numpy as np

class ValueIteration():
    def __init__(self, gamma, max_VI_iterations, VI_delta):
        self.gamma = gamma
        self.max_VI_iterations = max_VI_iterations
        self.VI_delta = VI_delta

        self.states = []  # stores all possible states
        self.actions_for_state = dict()  # all actions for state
        self.transitions = dict()  # (state_prime, pobability, reward, terminal_probability) pairs indexed by (s, a)

    def run_vi(self, values, states, actions_for_state, transitions, utopia_val):
        self.states = states
        self.actions_for_state = actions_for_state
        self.transitions = transitions

        new_values = dict()
        q_values = dict()

        for i in xrange(self.max_VI_iterations):
            stop = True
            for s in self.states:
                qs = self.calculate_qs(s, values, utopia_val)
                val = np.max(qs.values())
                q_values[s] = qs
                new_values[s] = val

                if s in values and np.abs(val - values[s])/val > self.VI_delta:
                    stop = False
            values = new_values.copy()
            if stop:
                break
        return values, q_values

    def calculate_qs(self, s, values, utopia_val):
        qs = dict()

        for a in self.actions_for_state[s]:
            val = 0
            transitions_sa = self.transitions[(s, a)]

            if len(transitions_sa) == 0:
                val = utopia_val
            else:
                for sp, p, r, t in transitions_sa:
                    if sp in values:
                        val += p * (r + self.gamma * values[sp] * (1 - t))
                    else:
                        val += p * (r + self.gamma * utopia_val)
            qs[a] = val
        return qs

