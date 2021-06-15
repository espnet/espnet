import numpy as np
from typeguard import check_argument_types

class CurriculumGenerator:
    def __init__(self, 
                curriculum_algo: str = "exp3s", 
                K: int =1, 
                init: str ="zeros",
                hist_size=10000,
                reservoir_size=1000,
                ):

        assert check_argument_types()

        if curriculum_algo=='exp3s':
            self.curriculum_algo = curriculum_algo
        else:
            raise NotImplementedError
        self.K = K 
        self.reward_history = np.array([])
        self.hist_size = hist_size
        self.reservoir_size = reservoir_size
        self.action_hist = []

        if init=='ones':
            self.weights = np.ones((1, K))
        elif init=='zeros':
            self.weights = np.zeros((1, K))
        elif init=='random':
            self.weights = np.random.rand(1, K)
        else:
            raise ValueError(
                f"Initialization type is not supported: {init}"
            )

        self.policy = np.zeros((1, K))

    def update_weights(self):
        pass

    def update_policy(self, k, epsilon=0.05):
        if self.curriculum_algo == 'exp3s':
            tmp1 = np.exp(self.weights[k-1])/np.sum(self.weights)
            pi_k = (1 - epsilon)*tmp1 + epsilon/self.K
            self.policy[k-1] = pi_k

    def get_reward(self, progress_gain):
        '''
        Calculates and scales reward based on previous reward history.
        '''
        if (len(self.reward_history) < self.hist_size) and (len(self.reward_history)!=0):
            q_lo = np.ceil(np.quantile(self.reward_history, 0.2))
            q_hi = np.ceil(np.quantile(self.reward_history, 0.8))
        elif len(self.reward_history):
            q_lo = 0.000000000098
            q_hi = 0.000000000099
        else:
            reservoir = np.random.choice(self.reward_history, size=self.reservoir_size, replace=False)
            q_lo = np.ceil(np.quantile(reservoir, 0.2))
            q_hi = np.ceil(np.quantile(reservoir, 0.8))

        ## Map reward to be in [-1, 1]
        if progress_gain < q_lo:
            reward = -1
        elif reward > q_hi:
            reward = 1
        else:
            reward = (2*(progress_gain - q_lo)/(q_hi-q_lo)) - 1

        if len(self.reward_history) > self.hist_size:
            self.reward_history = np.delete(self.reward_history, 0)
        
        self.reward_history = np.append(self.reward_history, reward)
        return reward
        

    def get_next_task_ind(self):
        return np.argmax(self.policy)
