import numpy as np
from typeguard import check_argument_types
from abc import ABC
from abc import abstractmethod

class AbsCurriculumGenerator(ABC):
    @abstractmethod
    def update_policy(self, k, epsilon=0.05):
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, progress_gain):
        raise NotImplementedError
        
    @abstractmethod
    def get_next_task_ind(self):
        raise NotImplementedError


class EXP3SCurriculumGenerator(AbsCurriculumGenerator):
    def __init__(self, 
                K: int =1, 
                init: str ="zeros",
                hist_size=10000,
                ):

        assert check_argument_types()

        self.K = K 
        self.reward_history = np.array([])
        self.hist_size = hist_size
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

    def get_next_task_ind(self):
        return np.argmax(self.policy)

    def update_policy(self, k, epsilon=0.05):
        tmp1 = np.exp(self.weights[k-1])/np.sum(np.exp(self.weights))
        pi_k = (1 - epsilon)*tmp1 + epsilon/self.K
        self.policy[k-1] = pi_k

    def get_reward(self, progress_gain, batch_lens):
        '''
        Calculates and scales reward based on previous reward history.
        '''
        progress_gain = progress_gain/np.sum(batch_lens)

        if len(self.reward_history)==0:
            q_lo = 0.000000000098
            q_hi = 0.000000000099
        else:
            q_lo = np.ceil(np.quantile(self.reward_history, 0.2))
            q_hi = np.ceil(np.quantile(self.reward_history, 0.8))
        

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

    def update_weights(self):
        pass