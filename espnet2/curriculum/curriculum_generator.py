import numpy as np
from typeguard import check_argument_types

class CurriculumGenerator:
    def __init__(self, 
                curriculum_algo: str = "exp3s", 
                K: int =1, 
                init: str ="zeros",
                ):

        assert check_argument_types()

        if curriculum_algo=='exp3s':
            self.curriculum_algo = curriculum_algo
        else:
            raise NotImplementedError
        self.K = K 
        self.reward_history = []
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
            tmp1 = np.exp(self.weights[k])/np.sum(self.weights)
            pi_k = (1 - epsilon)*tmp1 + epsilon/self.K
            self.policy[k-1] = pi_k
            print("Policy update:", self.policy)

    def get_reward(self):
        pass

    def get_next_task_ind(self):
        return np.argmax(self.policy)
