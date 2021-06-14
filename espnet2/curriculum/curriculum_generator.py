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

    @classmethod
    def update_weights(cls):
        pass

    @classmethod
    def update_policy(cls, epsilon=0.005):
        

    @classmethod
    def get_reward(cls):
        pass

    @classmethod
    def get_next_task_ind(cls):
        pass
