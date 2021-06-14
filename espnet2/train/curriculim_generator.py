import numpy as np

class CurriculumGenerator:
    def __init__(self, 
                curriculum_algo='exp3s', 
                K=1, 
                init="zeros"
                ):

        if curriculum_algo=='exp3':
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

        def update_weights(self):
            pass

        def get_reward(self):
            pass

        def get_next_task_ind(self):
            pass
