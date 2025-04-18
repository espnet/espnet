
from typing import Union, Tuple, Dict
import numpy as np


class ESPnetEZTransform:
    def __getitem__(self, data):
        uid = data[0]
        data = data[1]
        return (uid, {
            "speech": data["audio"]["array"].astype(np.float32),
            "text": data["text"],
        })
    
    def __call__(self, data):
        return self.__getitem__(data)
