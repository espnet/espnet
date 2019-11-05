from typing import Union, Dict, Mapping, List

import torch
import numpy as np
import yaml


class BatchSampler:
    pass

def collate_fn(data: List[Dict[str, np.ndarray]:) -> Dict[str, np.ndarray]:
    return


class Dataset:
    def __init__(self, config: Union[dict, str]):
        """

        config: e.g.

        id_list: /some/where/wav.scp
        data:
            - key: input1
              path: /some/where/wav.scp
              type: sound
            - key: output1
              path: /some/where/utt2tokenid
              type: text_int

        """
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.load(f, Loader=yaml.Loader)

        self.loader_dict = {}
        for data in config['data']:
            key = data['key']
            path = data['path']
            type = data['type']
            loader = Dataset.create_loader(path, type)
            self.loader_dict[key] = loader

        with open(config['id_list']) as f:
            seen_keys = set()
            keys = []
            for line in f:
                # Refer the first column
                uid = line.split()[0]
                if uid in seen_keys:
                    raise RuntimeError(f'{uid} is duplicated')
                keys.append(uid)
                seen_keys.add(uid)
            self.keys = keys

        # Check whether these ids existing
        for key, loader in self.loader_dict.items():
            if not seen_keys.issubset(set(loader)):
                raise RuntimeError(f'The ids mismatching is found between id_list and {key}: {config}')

    @staticmethod
    def create_loader(path: str, loader_type: str) -> Mapping[str, np.ndarray]:
        if loader_type == 'sound':
            return SoundScpReader(path)
        elif loader_type == 'ark-scp':
            return kaldiio.load_scp(path)
        elif loader_type == 'text_int':
            with open(path) as f:
                d = {}
                for line in f:
                    sps = line.split()
                    uid = sps[0]
                    if uid in d:
                        raise RuntimeError(f'{uid} is duplicated')
                    integers = np.array([int(v) for v in sps[1:]], dtype=np.long)
                    d[uid] = integers
        else:
            raise NotImplementedError(f'Not supported: loader_type={loader_type}')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, uid: Union[int, str]) -> Dict[str, np.ndarray]:
        if isinstance(uid, int):
            uid = self.keys[uid]

        data = {}
        for name, loader in self.loader_dict.items():
            value = loader[uid]
            data[name] = value

        return data

