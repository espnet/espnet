from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text

################ AUX FUNCS ################
def read_CR(cr_file):
    '''
    Reads the input comp_ratio.txt
    The input format: uttID, CR
        10001_8844_000000  [0.4956896]
    Params:
        cr_file (str): path to comp_ratio.txt
    '''
    with open(cr_file, 'r') as fo:
        cr_file = fo.read().split(']')[:-1]

    cr_dict = {}

    for i, line in enumerate(cr_file):
        line = line.replace('[', '').split()
        try:
            cr_dict[line[0]] = 1 - float(line[1])
        except IndexError:
            print("line:", line, i, len(line))
    return cr_dict

################ CURRICULUM SAMPLER CLASS ##################
class CurriculumSampler:
    '''
    Returns K iterators with the data sorted according to complexity
    Params: 
        cr_file (str): path to comp_ratio.txt
        K (int): number of tasks (iterators)
    '''
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        cr_file: str = 'comp_ratio.txt',
        K: int=1,
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "descending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        assert check_argument_types()
        assert batch_bins > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending" and sort_in_batch != "random":
            raise ValueError(
                f"sort_in_batch must be ascending, descending or random: {sort_in_batch}"
            )

        self.batch_bins = batch_bins
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last
        self.cr_file = cr_file
        self.K = K

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        ## load compression ratio file

        utt2cr = read_CR(self.cr_file)

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )
        
        #Check if keys match in CR file and shape files
        if set(utt2cr) != set(first_utt2shape):
            raise RuntimeError(
                f"keys are mismatched between {shape_files[0]} != {self.cr_file}"
            )

        # Sort samples in descending order
        keys = dict(sorted(utt2cr.items(), key=lambda k: k[1]))
        
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")
        if padding:
            # If padding case, the feat-dim must be same over whole corpus,
            # therefore the first sample is referred
            keys_dim = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
            feat_dims = [np.prod(d[keys_dim[0]][1:]) for d in utt2shapes]

        else:
            feat_dims = None

        # Decide batch-sizes
        batch_sizes = []
        current_batch_keys = []
        for key in keys:
            current_batch_keys.append(key)
            # shape: (Length, dim1, dim2, ...)
            if padding:
                for d, s in zip(utt2shapes, shape_files):
                    if tuple(d[key][1:]) != tuple(d[keys_dim[0]][1:]):
                        raise RuntimeError(
                            "If padding=True, the "
                            f"feature dimension must be unified: {s}",
                        )
                bins = sum(
                    len(current_batch_keys) * sh[key][0] * d
                    for sh, d in zip(utt2shapes, feat_dims)
                )
            else:
                bins = sum(
                    np.prod(d[k]) for k in current_batch_keys for d in utt2shapes
                )

            if bins > batch_bins and len(current_batch_keys) >= min_batch_size:
                batch_sizes.append(len(current_batch_keys))
                current_batch_keys = []
        else:
            if len(current_batch_keys) != 0 and (
                not self.drop_last or len(batch_sizes) == 0
            ):
                batch_sizes.append(len(current_batch_keys))

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        # If the last batch-size is smaller than minimum batch_size,
        # the samples are redistributed to the other mini-batches
        if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 1] += 1

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

        # Set mini-batch
        self.batch_list = []
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys = []
        for key in keys:
            minibatch_keys.append(key)
            if len(minibatch_keys) == bs:
                if sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending"
                        f" or descending: {sort_in_batch}"
                    )

                self.batch_list.append(tuple(minibatch_keys))
                minibatch_keys = []
                try:
                    bs = next(iter_bs)
                except StopIteration:
                    break

        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )

    def split_tasks(self):
        '''
        Split current batch_list into K tasks.
        '''
        num_batches = len(self.batch_list)
        task_size = num_batches//self.K
        left = 0
        right = 0
        
        tasks = []
        task_i = []
        
        i = 0
    
        while i < num_batches:
            if (len(task_i) < task_size):
                task_i.append(self.batch_list[i])
            else:
                tasks.append(task_i)
                task_i = []
                left = right
                right+=task_size
            i+=1
        tasks.append(task_i)
        del task_i
        return tasks

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-tasks={self.K}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )


    def get_tasks(self):
        '''
        Returns K iterators specified to each task.
        '''
        tasks = self.split_tasks()
        return [iter(t) for t in tasks]



testSampler = CurriculumSampler(
                batch_bins=14000000, 
                shape_files=['/shared/50k_train/mls_english_opus/exp/asr_stats_extracted_train_norm/train/speech_shape',
                        '/shared/50k_train/mls_english_opus/exp/asr_stats_extracted_train_norm/train/text_shape.bpe' ],
                sort_in_batch='descending',
                cr_file='/shared/workspaces/anakuzne/tmp/res/comp_ratio.txt',
                K=2
                )

print("Sampler:", testSampler)

task_iters = testSampler.get_tasks()
print(task_iters)