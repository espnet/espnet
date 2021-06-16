from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text

################ CURRICULUM SAMPLER CLASS ##################
class CurriculumSampler:
    '''
    Returns K iterators with the data sorted according to complexity
    Params: 
        task_file (str): path to task file
        K (int): number of tasks (iterators)
    '''
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        task_file: str = None,
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
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending, descending: {sort_in_batch}"
            )
        if task_file is None:
            raise ValueError(
                f"task_file required for curriculum learning"
            )
            
        self.batch_bins = batch_bins
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last
        self.task_file = task_file
        self.K = K

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]
        tasks = load_num_sequence_text(task_file, loader_type="text_int")

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )
        
        #Check if keys match in task file and shape files
        if set(tasks) != set(first_utt2shape):
            raise RuntimeError(
                f"keys are mismatched between {shape_files[0]} != {self.task_file}"
            )

        task_keys = [[] for k in range(self.K)]
        for id in tasks:
            task_keys[tasks[id][0]].append(id)
            
        # Sort samples in descending order
        sorted_task_keys = [sorted(t, key=lambda k: first_utt2shape[k][0]) for t in task_keys]

        if len(first_utt2shape) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")
        if padding:
            # If padding case, the feat-dim must be same over whole corpus,
            # therefore the first sample is referred
            feat_dims = [np.prod(d[sorted_task_keys[0][0]][1:]) for d in utt2shapes]
        else:
            feat_dims = None

        self.task_batch_lists = []
        for k in range(self.K):
            keys = sorted_task_keys[k]
            # Decide batch-sizes
            batch_sizes = []
            current_batch_keys = []
            for key in keys:
                current_batch_keys.append(key)
                # shape: (Length, dim1, dim2, ...)
                if padding:
                    for d, s in zip(utt2shapes, shape_files):
                        if tuple(d[key][1:]) != tuple(d[sorted_task_keys[0][0]][1:]):
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
            batch_list = []
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

                    batch_list.append(tuple(minibatch_keys))
                    minibatch_keys = []
                    try:
                        bs = next(iter_bs)
                    except StopIteration:
                        break

            if sort_batch == "ascending":
                pass
            elif sort_batch == "descending":
                batch_list.reverse()
            else:
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )

            self.task_batch_lists.append(batch_list)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-tasks={self.K}, "
            f"batch_bins={self.batch_bins}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )


    def get_tasks(self):
        '''
        Returns K iterators specified for each task.
        '''
        return [iter(batch_list) for batch_list in self.task_batch_lists]
