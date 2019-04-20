# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


class Job(object):
    '''Job for training'''

    def run(self, stats):
        '''run the job

        :param Stats stats: stats shared among jobs
        '''
        pass

    def terminate(self, stats):
        '''Indicate the end of training

        :param Stats stats: stats shared among jobs
        :return: exit training if True, continue otherwise
        :rtype: bool
        '''
        return False

    def state_dict(self):
        '''serialize states

        :return: data to save
        :rtype: dict
        '''
        return dict()

    def load_state_dict(self, state_dict):
        '''deserialize states

        :param dict state_dict: data to load
        '''
        for k, v in state_dict.items():
            m = getattr(self, k)
            if hasattr(m, 'load_state_dict'):
                m.load_state_dict()
            else:
                setattr(self, k, v)


class JobRunner(object):
    '''Runs loop for registered jobs (e.g., training, validation)

    :param List[Job] jobs: jobs to be executed
    :param str snapshot_root: root dir to place snapshots
    :param int max_epoch: maximum epoch
    '''

    def __init__(self, jobs, snapshot_root, max_epoch):
        from espnet.utils.training.reporter import Stats
        self.jobs = jobs
        self.snapshot_root = snapshot_root
        self.stats = Stats(max_epoch, snapshot_root)

    def run(self):
        from itertools import count
        import torch
        for epoch in count(self.stats.current_epoch):
            self.stats.current_epoch = epoch
            if epoch >= self.stats.max_epoch:
                return
            for job in self.jobs:
                if job is None:
                    continue

                if job.terminate(self.stats):
                    return

                job.run(self.stats)

                if job.terminate(self.stats):
                    return

            path = '{}/snapshot.ep.{}'.format(self.snapshot_root, epoch)
            torch.save(self.state_dict(), path)

    def state_dict(self):
        '''Serializes states

        :rparam Dict[str, object]: dictionary of states
        '''
        return {'jobs': [j.state_dict() for j in self.jobs], 'stats': self.stats.state_dict()}

    def load_state_dict(self, state_dict):
        '''Deserializes states

        :param Dict[str, object] state_dict: dictionary of states
        '''
        for k, v in state_dict.items():
            if k == 'jobs':
                assert len(v) == len(self.jobs)
                for job, state in zip(v, self.jobs):
                    job.load_state_dict(state)
            else:
                setattr(self, k, v)
