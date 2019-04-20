class Job(object):
    """Job for training"""

    def process(self, data):
        """Data processing method

        :param data: data to be processed in this job
        """
        pass

    def __iter__(self):
        """Iterator method

        :return: iterator object for one epoch
        :rtype: iterable
        """
        pass

    def __enter__(self):
        """Startup method at the beginning of the epoch"""
        pass

    def __exit__(self):
        """Cleanup method at the end of the epoch"""
        pass

    def terminate(self):
        """Indicate the end of training

        :return: exit training if True, continue otherwise
        :rtype: bool
        """
        return False

    def state_dict(self):
        """serialize states

        :return: data to save
        :rtype: dict
        """
        ret = dict()
        for attr in dir(self):
            m = getattr(self, attr)
            if hasattr(m, "state_dict"):
                ret[attr] = m.state_dict()
            # NOTE: maybe need to go deeper attr?
        return ret

    def load_state_dict(self, state_dict):
        """deserialize states
        :param dict state_dict: data to load
        """
        for k, v in state_dict.items():
            m = getattr(self, k)
            if hasattr(m, "load_state_dict"):
                m.load_state_dict()
            else:
                setattr(self, k, v)


class Trainer(object):
    """Trainer executes registered jobs(e.g., training, validation)

    this class only iterates jobs to avoid a for-loop boilerplate
    """

    def __init__(self, jobs, n_epoch, snapshot_root):
        """
        : param List[Job] jobs: jobs to be executed
        : param int n_epoch: the numeber of epochs
        : param str snapshot_root: root dir to place snapshots
        """
        self.jobs = jobs
        self.n_epoch = n_epoch
        self.last_epoch = 0
        self.snapshot_root = snapshot_root

    def run(self):
        import torch
        for epoch in range(self.last_epoch, self.n_epoch):
            for job in self.jobs:
                with job:
                    for data in iter(job):
                        job.process(data)
                if job.terminate():
                    return

            path = "{}/snapshot.ep.{}".format(self.snapshot_root, epoch)
            torch.save(self.state_dict(), path)

    def state_dict(self):
        return {
            "jobs": [j.state_dict() for j in self.jobs],
            "n_epoch": self.n_epoch,
            "last_epoch": self.last_epoch
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k == "jobs":
                assert len(v) == len(self.jobs)
                for i in enumerate(v):
                    self.jobs[i].load_state_dict(jv)
            else:
                setattr(self, k, v)
