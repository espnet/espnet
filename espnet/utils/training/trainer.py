class Job(object):
    """Job for training"""

    def process(self, data):
        """Data processing method

        :param data: data to be processed in this job
        """
        pass

    def iterator(self):
        """Iterator method

        :return: iterator object for one epoch
        :rtype: iterable
        """
        pass

    def begin_epoch(self):
        """Startup method at the beginning of the epoch"""
        pass

    def end_epoch(self):
        """Cleanup method at the end of the epoch"""
        pass

    def terminate(self):
        """Indicate the end of training

        :return: exit training if True, continue otherwise
        :rtype: bool
        """
        return False


class Trainer(object):
    """Trainer executes registered jobs (e.g., training, validation)

    this class only iterates jobs to avoid a for-loop boilerplate
    """

    def __init__(self, job_dict, n_epochs):
        """
        :param OrderedDict job_dict: jobs to be executed
        :param int n_epochs: the number of epochs
        """
        self.job_dict = job_dict
        self.n_epochs = n_epochs

    def run(self):
        for epoch in range(n_epochs):
            for job_key, job in self.job_dict:
                self.job.begin_epoch()
                for data in self.job.iterator():
                    self.job.process(data)
                self.job.end_epoch()
                if self.job.terminate():
                    return
