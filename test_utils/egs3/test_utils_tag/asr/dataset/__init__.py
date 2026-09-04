class DatasetBuilder:
    def is_source_prepared(self, **kwargs):
        return True

    def prepare_source(self, **kwargs):
        return None

    def is_built(self, **kwargs):
        return True

    def build(self, **kwargs):
        return None


class Dataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
