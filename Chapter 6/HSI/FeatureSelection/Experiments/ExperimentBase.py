class ExperimentBase:
    def config(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError