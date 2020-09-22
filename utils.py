class AverageMeter(object):
    """Helper class to track the running average."""

    def __init__(self, recent=None):
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n

    @property
    def average(self):
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0
