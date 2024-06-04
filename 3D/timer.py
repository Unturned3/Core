
import time

class Timer():

    def __init__(self, quiet: bool = False,
                 msg_prefix: str = 'Elapsed time: ',
                 time_fmt: str = '{:.5f} s',
                 conv_func: callable = lambda x: x):
        self.quiet = quiet
        self.msg_prefix = msg_prefix
        self.time_fmt = time_fmt
        self.conv_func = conv_func

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.dt = time.time() - self.start
        if not self.quiet:
            dt = self.conv_func(self.dt)
            print(self.msg_prefix + self.time_fmt.format(dt))
