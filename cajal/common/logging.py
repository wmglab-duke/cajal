import logging
import time

from cajal.mpi import Backend as MPI


TIME_STACK = []


class OneTimeLogger(logging.Logger):
    """
    A custom subclass of 'logging.Logger' that keeps record of a set of
    seen messages to implement {debug,info,etc.}_once() methods.
    """

    def __init__(self, name):
        super().__init__(name)
        self._seen = set()

    def debug_once(self, msg, *args, **kwargs):
        if msg not in self._seen:
            self.debug(msg, *args, **kwargs)
            self._seen.add(msg)

    def info_once(self, msg, *args, **kwargs):
        if msg not in self._seen:
            self.info(msg, *args, **kwargs)
            self._seen.add(msg)

    def warning_once(self, msg, *args, **kwargs):
        if msg not in self._seen:
            self.warning(msg, *args, **kwargs)
            self._seen.add(msg)

    def error_once(self, msg, *args, **kwargs):
        if msg not in self._seen:
            self.error(msg, *args, **kwargs)
            self._seen.add(msg)

    def critical_once(self, msg, *args, **kwargs):
        if msg not in self._seen:
            self.critical(msg, *args, **kwargs)
            self._seen.add(msg)


logging.setLoggerClass(OneTimeLogger)


class DisableLogger:
    def __init__(self, log):
        self.logger = log

    def __enter__(self):
        self.logger.disabled = True

    def __exit__(self, a, b, c):
        self.logger.disabled = False


# -- define base logger and formatting options --
logger = logging.getLogger(f"rank[{MPI.RANK}]")
logFormatter = logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)


# -- timing --
def tic(message=None, log=True):
    TIME_STACK.append(time.time())
    if message and log:
        logger.info(str(message))


def toc(message=None, log=True):
    try:
        t = time.time() - TIME_STACK.pop()
        output = f"Elapsed: {t:.3f}s"
        if message:
            output = f"{message}:: {output}"
        if log:
            logger.info(output)
        return t
    except IndexError:
        logger.error("You have to tic() before you toc()")
