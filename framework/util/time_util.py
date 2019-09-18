
import datetime
from pytz import timezone


def get_time(space=True):
    """
    Creates a nicely formated timestamp
    """
    if space:
        return datetime.datetime.now(timezone('US/Pacific')) \
            .strftime("%Y-%m-%d %H:%M:%S %Z%z")
    return datetime.datetime.now(timezone('US/Pacific')) \
        .strftime("%Y-%m-%d_%H-%M-%S_%Z%z")
