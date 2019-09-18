
from __future__ import print_function

import os
import time

from framework.util.time_util import get_time


class CompletionServiceLogger():
    """
    Helper class designed to log success and errors
    """

    def __init__(self, logdir):
        self.logdir = logdir
        self.logfile = os.path.abspath(os.path.join(self.logdir, "completion_service_log.txt"))
        self.logfile_handle = open(self.logfile, "a+")
        self.logfile_handle.write("[%s][%s] started new CS logging instance\n"
                                  % (get_time(), time.time()))

        self.logfile_handle.flush()
        self.prev_result_time = time.time()
        self.submit_counter = 0
        self.result_counter = 0

    def log_submission(self, experiment_name=None):
        self.logfile_handle.write("[%s][%s][%s][%s]%s: payload submitted\n"
                                  % (get_time(), time.time(),
                                    round(time.time() - self.prev_result_time, 2),
                                     self.submit_counter, experiment_name))

        self.logfile_handle.flush()
        self.prev_result_time = time.time()
        self.submit_counter += 1
        self.result_counter = 0

    def log_results(self, results, experiment_name=None):
        if results is None:
            self.log_failure(experiment_name)
        else:
            self.log_success(experiment_name)

        self.prev_result_time = time.time()
        self.result_counter += 1
        self.submit_counter = 0

    def log_success(self, experiment_name=None):
        experiment_name = '' if experiment_name is None else experiment_name
        self.logfile_handle.write("[%s][%s][%s][%s]%s: result return success\n"
                                  % (get_time(), time.time(),
                                    round(time.time() - self.prev_result_time, 2),
                                     self.result_counter, experiment_name))
        self.logfile_handle.flush()

    def log_failure(self, experiment_name=None):
        experiment_name = '' if experiment_name is None else experiment_name
        self.logfile_handle.write("[%s][%s][%s][%s]%s: results return error\n"
                                  % (get_time(), time.time(),
                                    round(time.time() - self.prev_result_time, 2),
                                     self.result_counter, experiment_name))
        self.logfile_handle.flush()
