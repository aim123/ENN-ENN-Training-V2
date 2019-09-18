
from __future__ import print_function

import os
import pickle
import time
import uuid

from past.builtins import basestring

from studio.completion_service.completion_service \
    import CompletionService as StudioCompletionService

from experimenthost.completion_service.completion_service_logger \
    import CompletionServiceLogger
from experimenthost.completion_service.completion_service_shutdown_exception \
    import CompletionServiceShutdownException
from experimenthost.completion_service.completion_service_shutdown_task \
    import CompletionServiceShutdownTask
from experimenthost.completion_service.minio_database_shutdown_task \
    import MinioDatabaseShutdownTask
from experimenthost.completion_service.minio_storage_shutdown_task \
    import MinioStorageShutdownTask
from experimenthost.completion_service.rabbit_shutdown_task \
    import RabbitShutdownTask
from experimenthost.util.composite_shutdown_task import CompositeShutdownTask


from framework.util.url_util import is_url


RESUMABLE = False
CLEAN_QUEUE = True
QUEUE_UPSCALING = False
DEBUG_MODE = False


class CompletionServiceWrapper(CompositeShutdownTask):
    """
    An abstraction for supporting multiple completion service backends
    """

    def __init__(self, config, experiment_dir, experiment_id,
                    studio_config_file, unique_id=True):

        super(CompletionServiceWrapper, self).__init__()

        self.config = config

        # A list of all the studio experiment ids this runtime instance
        # has seen.
        self.studio_experiment_ids = []

        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.unique_id = unique_id
        if self.unique_id:
            shutdown_del_queue = True
            self.experiment_id += "-%s" % str(uuid.uuid4())
            print("CompletionService: Using unique experiment id: %s" % \
                  self.experiment_id)
        else:
            shutdown_del_queue = False

        queue_name = self.config.get('queue')
        cloud_name = self.config.get('cloud')
        if self.config.get('local'):
            queue_name = None
            cloud_name = None
        elif queue_name is not None:
            shutdown_del_queue = False
            if cloud_name in ['ec2spot', 'ec2']:
                assert queue_name.startswith("sqs_")
        else:
            # queue = self.experiment_id if cloud_name is None else None
            queue_name = self.experiment_id
            if cloud_name in ['ec2spot', 'ec2']:
                queue_name = "sqs_" + queue_name

        self.compl_serv = StudioCompletionService(self.experiment_id,
                          config=studio_config_file,
                          num_workers=int(self.config.get('num_workers')),
                          resources_needed=self.config.get('resources_needed'),
                          queue=queue_name,
                          cloud=cloud_name,
                          cloud_timeout=self.config.get('timeout'),
                          bid=self.config.get('bid'),
                          ssh_keypair=self.config.get('ssh_keypair'),
                          resumable=RESUMABLE,
                          clean_queue=CLEAN_QUEUE,
                          queue_upscaling=QUEUE_UPSCALING,
                          shutdown_del_queue=shutdown_del_queue,
                          sleep_time=self.config.get('sleep_time')).__enter__()
        self.compl_serv_exited = False

        self.compl_serv_logger = CompletionServiceLogger(self.experiment_dir)

        if DEBUG_MODE or self.config.get('debug'):
            self.debug_dir = os.path.join(self.experiment_dir, "debug")
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)

        self.prepare_for_shutdown()

    def check_file(self, file_path):
        if not isinstance(file_path, basestring):
            raise ValueError("{0} is not a string".format(file_path))

        if not is_url(file_path):
            file_path = os.path.abspath(os.path.expanduser(file_path))
            # print file_path
            try:
                assert os.path.exists(file_path) and os.path.isfile(file_path)
            except:
                print("Error, file not found: %s" % file_path)
                raise
        return file_path

    def submit_task_with_files(self, experiment_id, client_code_file, payload,
                            file_dict, job_id=None):
        if self.compl_serv_exited:
            raise CompletionServiceShutdownException(
                        'completion service has been shut down')

        if DEBUG_MODE or self.config.get('debug'):
            start_time = time.time()
            with open(os.path.join(self.debug_dir,
                                   "submit_%s.payload" % start_time), 'wb') as my_file:
                pickle.dump(payload, my_file)
            with open(os.path.join(self.debug_dir,
                                   "submit_%s.filedict" % start_time), 'w') as my_file:
                my_file.write(str(file_dict))

        assert isinstance(experiment_id, str)
        client_code_file = self.check_file(client_code_file)
        for file_name in file_dict:
            file_dict[file_name] = self.check_file(file_dict[file_name])

        experiment_name = None
        try:
            experiment_name = self.compl_serv.submitTaskWithFiles(
                client_code_file,
                payload,
                file_dict,
                job_id=job_id
            )
        except:
            self.shutdown()
            raise

        if experiment_name is not None:
            self.compl_serv_logger.log_submission(experiment_name)
            self.studio_experiment_ids.append(experiment_name)

        return experiment_name

    def get_results_with_timeout(self, timeout):
        try:
            if self.compl_serv_exited:
                raise CompletionServiceShutdownException(
                        'completion service has been shut down')
            results = self.compl_serv.getResultsWithTimeout(timeout)
        except:
            self.shutdown()
            raise

        if results is None:
            self.compl_serv_logger.log_results(results)
            return results

        experiment_name, return_payload = results
        self.compl_serv_logger.log_results(return_payload, experiment_name)

        if DEBUG_MODE or self.config.get('debug'):
            start_time = time.time()
            with open(os.path.join(self.debug_dir,
                                   "return_%s.payload" % start_time), 'wb') as my_file:
                pickle.dump(return_payload, my_file)

        return return_payload


    def shutdown(self, signum=None, frame=None):
        """
        ShutdownTask interface fulfillment
        """
        if self.compl_serv_exited:
            return

        # Be sure no more work gets submitted
        self.compl_serv_exited = True

        self.do_shutdown(signum, frame)


    def prepare_for_shutdown(self):
        """
        Prepares completion service and studio-ml oriented tasks for
        shutdown.
        """

        # Shut down the completion service
        cs_shutdown = CompletionServiceShutdownTask(self.compl_serv)
        self.append_shutdown_task(cs_shutdown)

        cleanup = self.config.get('cleanup', False)
        if cleanup:
            # Shut down the queue, so no more work goes in or out
            rmq_shutdown = RabbitShutdownTask(self.config)
            self.append_shutdown_task(rmq_shutdown)

            # Remove stuff from minio server, if appropriate.
            minio_storage_shutdown = MinioStorageShutdownTask(self.config,
                                           self.studio_experiment_ids)
            self.append_shutdown_task(minio_storage_shutdown)

            minio_database_shutdown = MinioDatabaseShutdownTask(self.config,
                                           self.studio_experiment_ids)
            self.append_shutdown_task(minio_database_shutdown)

# TODO: Add support for non spot instances to studio CS
