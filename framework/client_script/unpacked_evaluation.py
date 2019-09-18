
import sys
import time
import traceback

from framework.client_script.client_evaluator_factory \
    import ClientEvaluatorFactory
from framework.client_script.worker_response_adapter \
    import WorkerResponseAdapter
from framework.util.logger import Logger
from framework.util.memory_util import check_memory_usage


class UnpackedEvaluation():
    """
    Class that takes over execution of the client evaluation just
    after the code from the library-tarball has been unpacked.

    From this point onwards in terms of execution, we should be able to
    use source files with imports all at the top -- no hidden imports
    for delayed loading are required from here on out.
    """

    def evaluate_with_logs(self, worker_request_dict, eval_start_time, file_dict):
        """
        Main entry point for unpacked evaluation from the Client.

        This is wrapped in a big try/catch by the packed evaluation.
        If something fails here, the outer try will not be able to report a
        proper WorkerResponse dictionary, but will instead just report a
        traceback.
        """

        sys.stdout = Logger("client.out", "a")
        sys.stderr = Logger("client.err", "a")

        worker_response_dict = self.evaluate(worker_request_dict,
                                             eval_start_time, file_dict)
        return worker_response_dict

    def evaluate(self, worker_request_dict, eval_start_time, file_dict):
        """
        Main entry point for unpacked evaluation for outside of the client.

        This is wrapped in a big try/catch by the packed evaluation.
        If something fails here, the outer try will not be able to report a
        proper WorkerResponse dictionary, but will instead just report a
        traceback.
        """

        self.reduce_memory_usage(worker_request_dict)
        worker_response_dict = self.evaluate_candidate(worker_request_dict,
                                              eval_start_time, file_dict)
        return worker_response_dict


    def reduce_memory_usage(self, worker_request_dict):

        config = worker_request_dict.get('config', {})

        empty = {}
        domain_config = config.get('domain_config', empty)
        gpu_mem_frac = domain_config.get('gpu_mem_frac', False)

        if gpu_mem_frac:
            check_memory_usage(gpu_mem_frac)
        else:
            check_memory_usage()


    def evaluate_candidate(self, worker_request_dict, eval_start_time,
                            file_dict):

        identity = worker_request_dict.get('identity', {})
        config = worker_request_dict.get('config', {})

        # Initialize return payload.
        # Old-style, for now.
        queue_wait_time = time.time() - identity.get('submit_timestamp', 0)
        worker_response_dict = {
            'experiment_id': identity.get('experiment_id', None),
            'experiment_timestamp': identity.get('experiment_timestamp', None),
            'gen': identity.get('gen', None),
            'generation_timestamp': identity.get('generation_timestamp', None),
            'id': worker_request_dict.get('id', None),
            'interpretation': worker_request_dict.get('interpretation', None),
            'queue_wait_time': queue_wait_time,
            'submit_timestamp': identity.get('submit_timestamp', None)
        }

        # Train and evaluate.
        try:
            # Get the right client evaluator
            client_evaluator_factory = ClientEvaluatorFactory()

            domain_config = config.get('domain_config', {})
            name = domain_config.get('dummy_run', None)
            client_evaluator = client_evaluator_factory.create_evaluator(name)

            metrics = client_evaluator.evaluate(worker_request_dict, file_dict)

            # Finalize return payload.
            worker_response_dict['fitness'] = metrics['fitness']
            worker_response_dict['metrics'] = metrics
        except Exception:
            worker_response_dict['eval_error'] = traceback.format_exc()

        time_now = time.time()
        worker_response_dict['client_start_time'] = eval_start_time
        worker_response_dict['client_finish_time'] = time_now
        worker_response_dict['client_elapsed_time'] = time_now - eval_start_time


        adapter = WorkerResponseAdapter()
        worker_response_dict = adapter.to_new_style(worker_response_dict)

        # Send return payload back to server through completion service.
        return worker_response_dict
