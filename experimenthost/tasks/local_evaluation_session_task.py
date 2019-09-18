
import os
import time

from future.utils import iteritems

from experimenthost.tasks.get_candidate_session_task \
    import GetCandidateSessionTask
from framework.client_script.unpacked_evaluation import UnpackedEvaluation
from framework.evaluator.data_pathdict import generate_data_pathdict
from framework.evaluator.data_pathdict import download_url_to_file_path


class LocalEvaluationSessionTask(GetCandidateSessionTask):
    """
    SessionTask which locally evaluates a specific candidate *not* via studio.
    Useful for debugging.
    """

    def act_on_candidate(self, candidate, generation):
        """
        :param candidate: The candidate dictionary we want to do something with
        :param generation: the generation number the candidate belongs to
        """
        timestamp = time.time()
        domain_config = self.master_config.get('domain_config')

        candidate_id = candidate.get('id', None)
        interpretation = candidate.get('interpretation', None)

        # Create the file dictionary
        file_dict = {}
        if domain_config.get('send_data', True):
            data_pathdict = generate_data_pathdict(domain_config,
                                                   convert_urls=False)
            file_dict.update(data_pathdict)

        # Normally studio would download the data files as part of its setup.
        # Here, we have to do it ourselves and return a dictionary with the
        # same keys but with references to local files.
        print("Looking for data files")
        local_file_dict = {}
        for key, url in iteritems(file_dict):

            if url is None:
                continue

            if not os.path.exists(key):
                download_url_to_file_path(key, url)
            local_file_dict[key] = key

        # Package the core payload into the payload dictionary
        experiment_config = self.master_config.get('experiment_config')
        worker_request_dict = {
            'config': {
                'domain': experiment_config.get('domain'),
                'domain_config': domain_config,
                'evaluator': experiment_config.get('network_builder'),
                'file_dict': local_file_dict, # Note: Gets transformed by studio
                'resources_needed': experiment_config.get('cs_resources_needed')
            },
            'id': candidate_id,
            'identity': {
                # XXX Different from experiment_id
                'experiment_id': self.experiment_dir,
                'experiment_timestamp': timestamp,
                'generation': generation,
                'generation_timestamp': timestamp,
                'submit_timestamp': timestamp
            },
            'interpretation': interpretation
        }

        print("Locally evaluating with worker request dictionary:")
        print(str(worker_request_dict))

        print("Starting local evaluation")
        unpacked = UnpackedEvaluation()
        worker_response_dict = unpacked.evaluate(worker_request_dict,
                                                 timestamp, local_file_dict)

        print("Locally evaluation complete with worker response dictionary:")
        print(str(worker_response_dict))

        metrics = worker_response_dict.get('metrics', {})
        execution = metrics.get('execution', {})
        eval_error = execution.get('eval_error', None)
        if eval_error is not None:
            print("Got evaluation error:")
            print(str(eval_error))
