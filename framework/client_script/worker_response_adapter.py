

class WorkerResponseAdapter():
    """
    Utility class for going between old-style and new-style
    worker response dictionaries.
    """

    def to_new_style(self, old_style):
        """
        :param old_style: the old-style worker response dictionary
        :return: the new-style worker response dictionary
        """

        empty = {}
        if old_style is None:
            old_style = empty

        runinfo = old_style.get('metrics', empty)

        identity = {
            'generation': old_style.get('gen', None),
            'generation_timestamp': old_style.get('generation_timestamp', None),
        }

        execution = {
            'client_elapsed_time': old_style.get('client_elapsed_time', None),
            'eval_error': old_style.get('eval_error', None),
            'return_timestamp': old_style.get('return_timestamp', None),
            'queue_wait_time': old_style.get('queue_wait_time', None),
        }

        novelty = {
            'behavior': runinfo.get('behavior', None)
        }

        regression = {
            'features': runinfo.get('features', None)
        }

        metrics = {
            'execution': execution,
            'fitness': old_style.get('fitness', None),
            'alt_objective': runinfo.get('alt_objective', None),
            'novelty': novelty,
            'regression': regression,
            'total_num_epochs_trained': \
                runinfo.get('total_num_epochs_trained', None),
            'training_time': runinfo.get('training_time', None),
            'weights_l2norm': runinfo.get('weights_l2norm', None)
        }


        # A list of fields already handled above that come from runinfo.
        already_handled_fields = [
            'behavior',
            'features',
            'total_num_epochs_trained',
            'training_time',
            'weights_l2norm'
        ]

        # Take a look at all the measurements coming back from runinfo
        # If it's not something already handled, then it's a domain-specific
        # metric.  Put those metrics on the top-level metrics object.
        for field in list(runinfo.keys()):
            if field not in already_handled_fields:
                value = runinfo.get(field)
                metrics[field] = value

        new_style = {
            'id': old_style.get('id', None),
            'identity': identity,
            'interpretation': old_style.get('interpretation', None),
            'metrics': metrics
        }

        return new_style
