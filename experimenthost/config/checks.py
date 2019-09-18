"""
Checks to make sure that there are no invalid configuration settings
Add any checks to ensure that there is no configuration setting which is not legal
"""

ALL_CHECKS = []


def check_regression(server):
    """
    Checks if regression is turned on, persist weights is turned on too
    """
    experiment_config = server.master_config.get('experiment_config')
    domain_config = server.master_config.get('domain_config')
    if experiment_config.get('online_regression'):
        assert domain_config.get('persist_weights', False) is True


ALL_CHECKS.append(check_regression)


def check_batch_size(server):
    """
    Checks to ensure that the train/test dataset subsample amount is at least
    larger than batch size"""
    domain_config = server.master_config.get('domain_config')
    if domain_config.get('subsample', None):
        assert domain_config.get('subsample_amount') >= \
               domain_config.get('batch_size')
    if domain_config.get('test_subsample', None):
        assert domain_config.get('test_subsample_amount') >= \
               domain_config.get('batch_size')


ALL_CHECKS.append(check_batch_size)
