
from experimenthost.regression.null_fitness_regression \
    import NullFitnessRegression


class FitnessRegressionFactory():
    """
    Factory class for creating FitnessRegression policy.
    """

    def create_fitness_regression(self, config, fitness_objectives,
                                  regression_archive):
        """
        :param config: The experiment config
        :param fitness_objectives: The FitnessObjecitves object
        :param regression_archive: The name of the regression archive file
        :return: a FitnessRegression implementation as dictated by the config
        """

        fitness_regression = None

        if config.get('online_regression'):

            # Don't import unless we have to.
            # This allows us to not expose unready stuff externally
            from experimenthost.regression.online_fitness_regression \
                import OnlineFitnessRegression
            fitness_regression = OnlineFitnessRegression(config,
                                                         fitness_objectives,
                                                         regression_archive)

        if fitness_regression is None:
            fitness_regression = NullFitnessRegression()

        return fitness_regression
