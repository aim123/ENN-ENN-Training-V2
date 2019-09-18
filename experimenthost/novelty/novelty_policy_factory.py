
from experimenthost.novelty.null_novelty_policy import NullNoveltyPolicy


class NoveltyPolicyFactory():
    """
    Factory class for creating NoveltyPolicy
    """

    def create_novelty_policy(self, config, experiment_dir):
        """
        :param config: The experiment config
        :param experiment_dir: The directy where experiment files go
        :return: a NoveltyPolicy implementation as dictated by the config
        """

        novelty_policy = None

        if config.get('novelty_search'):

            # Don't import unless we have to.
            # This allows us to not expose unready stuff externally
            from experimenthost.novelty.original_novelty_policy \
                import OriginalNoveltyPolicy

            novelty_policy = OriginalNoveltyPolicy(config, experiment_dir)

        if novelty_policy is None:
            novelty_policy = NullNoveltyPolicy()

        return novelty_policy
