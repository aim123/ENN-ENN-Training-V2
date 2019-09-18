
from framework.resolver.resolver import Resolver


class EvaluatorResolver():
    """
    Resolves an Evaluator object for a domain.
    """

    def resolve(self, domain_name, class_name=None,
                evaluator_name=None, extra_packages=None, verbose=False):
        """
        :param domain_name: The name of the domain
        :param class_name: The class name for the domain.
                    If None, it's assumed that this is an old-school reference
                    will be passed in via the evaluator_name arg
        :param evaluator_name: An old-school reference to an evaluator class
        :param extra_packages: A list of packages to search for code.
        :param verbose: Controls how chatty the process is. Default False.
        :return: A python class of the found Domain object.
                This method will raise ValueError with information about what
                was searched if the class was not found.
        """


        use_packages = ["domain." + domain_name]
        if extra_packages is not None:
            if isinstance(extra_packages, list):
                use_packages.extend(extra_packages)
            else:
                use_packages.append(extra_packages)

        resolver = Resolver(use_packages)
        if class_name is not None:
            class_to_resolve = class_name
            module_to_resolve = None
        else:
            module_to_resolve = "{0}_evaluator".format(evaluator_name)
            evaluator_class = evaluator_name[0].upper() + evaluator_name[1:]
            class_to_resolve = "{0}NetworkEvaluator".format(evaluator_class)

        resolver = Resolver(use_packages)
        resolved_class = resolver.resolve_class_in_module(class_to_resolve,
                                                          module_to_resolve,
                                                          verbose=verbose)
        return resolved_class
