
import os

from framework.util.experiment_filer import ExperimentFiler

class GenerationFiler():
    """
    Class to handle creation of file names that go in generation folders.
    """

    def __init__(self, experiment_dir, generation=0):
        """
        Constructor.

        :param experiment_dir: The directory where experiment results go
        :param generation: The generation number of the experiment
        """

        self.experiment_filer = ExperimentFiler(experiment_dir)
        self.generation = generation
        self.prefix = "gen_"

    def get_generation_file(self, filename):
        """
        :param filename: A string filename which does not have any path
                         information associated with it.
        :return: A new string path to the filename in the appropriate
                generation folder, given the constructor arguments
        """

        gen_dir = self.get_generation_dir()
        gen_file = os.path.join(gen_dir, filename)
        return gen_file


    def get_generation_dir(self):
        """
        :return: A string path to the generation folder,
                 given the constructor arguments
        """

        name = self.get_generation_name()
        gen_dir = self.experiment_filer.experiment_file(name)

        return gen_dir


    def get_generation_name(self):
        """
        :return: A cannonical string for the generation.
                 This is used as the primary component for the generation
                 folder, but it can be used for other purposes as well.
        """

        name = "{0}{1:02d}".format(self.prefix, self.generation)
        return name

    def get_generation_from_path(self, path):
        """
        :param path: The path from which we will get generation information.
        :return: the generation number from the given path.
        """

        generation_number = -1

        # Find the component of the path that start with the prefix
        (head, component) = os.path.split(path)
        while component is not None and \
            not component.startswith(self.prefix):
            (head, component) = os.path.split(head)

        if component is None:
            raise ValueError("Could not find prefix {0} in {1}".format(
                                self.prefix, path))

        # Strings are "gen_XX".  Find the number part of that string.
        number_part = None
        if component.startswith(self.prefix):
            number_part = component[len(self.prefix):]

        if number_part is None:
            raise ValueError(
                    "Could not find prefix {0} in path component {1}".format(
                                self.prefix, component))

        try:
            generation_number = int(number_part)
        except:
            raise ValueError("Could not find generation number in {1}".format(
                                component))

        return generation_number
