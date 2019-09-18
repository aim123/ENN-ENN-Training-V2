
import datetime
import getpass
import os


class ExperimentNamer():
    """
    Contains experiment auto-naming policy.
    """


    def name_this_experiment(self, config_file, master_config):
        """
        :param config_file: The reference to the top-level config file.
        :param master_config: The master configuration as read from that
                                config file.
        :return: a string name to use for the experiment of the form
                    "<username>_<domain>_<config>_<datetime>"
        """


        username = os.environ.get("ENN_USER", None)
        if username is None:
            username = getpass.getuser()

        experiment_config = master_config.get('experiment_config')
        domain_name = experiment_config.get('domain')

        config_name = self.get_config_name(config_file)

        date_time = self.get_date_time_string()

        name = "{0}_{1}_{2}_{3}".format(username, domain_name,
                                        config_name, date_time)

        return name


    def get_config_name(self, config_file):
        """
        :param config_file: the config file path
        :return: a name that can be used to identify the config,
                given conventions used around 11/2018
        """

        config_name = None

        dirname, filename = os.path.split(config_file)
        if filename == "experiment":
            # Old-school config. Use the directory name
            # Note: _ is pythonic for unused variable
            _, one_dir_name = os.path.split(dirname)
            config_name = one_dir_name

        else:
            # New-style single-file config
            # Use the base of the file name, minus the file extension
            name_components = filename.split(".")
            config_name = name_components[0]

        return config_name


    def get_date_time_string(self):
        """
        :return: a date-time string for the experiment name
        """
        # Use local time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y%m%d%H%M")
        return date_time
