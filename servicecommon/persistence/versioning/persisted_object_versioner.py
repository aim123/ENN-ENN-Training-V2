

class PersistedObjectVersioner():
    """
    Interface which gives version information about persisted objects.
    """

    def get_current_data_version(self):
        """
        :return: A string providing information as to the current
            data version of the persisted object.

            This value is often persisted with the object in some member
            variable field on the object.
        """
        raise NotImplementedError


    def get_minimum_compatible_data_version(self):
        """
        :return: A string providing information as to the minimum compatible
            data version of the persisted object.

            This is often compared against the version returned by
            get_current_data_version() to see if an object can be loaded
            and understood by a newer version of software.
        """
        raise NotImplementedError
