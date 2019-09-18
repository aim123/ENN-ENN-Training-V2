
from servicecommon.persistence.versioning.persisted_object_versioner \
    import PersistedObjectVersioner

class SpecifiedPersistedObjectVersioner(PersistedObjectVersioner):
    """
    PersistedObjectVersioner implmentation where the versions are
    specified externally.
    """

    def __init__(self, current_data_version=None,
                    minimum_compatible_data_version=None):

        self.current_data_version = current_data_version
        self.minimum_compatible_data_version = minimum_compatible_data_version


    def get_current_data_version(self):
        """
        :return: A string providing information as to the current
            data version of the persisted object.

            This value is often persisted with the object in some member
            variable field on the object.
        """
        return self.current_data_version


    def get_minimum_compatible_data_version(self):
        """
        :return: A string providing information as to the minimum compatible
            data version of the persisted object.

            This is often compared against the version returned by
            get_current_data_version() to see if an object can be loaded
            and understood by a newer version of software.
        """
        return self.minimum_compatible_data_version
