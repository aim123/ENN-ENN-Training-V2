
import os

from servicecommon.persistence.factory.persistence_factory \
    import PersistenceFactory
from servicecommon.persistence.interface.persistence \
    import Persistence
from servicecommon.persistence.mechanism.persistence_mechanisms \
    import PersistenceMechanisms
from servicecommon.serialization.format.serialization_formats \
    import SerializationFormats


class NetworkVisualizerPersistence(Persistence):
    """
    A class which knows how to persist a visualization of a network
    to a file.

    This class will produce a file of raw bytes for image data.
    The file itself is intended to be human-viewable.
    """

    def __init__(self, image_base, extension, logger=None):
        """
        Constructor.

        :param image_base: the base name for the image file
        :param extension: string representing the image format
        :param logger: A logger to send messaging to
        """

        full_path = "{0}.{1}".format(image_base, extension)
        (image_dir, image_file) = os.path.split(full_path)

        factory = PersistenceFactory(object_type="image",
                                     dictionary_converter=None,
                                     logger=logger)
        self.image_persistence = factory.create_persistence(image_dir,
                        image_file,
                        persistence_mechanism=PersistenceMechanisms.LOCAL,
                        serialization_format=SerializationFormats.RAW_BYTES,
                        must_exist=False)


    def persist(self, obj):
        """
        Persists the object passed in.

        :param obj: an object to persist
                    In this implementation, we expect a dictionary.
        """

        self.image_persistence.persist(obj)


    def restore(self):
        """
        :return: an object from some persisted store.
        """

        image_bytes = self.image_persistence.restore()
        return image_bytes
