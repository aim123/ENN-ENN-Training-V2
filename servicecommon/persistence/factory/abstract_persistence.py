
import io
import os
import shutil

from servicecommon.persistence.interface.persistence \
    import Persistence


class AbstractPersistence(Persistence):
    """
    Partial implementation of the Persistence interface which
    saves some serialized data for an object via some persistence mechanism.

    Implementations should only need to override the method:
        get_serialization_format()
    """

    def __init__(self, persistence_mechanism):
        """
        Constructor

        :param persistence_mechanism: the PersistenceMechanism to use
                for storage
        """

        super(AbstractPersistence, self).__init__()
        self._mechanism = persistence_mechanism


    def get_serialization_format(self):
        """
        :return: The SerializationFormat instance to be used in persist()
                 and restore()
        """
        raise NotImplementedError


    def persist(self, obj):
        """
        Persists the object passed in.

        :param obj: an object to persist
        """

        serialization = self.get_serialization_format()
        buffer_fileobj = serialization.from_object(obj)
        with buffer_fileobj:

            # Write contents from buffer.
            dest_fileobj = self._mechanism.open_dest_for_write(buffer_fileobj,
                                            serialization)
            if dest_fileobj is not None:
                with dest_fileobj:
                    shutil.copyfileobj(buffer_fileobj, dest_fileobj)

    def restore(self):
        """
        :return: an object from some persisted store
        """

        previous_state = None
        serialization = self.get_serialization_format()
        with io.BytesIO() as buffer_fileobj:
            # Read data into buffer.
            source_fileobj = self._mechanism.open_source_for_read(buffer_fileobj,
                                            serialization)
            dest_obj = None
            if source_fileobj is not None:

                # Check to see if the source_fileobj is a file-like object
                # If so copy into the buffer and set the seek pointer
                # to the start of the buffer
                if hasattr(source_fileobj, 'close'):
                    with source_fileobj:
                        shutil.copyfileobj(source_fileobj, buffer_fileobj)

                    # Set to the beginning of the memory buffer
                    # So next copy can work
                    buffer_fileobj.seek(0, os.SEEK_SET)
                else:
                    # We assume that open_source_for_read() has copied the
                    # data into the buffer_fileobj already
                    pass
                dest_obj = buffer_fileobj

            previous_state = serialization.to_object(dest_obj)

        return previous_state


    def get_file_reference(self):
        """
        :return: A string reference to the file that would be accessed
                by this instance.
        """
        serialization = self.get_serialization_format()
        file_reference = self._mechanism.get_path(serialization)
        return file_reference
