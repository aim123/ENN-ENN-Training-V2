

class FileExtensionProvider():
    """
    Interface whose implementations give a file extension.
    """

    def get_file_extension(self):
        """
        :return: A string representing a file extension for the
                serialization method, including the ".".
        """
        raise NotImplementedError
