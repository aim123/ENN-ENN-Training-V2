

class Parser():
    """
    Interface for classes that take an object as input and turn that
    object into some other construct
    """

    def parse(self, input_obj):
        """
        :param input_obj: the string (or other object) to parse

        :return: an object parsed from that string
        """
        raise NotImplementedError
