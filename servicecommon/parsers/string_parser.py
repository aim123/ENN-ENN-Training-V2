
from servicecommon.parsers.parser import Parser


class StringParser(Parser):
    """
    Parser implementation getting a string from an object.
    """

    def parse(self, input_obj):
        """
        :param input_obj: the object to parse

        :return: a string parsed from that input object
        """
        return str(input_obj)
