
from past.builtins import basestring

from servicecommon.parsers.parser import Parser


class BooleanParser(Parser):
    """
    Parser implementation getting a boolean from an object.
    """

    def parse(self, input_obj):
        """
        :param input_obj: the object to parse

        :return: a boolean parsed from that object
        """

        if input_obj is None:
            return False

        if isinstance(input_obj, basestring):
            lower = input_obj.lower()

            true_values = ['true', '1', 'on', 'yes']
            if lower in true_values:
                return True

            false_values = ['false', '0', 'off', 'no']
            if lower in false_values:
                return False

            return False

        return bool(input_obj)
