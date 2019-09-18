

from servicecommon.parsers.boolean_parser import BooleanParser
from servicecommon.parsers.list_parser import ListParser


class BooleanListParser(ListParser):
    """
    A ListParser implementation that parses lists of boolean values
    from a string.
    """

    def __init__(self, delimiter_regex=None):
        """
        Constructor

        :param delimiter_regex: the delimiter_regex used to separate
                string names of values in a parsed string.
                By default the delimiters are commas *and* spaces.
        """
        super(BooleanListParser, self).__init__(delimiter_regex,
                                                BooleanParser())
