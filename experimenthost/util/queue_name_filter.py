
from past.builtins import basestring


class QueueNameFilter():
    """
    A String filter to create allowable queue names out of
    a potentially invalid string for queue names.

    From RMQ docs:

    *   Queue names can be at most 255 characters.
    *   12/18 We have had anecdotal evidence that queue names with
        dashes in them cause problems. Unclear in this is an RMQ thing
        or a Studio interaction thing.
    """

    def filter(self, string):

        # Don't take no mess.
        if string is None or \
            not isinstance(string, basestring):
            return None

        newstring = string
        newstring = self.filter_max_length(newstring)
        newstring = self.filter_dashes(newstring)
        return newstring


    def filter_max_length(self, string):
        """
        Ensure the string is at most 255 characters long
        by chopping off any extra characters at the end.
        """
        newstring = string
        length = len(newstring)
        max_length = 255
        if length > max_length:
            newstring = newstring[0:max_length]

        return newstring

    def filter_dashes(self, string):
        """
        Replaces any dashes with underscores.
        """
        newstring = string.replace('-', '_')
        return newstring
