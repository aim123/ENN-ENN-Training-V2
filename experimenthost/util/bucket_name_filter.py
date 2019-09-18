
from past.builtins import basestring


class BucketNameFilter():
    """
    A String filter to create allowable bucket names out of
    a potentially invalid string for bucket names.

    From:
    https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-s3-bucket-naming-requirements.html

    *   The bucket name can be between 3 and 63 characters long, and can contain
        only lower-case characters, numbers, periods, and dashes.
    *   Each label in the bucket name must start with a lowercase letter or
        number.
    *   The bucket name cannot contain underscores, end with a dash, have
        consecutive periods, or use dashes adjacent to periods.
    *   The bucket name cannot be formatted as an IP address (198.51.100.24).
    """

    def filter(self, string):

        # Don't take no mess.
        if string is None or \
            not isinstance(string, basestring):
            return None

        newstring = string
        newstring = self.filter_max_length(newstring)
        newstring = self.filter_lowercase(newstring)
        newstring = self.filter_underscores(newstring)
        newstring = self.filter_invalid_characters(newstring)
        newstring = self.filter_ip_address(newstring)
        newstring = self.filter_period_period(newstring)
        newstring = self.filter_dash_period(newstring)
        newstring = self.filter_period_dash(newstring)
        newstring = self.filter_end_dashes(newstring)
        newstring = self.filter_leading_punctuation(newstring)
        newstring = self.filter_min_length(newstring)
        return newstring


    def filter_min_length(self, string):
        """
        Ensure the string is at least 3 characters long
        by appending x's at the end of it.
        """
        newstring = string
        length = len(newstring)
        min_length = 3
        num_to_add = min_length - length
        while num_to_add > 0:
            newstring = newstring + "x"
            num_to_add = num_to_add - 1

        return newstring

    def filter_max_length(self, string):
        """
        Ensure the string is at most 63 characters long
        by chopping off any extra characters at the end.
        """
        newstring = string
        length = len(newstring)
        max_length = 63
        if length > max_length:
            newstring = newstring[0:max_length]

        return newstring

    def filter_underscores(self, string):
        """
        Replaces any underscores with dashes.
        """
        newstring = string.replace('_', '-')
        return newstring

    def filter_lowercase(self, string):
        """
        Replaces any underscores with dashes.
        """
        newstring = string.lower()
        return newstring

    def filter_period_period(self, string):
        """
        Replaces any occurrence of ..  with --
        """
        newstring = string
        newer_string = newstring.replace("..", "--")
        while newstring != newer_string:
            newer_string = newstring.replace("..", "--")

        return newstring

    def filter_dash_period(self, string):
        """
        Replaces any occurrence of -.  with --
        """
        newstring = string
        newer_string = newstring.replace("-.", "--")
        while newstring != newer_string:
            newer_string = newstring.replace("-.", "--")

        return newstring

    def filter_period_dash(self, string):
        """
        Replaces any occurrence of .-  with --
        """
        newstring = string
        newer_string = newstring.replace(".-", "--")
        while newstring != newer_string:
            newer_string = newstring.replace(".-", "--")

        return newstring

    def filter_ip_address(self, string):
        """
        Replaces periods in strings with 3 or more periods with dashes
        as long as the leading 4 components between the dashes look like
        numbers -- an IP address.
        """
        count = string.count('.')
        newstring = string
        if count < 3:
            # Not enough components to matter
            return newstring

        dot_split = string.split('.')

        # Count the number of components that convert to an integer
        int_count = 0
        for component in dot_split:
            try:
                # Note: _ is pythonic for unused variable
                _ = int(component)
                int_count = int_count + 1
            except ValueError:
                pass

        if int_count >= 4:
            # Replace everything
            newstring = string.replace('.', '-')

        return newstring

    def filter_end_dashes(self, string):
        """
        Removes any end dashes from a string
        """
        newstring = string
        while newstring[-1] == '-':
            newstring = newstring[0:-1]
        return newstring

    def filter_invalid_characters(self, string):
        """
        Replaces any invalid characters with dashes
        """
        valid_chars = "abcdefghijklmnopqrstuvwxyz0123456789-."
        newstring = ""
        for char in string:
            use_char = char
            if char not in valid_chars:
                use_char = '-'
            newstring = newstring + use_char

        return newstring


    def filter_leading_punctuation(self, string):
        """
        Removes any leading instances of - or .
        """
        invalid_start_chars = ".-"
        valid_start = 0
        for char in string:
            if char in invalid_start_chars:
                valid_start = valid_start + 1
            else:
                break
        newstring = string[valid_start:-1] + string[-1]
        return newstring
