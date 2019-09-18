
import sys


class Logger():
    """
    Simple Logger to log to both a output file and terminal at the same time
    """

    def __init__(self, outfile, write_mode="w"):
        self.terminal = sys.stdout
        self.log_file = open(outfile, write_mode)
        self.log_enabled = True

    def write(self, message):

        if message is None:
            return

        self.terminal.write(message)
        if self.log_enabled:
            self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        if self.log_enabled:
            self.log_file.flush()

    def debug(self, msg, *args, **kwargs):
        """
        Logs a message with level DEBUG on this logger.

        :param msg: The main message to be logged.
        :param args: arguments for the formatting of the string to be logged
        :param kwargs: keyword arguments to be used to pass to the underlying
                    logger implementation.
        """
        self.log(0, msg, *args, **kwargs)


    def info(self, msg, *args, **kwargs):
        """
        Logs a message with level INFO on this logger.

        :param msg: The main message to be logged.
        :param args: arguments for the formatting of the string to be logged
        :param kwargs: keyword arguments to be used to pass to the underlying
                    logger implementation.
        """
        self.log(0, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Logs a message with level WARNING on this logger.

        :param msg: The main message to be logged.
        :param args: arguments for the formatting of the string to be logged
        :param kwargs: keyword arguments to be used to pass to the underlying
                    logger implementation.
        """
        self.log(0, msg, *args, **kwargs)


    def error(self, msg, *args, **kwargs):
        """
        Logs a message with level ERROR on this logger.

        :param msg: The main message to be logged.
        :param args: arguments for the formatting of the string to be logged
        :param kwargs: keyword arguments to be used to pass to the underlying
                    logger implementation.
        """
        self.log(0, msg, *args, **kwargs)


    def critical(self, msg, *args, **kwargs):
        """
        Logs a message with level CRITICAL on this logger.

        :param msg: The main message to be logged.
        :param args: arguments for the formatting of the string to be logged
        :param kwargs: keyword arguments to be used to pass to the underlying
                    logger implementation.
        """
        self.log(0, msg, *args, **kwargs)


    def log(self, lvl, msg, *args, **kwargs):
        """
        Logs a message with level CRITICAL on this logger.

        :param msg: The main message to be logged.
        :param args: arguments for the formatting of the string to be logged
        :param kwargs: keyword arguments to be used to pass to the underlying
                    logger implementation.
        """

        # Note: _ is pythonic for unused variable
        _ = lvl
        _ = kwargs

        if msg is None:
            return

        # By this point we know the logger will accept the log message
        # So there is nothing lost by formatting the string now

        message = str(msg)
        if len(args) > 0:
            message = message.format(*args)

        add_newline = len(message) > 0 and \
                        not message.rstrip(" ").endswith("\n")
        if add_newline:
            message += "\n"

        # Actually do the log
        self.write(message)
