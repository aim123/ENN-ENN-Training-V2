

class CompletionServiceShutdownException(Exception):
    """
    Completion Service internal exception indicating that
    the service had been shutdown, most likely by a KeyboardException.
    """
