

class PersistenceMechanisms():
    """
    Class containing string constants for persistence mechanisms.
    """
    # Persistence Mechanisms
    NULL = "null"           # No persistence
    LOCAL = "local"         # local file
    S3 = "s3"               # AWS S3 storage

    PERSISTENCE_MECHANISMS = [NULL, LOCAL, S3]
