

class SerializationFormats():
    """
    Class containing string constants for serialization formats.
    """

    # SerializationFormats
    GZIP = "gzip"
    JSON = "json"
    JSON_GZIP = JSON + "_"+ GZIP
    LEGACY_PICKLE = "legacy_pickle"
    RAW_BYTES = "raw_bytes"
    TEXT = "text"
    YAML = "yaml"

    SERIALIZATION_FORMATS = [LEGACY_PICKLE, JSON, JSON_GZIP, RAW_BYTES, TEXT, YAML]
