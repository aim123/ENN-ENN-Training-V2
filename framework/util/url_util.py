

def is_url(instring):
    return instring.startswith("https://") or \
            instring.startswith("http://") or \
            instring.startswith("www.") or \
            instring.startswith("s3://")
