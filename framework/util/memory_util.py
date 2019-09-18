
def check_memory_usage(frac=None):
    try:
        import tensorflow as tf
        config = tf.ConfigProto()

        # ConfigProto.gpu_options is a protobufs definition and
        # pylint has trouble seeing into protobuf structures.
        # pylint: disable=no-member
        if frac is None:
            config.gpu_options.allow_growth = True
            print("GPU memory set to dynamic growth")
        else:
            assert 0.0 <= frac <= 1.0
            config.gpu_options.per_process_gpu_memory_fraction = frac
            print("GPU memory fraction set to {}".format(frac))

        from keras import backend as K
        K.set_session(tf.Session(config=config))
    except Exception:
        pass
