
class ServiceRouting(object):

    # This entrypoint version is a bit that is added to the host address
    # so that the client has the ability to route to the correct service
    # itself without dealing with the fan-out of all the config files already
    # in use.
    #
    # XXX Note we look at this as a bit of a hack until we resolve
    #     some technical debt inside the service.
    #
    #   Also note that this file is explicitly not included by the
    #   minimal manifest file so as to prevent accidental upgrades
    #   that are not yet ready from prime-time
    CLIENT_ROUTING_VERSION="v10"
    
