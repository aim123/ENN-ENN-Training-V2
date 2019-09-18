
class ShapeDefiner():
    """
    Interface which allows DomainConfigs to define aspects of input and
    output shapes.
    """

    def determine_num_tasks(self, domain_specific_dict):
        """
        Called by the SessionServer.

        :param domain_specific_dict: The assembled domain-specific configuration
            dictionary, with defaults and read-in values already mixed together.
        :return: An integer representing the number of tasks for the network.
            By default a value of "None" is returned, indicating that the
            value is indicated in the network builder configuration
            directly, or is determined by that config's defaults.

        This method translates some domain-specific configuration information
        into generalized specifications for the inputs and outputs of the
        network.  The information created here gets sent to the service
        to be digested by the network constructor.

        While it is conceivable that what gets specified here could be done
        statically in the builder config file, that, however, can be a bit
        cumbersome, and the programmatic interpretation of a few domain config
        parameters tends to have fewer errors.
        """
        raise NotImplementedError


    def determine_input_shapes(self, domain_specific_dict):
        """
        Called by the SessionServer.

        :param domain_specific_dict: The assembled domain-specific configuration
            dictionary, with defaults and read-in values already mixed together.
        :return: A list representing the input shape specifications.
            By default a value of "None" is returned, indicating that the
            value is indicated in the network builder configuration
            directly, or is determined by that config's defaults.

        The input and/or output shapes specification can each take one of
        a few forms:

                1.  A single list that describes a single shape
                    for a single terminal to be used for all tasks.
                    Like this:  [ 128, 128, 1 ]

                2.  A list of lists that describes a different shape
                    for each single-terminal task. Length of the outer
                    list must equal 'num_tasks' in the config for this
                    to be valid. Like this for num_tasks = 3:
                        [ [ 56, 56, 1 ], [ 23, 23, 1 ], [ 4096, 1 ] ]

                    Note that semantically, this *could* describe a
                    common multi-terminal spec for each task when all
                    tasks terminals are the same. Unfortunately this
                    case is indistinguishable from the case described
                    above and any multi-terminal task must use a fully-
                    specified list described in method (3) below,
                    repeats or no.

                3.  A list of lists of lists that describes a different
                    shape for each terminal of a multi-terminal task.
                    Length of the outer list must equal 'num_tasks' in
                    the config for this to be valid.

                    The outer-most list has one entry per task.
                    The middle-most list has one entry per task-terminal,
                    even if any given task only has a single terminal.
                    The inner-most list has one entry per shape dimension
                    for the task-terminal.

                    Like this:  (1st task - 1 terminal,
                                 2nd task - 2 terminals,
                                 3rd task - 3 terminals)
                        [   [ [ 4096, 1 ] ],
                            [ [ 56, 56, 1 ], [ 23, 23, 1 ] ],
                            [ [ 256, 256, 1 ], [ 56, 56, 1 ], [ 18, 1 ] ]
                        ]

        This method translates some domain-specific configuration information
        into generalized specifications for the inputs and outputs of the
        network.  The information created here gets sent to the service
        to be digested by the network constructor.

        While it is conceivable that what gets specified here could be done
        statically in the builder config file, that, however, can be a bit
        cumbersome, and the programmatic interpretation of a few domain config
        parameters tends to have fewer errors.
        """
        raise NotImplementedError


    def determine_output_shapes(self, domain_specific_dict):
        """
        Called by the SessionServer.

        :param domain_specific_dict: The assembled domain-specific configuration
            dictionary, with defaults and read-in values already mixed together.
        :return: A list representing the output shape specifications.
            By default a value of "None" is returned, indicating that the
            value is indicated in the network builder configuration
            directly, or is determined by that config's defaults.

        The input and/or output shapes specification can each take one of
        a few forms:

                1.  A single list that describes a single shape
                    for a single terminal to be used for all tasks.
                    Like this:  [ 128, 128, 1 ]

                2.  A list of lists that describes a different shape
                    for each single-terminal task. Length of the outer
                    list must equal 'num_tasks' in the config for this
                    to be valid. Like this for num_tasks = 3:
                        [ [ 56, 56, 1 ], [ 23, 23, 1 ], [ 4096, 1 ] ]

                    Note that semantically, this *could* describe a
                    common multi-terminal spec for each task when all
                    tasks terminals are the same. Unfortunately this
                    case is indistinguishable from the case described
                    above and any multi-terminal task must use a fully-
                    specified list described in method (3) below,
                    repeats or no.

                3.  A list of lists of lists that describes a different
                    shape for each terminal of a multi-terminal task.
                    Length of the outer list must equal 'num_tasks' in
                    the config for this to be valid.

                    The outer-most list has one entry per task.
                    The middle-most list has one entry per task-terminal,
                    even if any given task only has a single terminal.
                    The inner-most list has one entry per shape dimension
                    for the task-terminal.

                    Like this:  (1st task - 1 terminal,
                                 2nd task - 2 terminals,
                                 3rd task - 3 terminals)
                        [   [ [ 4096, 1 ] ],
                            [ [ 56, 56, 1 ], [ 23, 23, 1 ] ],
                            [ [ 256, 256, 1 ], [ 56, 56, 1 ], [ 18, 1 ] ]
                        ]

        This method translates some domain-specific configuration information
        into generalized specifications for the inputs and outputs of the
        network.  The information created here gets sent to the service
        to be digested by the network constructor.

        While it is conceivable that what gets specified here could be done
        statically in the builder config file, that, however, can be a bit
        cumbersome, and the programmatic interpretation of a few domain config
        parameters tends to have fewer errors.
        """
        raise NotImplementedError
