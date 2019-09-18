#!/usr/bin/python

import json
import pprint
import os
import sys
import time
import traceback


#
# There are 2 classes defined in this file:
#   CodeTransfer
#   Client
#
# Normally we like one class per file, however this is an exception
# because this class being here solves a chicken-and-egg problem
# with respect to files being unpacked on the studio worker.
#

class CodeTransfer():
    """
    Utilities that aid in packing and unpacking code sent to the worker.
    The use of this class is shared between the the Client class below
    (run on the Studio Worker), and the Experiment Host.
    """

    def __init__(self, domain_name, user_dirs=None):
        """
        Constructor.

        :param domain_name: The specific domain name string which gives
                the hint as to which directory under the domain directory
                gets packaged up.  Can be None, in which case the whole
                domain directory gets packaged up (used only in some odd cases)
        :param user_dirs: an optional list of strings to add to the list
                of directories and files packaged up.  The idea is that
                this could be specified by some domain config if necessary.
                By default this is None.
        """
        self.domain_name = domain_name
        self.user_dirs = user_dirs

    def get_common_code_directories(self):
        """
        The single place where the list of directories + files that
        get packaged up to be sent to the worker.

        :return: The list of directories and files to pack
        """

        directories = ['framework',
                       'servicecommon/__init__.py',
                       'servicecommon/persistence',
                       'servicecommon/serialization',
                       'domain/__init__.py']

        if self.domain_name is not None:
            domain_dir = 'domain/' + self.domain_name
            print("Adding domain path {0}".format(domain_dir))
            directories.append(domain_dir)
        else:
            print("No domain path to add")

        if self.user_dirs is not None:
            if isinstance(self.user_dirs, list):
                directories.extend(self.user_dirs)
            else:
                directories.append(self.user_dirs)

        return directories

    def get_absolute_path_code_directories(self):
        """
        Compile a list of directories (relative to this file)
        that contain code needed by the studio worker to complete evaluation.

        :return: A list of absolute paths to pack
        """

        this_file_dir = os.path.abspath(os.path.dirname(__file__))
        path_to_repo_trunk = "../../"

        directories = self.get_common_code_directories()

        abs_path_directories = []
        for directory in directories:
            from_repo_dir = path_to_repo_trunk + directory
            abs_path_dir = os.path.join(this_file_dir, from_repo_dir)
            exists = os.path.exists(abs_path_dir)
            if not exists:
                raise ValueError("Path {0} does not exist".format(abs_path_dir))
            abs_path_directories.append(abs_path_dir)

        return abs_path_directories


    def pack_code(self, experiment_dir, archive="worker_code_snapshot", verbose=False):
        """
        Archives code in the state that it was run into a tar file and stores
        it in the experiment directory

        :param experiment_dir: the directory where the experiment results go
        :param archive: the name of the file to be used as the archive
        :param verbose: How chatty the process is. Default False.
        """
        directories = self.get_absolute_path_code_directories()

        if isinstance(directories, list):
            directories = " ".join(directories)
        if archive[-7:] != ".tar.gz":
            archive += ".tar.gz"

        tar_verbose = ""
        if verbose:
            tar_verbose = "v"

        os.system("tar" \
                  " --exclude='visualizations'" \
                  " --exclude='datasets'" \
                  " --exclude='*.jpg'" \
                  " --exclude='*.png'" \
                  " --exclude='*.git'" \
                  " --exclude='*.ipynb'" \
                  " -z%scf %s/%s %s"
                  % (tar_verbose, experiment_dir, archive, directories))


    def unpack_code(self, archive):
        """
        Unpacks the code from the named archive.

        :param archive: the name of the file to be used as the archive
        """
        if archive is None:
            raise ValueError("unpack_code archive is None")

        print("Archive is {0}".format(archive))
        exists = os.path.exists(archive)
        if not exists:
            raise ValueError("Archive {0} does not exist".format(archive))

        retval = os.system("tar -xzf %s" % archive)
        if retval != 0:
            raise ValueError("untar of archive {0} returned {1}".format(
                                    archive, retval))

        dirs = self.get_common_code_directories()
        for one_dir in dirs:
            exists = os.path.exists(one_dir)
            if not exists:
                raise ValueError("Directory {0} does not exist".format(one_dir))


    def is_unpacked(self):
        """
        Determines if the code has already been unpacked

        :return: True if all the code was successfully unpacked.
                    False otherwise.
        """
        untarred = True
        dirs = self.get_common_code_directories()
        for directory in dirs:
            if not os.path.exists(directory):
                untarred = False
        return untarred


######################

class Client():
    """
    Class which gets run by the Studio ML Worker, potentially on a remote
    machine.

    This client class takes care of all bootstrapping for code unpacking,
    at which point it hands off control to the (external) UnpackedEvaluation
    class.
    """

    def get_worker_request_value(self, worker_request_dict,
                           new_location=None,
                           new_field=None,
                           old_field=None,
                           default_value=None):
        """
        Get a value from the worker request dict in a new or old
        location.  Allows for compatibility in evolving format.
        """
        if worker_request_dict is None:
            return default_value

        # New-Style
        struct = worker_request_dict.get(new_location, {})
        value = struct.get(new_field, None)

        # Old-style
        if value is None:
            value = worker_request_dict.get(old_field, default_value)

        return value


    def get_domain_name(self, worker_request_dict):
        """
        Get the domain name from the worker_request_dict.
        Allows for compatibility in evolving format.
        """
        return self.get_worker_request_value(worker_request_dict,
                                    new_location='config',
                                    new_field='domain',
                                    old_field='domain_name',
                                    default_value=None)

    def get_dummy_run(self, worker_request_dict):
        """
        Get the dummy_run from the worker_request_dict.
        Allows for compatibility in evolving format.
        """
        return self.get_worker_request_value(worker_request_dict,
                                    new_location='domain_config',
                                    new_field='dummy_run',
                                    old_field='dummy_run',
                                    default_value=0)

    def get_experiment_timestamp(self, worker_request_dict):
        """
        Get the experiment_timestamp from the worker_request_dict.
        Allows for compatibility in evolving format.
        """
        return self.get_worker_request_value(worker_request_dict,
                                    new_location='identity',
                                    new_field='experiment_timestamp',
                                    old_field='experiment_timestamp',
                                    default_value=None)


    def unpack_library(self, file_dict, worker_request_dict=None):
        """
        Unpacks a file dictionary to get latest source code files
        """

        # Get the minimal number of fields from worker_request_dict 'manually'
        # before we have other classes available to us from CodeTransfer.
        domain_name = self.get_domain_name(worker_request_dict)
        dummy_run = self.get_dummy_run(worker_request_dict)
        experiment_timestamp = self.get_experiment_timestamp(
                                        worker_request_dict)

        # Determine if this is to be a real or "dummy" run.
        is_real_run = dummy_run is None or str(dummy_run).lower() == "real"

        # The CodeTransfer object needs to be included in this file as opposed
        # to being imported so that chicken-and-egg problems do not result.
        code_transfer = CodeTransfer(domain_name)
        if is_real_run or \
            not os.path.exists("%s.flag" % experiment_timestamp) or \
            not code_transfer.is_unpacked():

            archive = file_dict.get('lib', None)
            code_transfer.unpack_code(archive)

        # A flag to ensure we extract only once at start
        if not is_real_run:
            os.system("touch %s.flag" % experiment_timestamp)

        framework_path = os.path.join(os.path.dirname(__file__), 'framework')
        if framework_path not in sys.path:
            sys.path.append(framework_path)


    def client_function(self, args, file_dict):
        """
        This is the main entrypoint for evaluating
        Population Service candidates via the completion service.

        It implements the required function for calling submitTaskWIthFile
        from the completionService, given a path to client.py.

        arg is the payload, and file_dict is a dictionary of auxiliary
        files also sent through the completion service.
        """
        try:
            eval_start_time = time.time()
            print("Current working directory: {}".format(os.getcwd()))
            print("Python path: {}".format(os.environ.get('PYTHONPATH', None)))
            print("Environment:")
            pprint.pprint(dict(os.environ))
            print("File Dict:")
            pprint.pprint(file_dict)

            try:
                # Load payload.
                worker_request_dict = args
                # Set up client environment.
                self.unpack_library(file_dict, worker_request_dict)
            except ImportError:
                traceback.print_exc()
                self.unpack_library(file_dict)
                worker_request_dict = args

            # This import has to be done at the site of where they are used
            # because the code for it comes from unpacking the tarball with all the
            # code in it.
            from framework.client_script.unpacked_evaluation \
                import UnpackedEvaluation
            unpacked = UnpackedEvaluation()
            worker_response = unpacked.evaluate_with_logs(worker_request_dict,
                                                eval_start_time,
                                                file_dict)
            return worker_response
        except Exception:
            return traceback.format_exc()


    def main(self, argv):
        """
        Main entry point for parsing command line arguments.
        """

        if len(argv) != 3:
            print("usage: python client.py [args worker-request-file] [file_dict file]")

        with open(argv[1]) as my_file:
            # file of worker_response_dict
            args = my_file.read()
        with open(argv[2]) as my_file:
            # Read dictionary from file reference
            file_dict_json = my_file.read()
            file_dict = json.loads(file_dict_json)
        retval = self.client_function(args, file_dict)
        print(retval)


# This is an interface usedby StudioML, so we cannot change it.
# pylint: disable=invalid-name
def clientFunction(args, file_dict):
    """
    Interface used by StudioML as an entry point from the Completion Service.
    """
    my_client = Client()
    return my_client.client_function(args, file_dict)


if __name__ == "__main__":
    # Interface for running the Client from standalone.
    MAIN_CLIENT = Client()
    MAIN_CLIENT.main(sys.argv)
