
import os

import boto3
import requests

from awscli.customizations.s3.utils import split_s3_bucket_key
from tqdm import tqdm

from framework.util.url_util import is_url


def generate_data_pathdict(domain_config, convert_urls):

    data_basedir = domain_config.get('data_basedir', None)
    local_data_cachedir = domain_config.get('local_data_cachedir', None)
    filename_dict = domain_config.get('filename_dict', None)

    if is_url(data_basedir) and convert_urls:
        return generate_data_pathdict_s3(data_basedir,
                                         local_data_cachedir,
                                         filename_dict)
    return generate_data_pathdict_local(data_basedir, filename_dict)


def generate_data_pathdict_local(local_data_basedir, filename_dict):
    """
    Generates local pathdict
    """
    if not is_url(local_data_basedir):
        local_data_basedir = os.path.abspath(os.path.expanduser(local_data_basedir))
    data_pathdict = {}
    for pathdict_name, filename in list(filename_dict.items()):
        data_pathdict[pathdict_name] = os.path.join(local_data_basedir, filename)
    return data_pathdict


def generate_data_pathdict_s3(s3_data_basedir, local_data_cachedir,
                              filename_dict):
    """
    Resolves s3 datafile paths into local paths
    """
    # Note: _ is pythonic for unused variable
    _, data_dir_name = s3_data_basedir.rsplit("/", 1)

    local_data_basedir = os.path.join(os.path.abspath(
        os.path.expanduser(local_data_cachedir)), data_dir_name)
    if not os.path.exists(local_data_basedir):
        os.makedirs(local_data_basedir)

    data_pathdict = {}
    for pathdict_name, filename in list(filename_dict.items()):

        file_url = os.path.join(s3_data_basedir, filename)
        file_path = os.path.join(local_data_basedir, filename)

        download_url_to_file_path(file_path, file_url)

        data_pathdict[pathdict_name] = file_path

    return data_pathdict


def download_url_to_file_path(file_path, file_url):

    print("Downloading file {0} from url {1}".format(file_path, file_url))
    with open(file_path, "wb") as handle:
        if file_url.startswith("s3:"):
            s3_bucket, s3_key = split_s3_bucket_key(file_url)
            s3_client = boto3.client('s3')
            s3_client.download_fileobj(Fileobj=handle,
                                Bucket=s3_bucket,
                                Key=s3_key)
        else:
            response = requests.get(file_url)
            for data in tqdm(response.iter_content()):
                handle.write(data)


def get_data_dict_filename(data_dict, key):
    """
    Call this for opening data files inside the Worker doing evaluation.
    This helps standardize some error checking and reporting
    on data dict entries.

    :param data_dict: The data file dictionary sent to the Worker
        by the Experiment Host.  Individual data file references are
        set up as keys in DomainConfig.generate_filename_dict().
    :param key: The key by which to reference the data file in
        the data_dict. These are domain-specific.
    :return: The value of the key in the data_dict
    """

    if data_dict is None or \
        not isinstance(data_dict, dict):
        raise ValueError("data_dict is not a dictionary.")

    if key is None:
        raise ValueError("No key into data_dict")

    value = data_dict.get(key, None)
    if value is None:
        message = "Could not find key {0} in data_dict:\n{1}\n".format(
                        key, data_dict)
        message += """

    If this is a truly valid key, this could mean that the data did not arrive
    to the Worker intact. Some things to try:
        *   Double check that your data actually exists in the location you
            think it is.  This is defined by the config key
            'domain_config.data_basedir' and is often augmented by the
            environment variable ${DOMAIN_DATA_ENDPOINT}
        *   If using AWS S3 or Minio, double check your credentials
            in the environment variables ${AWS_ACCESS_KEY_ID},
            ${AWS_SECRET_ACCESS_KEY}, and ${AWS_DEFAULT_REGION}
            match the credentials that are needed to access the data.
        *   Double-check that the amount of ram or hdd specified in the config
            key 'experiment_config.completion_service.resources_needed'
            is sized correctly for your data.
    """
        raise ValueError(message)

    return value


def open_data_dict_file(data_dict, key, binary=True):
    """
    Call this for opening data files inside the Worker doing evaluation.
    This helps standardize some error checking and reporting
    on data dict entries.

    :param data_dict: The data file dictionary sent to the Worker
        by the Experiment Host.  Individual data file references are
        set up as keys in DomainConfig.generate_filename_dict().
    :param key: The key by which to reference the data file in
        the data_dict. These are domain-specific.
    :param binary: When True (the default), opens the data file
        in binary mode.
    :return: The a file-like object for the key in the data_dict
    """

    value = get_data_dict_filename(data_dict, key)

    mode = "r"
    if binary:
        mode = "rb"

    return open(value, mode)
