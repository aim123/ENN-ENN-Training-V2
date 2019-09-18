
import logging
import sys

from past.builtins import basestring

import boto3
import studio.logs as logs

from experimenthost.util.shutdown_task import ShutdownTask
from experimenthost.util.bucket_name_filter import BucketNameFilter


class MinioShutdownTask(ShutdownTask):
    """
    Abstract Shutdown Task for removing files from minio server upon shutdown.
    """

    def __init__(self, config, studio_experiment_ids,
                 config_key=None, folders=None):
        """
        :param config: The completion_service config dictionary
        :param studio_experiment_ids: A list of known studio experiment ids.
        :param config_key: The config key with the target persistent store
                            from which we would like to delete stuff
        :param folders: The folders within that persistent store we would
                            like to delete things from.
        """

        self.config = config
        self.studio_experiment_ids = studio_experiment_ids
        self.config_key = config_key
        self.folders = folders

        self.persistence = self.find_persistence(config, config_key)

        self.logger = logs.getLogger('MinioShutdown')
        self.logger.setLevel(logging.INFO)


    def shutdown(self, signum=None, frame=None):
        """
        ShutdownTask interface fulfillment
        """
        if self.persistence is None:
            self.logger.info("Not doing minio cleanup for target %s",
                                str(self.config_key))
            return

        self.logger.info("Initiating minio cleanup for target %s",
                                str(self.config_key))
        self.cleanup()


    def find_studio_ml_config(self, config):
        """
        :param config: The completion_service config dictionary
        :return: the studio ml config dictionary (if any)
        """

        empty = {}
        studio_config = config.get('studio_ml_config', empty)
        if studio_config is None or \
            isinstance(studio_config, basestring):
            # XXX We could go through studio config yaml files, but not yet.
            studio_config = empty

        return studio_config


    def find_persistence(self, config, key):
        """
        :param config: The completion_service config dictionary
        :param key: the key for the persistence dictionary we are looking for
        :return: the dictionary corresponding to the key (if any)
        """

        studio_config = self.find_studio_ml_config(config)

        empty = {}
        persistence_dict = studio_config.get(key, empty)
        if persistence_dict is None:
            persistence_dict = empty

        # For now, only clean up when type is s3
        persistence_type = persistence_dict.get('type', None)
        if persistence_type is None or \
            persistence_type != 's3':
            persistence_dict = empty

        if len(list(persistence_dict.keys())) == 0:
            persistence_dict = None

        return persistence_dict


    def cleanup(self):
        """
        Do the work of cleaning up the persisted store
        """
        if self.persistence is None:
            return

        if self.folders is None:
            return

        endpoint_url = self.persistence.get('endpoint', None)
        s3_client = boto3.resource('s3', endpoint_url=endpoint_url)

        orig_bucket_name = self.persistence.get('bucket', None)
        bucket_name_filter = BucketNameFilter()
        bucket_name = bucket_name_filter.filter(orig_bucket_name)
        bucket = s3_client.Bucket(bucket_name)

        # Note: As of 12/10/2018, StudioML itself does not support
        #       the 'cleanup_bucket' key.
        cleanup_bucket = self.persistence.get('cleanup_bucket', False)
        if cleanup_bucket:
            self.delete_bucket(bucket)
        else:
            self.delete_experiments(bucket)


    def delete_bucket(self, bucket):
        """
        :param bucket: The S3 Bucket object to delete.

        Per https://boto3.amazonaws.com/v1/documentation/api/latest/guide/migrations3.html
        All of the keys in a bucket must be deleted before the bucket
        itself can be deleted.
        """

        print(("About to delete bucket {0}".format(bucket.name)))
        sys.stdout.flush()

        # Delete all the keys in the bucket.
        # If there are none, then no big deal
        try:
            bucket.objects.all().delete()
        except Exception:
            # swallow the exception
            pass

        # Delete the bucket itself.
        # In normal operation when the bucket is empty
        # this can return a NoSuchBucket exception
        # So we swallow the exception.
        # This ends up working if the bucket had not been created yet as well.
        try:
            bucket.delete()
        except Exception:
            # swallow the exception
            pass

        print(("Finished deleting bucket {0}".format(bucket.name)))
        sys.stdout.flush()

    def delete_experiments(self, bucket):
        """
        :param bucket: The S3 Bucket object to delete.
        """
        for folder in self.folders:
            self.delete_experiments_in_folder(bucket, folder)


    def delete_experiments_in_folder(self, bucket, key_prefix):
        """
        :param bucket: A handle to a Boto3 Bucket instance
                        from which we will delete things.
        :param key_prefix: A 'folder' prefix for the key which will be
                        prepended to any studio experiment id
                        in the list we were sent upon construction.
        """

        for studio_exp in self.studio_experiment_ids:

            folder_key = key_prefix + studio_exp
            folder_prefix_key = folder_key + "/"

            try:
                bucket.objects.filter(Prefix=folder_prefix_key).delete()
            except Exception:
                # Slurp up any errors for folders that do not actually
                # exist yet
                pass
