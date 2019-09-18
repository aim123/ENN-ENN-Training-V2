

import logging
import threading
import time

import pika

from past.builtins import basestring

import studio.logs as logs

from experimenthost.util.shutdown_task import ShutdownTask


class RabbitShutdownTask(ShutdownTask):
    """
    Given some information about a RabbitMQ queue,
    This shutdown task will:
        1. Delete the queue
        2. Remove all outgoing work (to Workers)
        3. Remove all incoming results (from Workers)

    The basis for this implementation came from StudioML's own rabbit_queue.py
    which uses the pika library.  Pika works through a series of asynchronous
    callbacks that are daisy-chained together:

        Do X and register callback Y.
        When pika sees X is complete, the callback Y is invoked
        and then perhaps another pika command/callback combination is issued
        until what you want is complete.
    """

    # Tied for Public Enemy #8 for too-many-instance-attributes
    # pylint: disable=too-many-instance-attributes
    def __init__(self, config):
        """
        Setup the example publisher object, passing in the URL we will use
        to connect to RabbitMQ.

        :param config: The completion_service config dictionary
        """

        self._config = config
        self._queue = self._find_queue_name(config)
        self._url = self._find_queue_server_url(config)

        # Initialize member variables

        self._routing_key = 'StudioML.'
        if self._queue is not None:
            self._routing_key = self._routing_key + self._queue

        self._rmq_lock = threading.RLock()
        self._connection = None
        self._channel = None

        self._stopping = False
        self._exchange = 'StudioML.topic'
        self._exchange_type = 'topic'

        self._logger = logs.getLogger('RabbitMQShutdown')
        self._logger.setLevel(logging.INFO)

        self._cleanup_done = False


    def shutdown(self, signum=None, frame=None):
        """
        Perhaps a misleading name given the context...
        ShutdownTask interface and main entry point to initiate
        the cleanup.
        """

        # Only do this cleanup if the queue is designated as a RabbitMQ queue.
        if self._queue is None or \
           not self._queue.startswith('rmq_'):
            self._logger.info("Not doing RMQ cleanup for queue %s",
                                str(self._queue))
            return

        self._logger.info("Initiating RMQ cleanup for queue %s",
                            str(self._queue))
        self._cleanup()


    def _find_studio_ml_config(self, config):
        """
        :return: the studio ml config dictionary (if any)
        """

        empty = {}
        studio_config = config.get('studio_ml_config', empty)
        if studio_config is None or \
            isinstance(studio_config, basestring):
            # XXX We could go through studio config yaml files, but not yet.
            studio_config = empty

        return studio_config


    def _find_queue_name(self, config):
        """
        :return: the queue name (if any)
        """

        queue_name = config.get('queue', None)
        if queue_name is None:
            studio_config = self._find_studio_ml_config(config)
            # XXX Is this correct?
            queue_name = studio_config.get('queue', None)

        # Don't clean up a queue if nobody wanted it cleaned up
        if not config.get('cleanup', False):
            queue_name = None

        return queue_name


    def _find_queue_server_url(self, config):
        """
        Find the url for the queue server
        """

        studio_config = self._find_studio_ml_config(config)
        empty = {}

        cloud = studio_config.get('cloud', empty)
        if cloud is None:
            cloud = empty

        cloud_queue = cloud.get('queue', empty)
        if cloud_queue is None:
            cloud_queue = empty

        queue_server_url = cloud_queue.get('rmq', None)

        return queue_server_url


    def _cleanup(self):
        """
        Do the work of cleaning up.
        """

        # The pika library for RabbitMQ has an asynchronous run method
        # that needs to run forever and will do reconnections etc
        # automatically for us
        thr = threading.Thread(target=self.run, args=(), kwargs={})
        thr.setDaemon(True)
        thr.start()

        while not self._cleanup_done:
            time.sleep(1)


    def run(self):
        """
        Blocking run loop, connecting and then starting the IOLoop.
        """

        self._logger.info('RMQ cleanup started')
        while not self._stopping:
            self._connection = None
            try:
                with self._rmq_lock:
                    self._connection = self.connect()
                self._logger.info('RMQ connected')
                self._connection.ioloop.start()
            except KeyboardInterrupt:
                self.stop()
                if (self._connection is not None and
                        not self._connection.is_closed):
                    # Finish closing
                    self._connection.ioloop.start()

        self._logger.info('RMQ stopped')


    def connect(self):
        """
        When the connection is established, the on_connection_open method
        will be invoked by pika. If you want the reconnection to work, make
        sure you set stop_ioloop_on_close to False, which is not the default
        behavior of this adapter.

        :rtype: pika.SelectConnection
        """

        params = pika.URLParameters(self._url)
        return pika.SelectConnection(
            params,
            on_open_callback=self.on_connection_open,
            on_close_callback=self.on_connection_closed,
            stop_ioloop_on_close=False)


    def on_connection_open(self, unused_connection):
        """
        :type unused_connection: pika.SelectConnection
        """

        self.open_channel()


    def open_channel(self):
        """
        open a new channel using the Channel.Open RPC command. RMQ confirms
        the channel is open by sending the Channel.OpenOK RPC reply, the
        on_channel_open method will be invoked.
        """

        self._logger.debug('creating a new channel')

        with self._rmq_lock:
            self._connection.channel(on_open_callback=self.on_channel_open)


    def on_channel_open(self, channel):
        """
        on channel open, declare the exchange to use

        :param pika.channel.Channel channel: The channel object
        """

        self._logger.debug('created a new channel')

        with self._rmq_lock:
            self._channel = channel
            self._channel.basic_qos(prefetch_count=0)
            self._channel.add_on_close_callback(self.on_channel_closed)

        self.setup_exchange(self._exchange)


    def setup_exchange(self, exchange_name):
        """
        Exchange setup by invoking the Exchange.Declare RPC command.

        An 'exchange' is a pika concept which is just a logical grouping
        of queues.  The grouping is known as a 'topic'.

        When complete, the on_exchange_declare_ok method will be invoked
        by pika.

        :param str|unicode exchange_name: The name of the exchange to declare
        """

        self._logger.debug("declaring exchange %s", str(exchange_name))
        with self._rmq_lock:
            self._channel.exchange_declare(callback=self.on_exchange_declare_ok,
                                           exchange=exchange_name,
                                           exchange_type=self._exchange_type,
                                           durable=True,
                                           auto_delete=True)


    def on_exchange_declare_ok(self, unused_frame):
        """
        completion callback for the Exchange.Declare RPC command.

        :param pika.Frame.Method unused_frame: Exchange.DeclareOk response
        """

        self._logger.debug("declared exchange %s", str(self._exchange))
        self.setup_queue(self._queue)


    def setup_queue(self, queue_name):
        """
        Setup the queue invoking the Queue.Declare RPC command.
        The completion callback is, the on_queue_declareok method.

        :param str|unicode queue_name: The name of the queue to declare.
        """
        self._logger.debug("declare queue %s", str(queue_name))
        with self._rmq_lock:
            self._channel.queue_declare(self.on_queue_declare_ok, queue_name)


    def on_queue_declare_ok(self, method_frame):
        """
        Queue.Declare RPC completion callback.
        In this method the queue and exchange are bound together
        with the routing key by issuing the Queue.Bind
        RPC command.

        The completion callback is the on_bind_ok method.

        :param pika.frame.Method method_frame: The Queue.DeclareOk frame
        """

        # Note: _ is pythonic for unused argument
        _ = method_frame

        self._logger.debug("Binding %s to %s with %s",
                            str(self._exchange),
                            str(self._queue),
                            str(self._routing_key))
        with self._rmq_lock:
            self._channel.queue_bind(self.on_bind_ok, self._queue,
                                     self._exchange, self._routing_key)


    def on_bind_ok(self, unused_frame):
        """
        This method is invoked by pika when it receives the Queue.BindOk
        response from RabbitMQ. Since we know we're now setup and bound, it's
        time to start publishing.
        """

        self._logger.info("Bound %s to %s with %s",
                             str(self._exchange),
                             str(self._queue),
                             str(self._routing_key))

        with self._rmq_lock:
            self._channel.queue_purge(callback=self.on_purge_ok,
                                      queue=self._queue)


    def on_purge_ok(self, unused_frame):
        """
        This method is invoked by pika when it receives the Queue.PurgeOk
        response from RabbitMQ.
        """

        self._logger.info("queue %s purged.", str(self._queue))
        with self._rmq_lock:
            self._channel.queue_unbind(callback=self.on_unbind_ok,
                                       queue=self._queue,
                                       exchange=self._exchange,
                                       routing_key=self._routing_key)


    def on_unbind_ok(self, unused_frame):
        """
        This method is invoked by pika when it receives the Queue.UnbindOk
        response from RabbitMQ.
        """

        self._logger.info("Unbound %s from %s with %s",
                            str(self._exchange),
                            str(self._queue),
                            str(self._routing_key))

        with self._rmq_lock:
            self._channel.queue_delete(callback=self.on_delete_ok,
                                       queue=self._queue)


    def on_delete_ok(self, unused_frame):
        """
        This method is invoked by pika when it receives the Queue.DeleteOk
        response from RabbitMQ.
        """

        self._logger.info("Deleted queue %s", str(self._queue))

        # We are done with Rabbit MQ, though we still might want to
        # delete things via minio.  That can be for another class.
        self.stop()


    def stop(self):
        """
        Stop the by closing the channel and connection and setting
        a stop state.

        The IOLoop is started independently which means we need this
        method to handle things such as the Try/Catch when KeyboardInterrupts
        are caught.
        Starting the IOLoop again will allow the publisher to cleanly
        disconnect from RMQ.
        """

        self._logger.info('stopping')
        self._stopping = True
        self.close_channel()
        self.close_connection()


    def close_channel(self):
        """
        Close channel by sending the Channel.Close RPC command.
        """

        with self._rmq_lock:
            if self._channel is not None:
                self._logger.info('closing the channel')
                self._channel.close()


    def on_channel_closed(self, channel, reply_code, reply_text):
        """
        physical network issues and logical protocol abuses can
        result in a closure of the channel.

        :param pika.channel.Channel channel: The closed channel
        :param int reply_code: The numeric reason the channel was closed
        :param str reply_text: The text reason the channel was closed
        """

        # Note: _ is pythonic for unused argument
        _ = channel

        self._logger.info("channel closed %s %s", str(reply_code),
                            str(reply_text))
        with self._rmq_lock:
            self._channel = None
            if not self._stopping:
                self._connection.close()


    def close_connection(self):
        with self._rmq_lock:
            if self._connection is not None:
                self._logger.info('closing connection')
                self._connection.close()


    def on_connection_closed(self, connection, reply_code, reply_text):
        """
        on any close reconnect to RabbitMQ, until the stopping is set

        :param pika.connection.Connection connection: The closed connection obj
        :param int reply_code: The server provided reply_code if given
        :param str reply_text: The server provided reply_text if given
        """

        # Note: _ is pythonic for unused argument
        _ = connection

        with self._rmq_lock:
            self._channel = None
            if self._stopping:
                self._connection.ioloop.stop()
            else:
                # retry in 5 seconds
                self._logger.info(
                    "connection closed, retry in 5 seconds: %s %s",
                        str(reply_code), str(reply_text))
                self._connection.add_timeout(5, self._connection.ioloop.stop)

            self._cleanup_done = True
