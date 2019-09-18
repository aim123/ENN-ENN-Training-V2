
#!/bin/bash


# A script intended to be sourced so that initial
# python virtualenv and essential environment variables
# can be set up

# This virtualenv is established in the companion setup.sh script
VENV=$HOME/venv/enn-PYTHON_VERSION_FROM_TEMPLATE
echo
echo "Setting up python virtualenv $VENV"
source $VENV/bin/activate


####
# Essential Environment Variables
####

if [ "x$DOMAIN_DATA_ENDPOINT" == "x" ]
then
    # Tells where domain data should come from.
    # Often this is S3, but can be your own Minio server for private data.
    # Note there is no slash '/' at the end here.
    export DOMAIN_DATA_ENDPOINT=http://s3.us-west-2.amazonaws.com
fi


# Places where Studio ML output from distributed workers should go
# Often this is S3, but can be your own Minio server for private data.
# Note there is a slash '/' at the end here.
if [ "x$STUDIO_DATABASE_ENDPOINT" == "x" ]
then
    export STUDIO_DATABASE_ENDPOINT=http://s3.us-east-1.amazonaws.com/
fi

if [ "x$STUDIO_STORAGE_ENDPOINT" == "x" ]
then
    export STUDIO_STORAGE_ENDPOINT=http://s3.us-east-1.amazonaws.com/
fi


if [ "x$ENN_USER" == "x" ]
then
    # For now this ENN_USER variable is fine to be some name unique enough for us
    # to tell who you are on the service back-end when there are problems.
    # A linux username could be good enough here (please not 'root' or 'guest').
    # Eventually this will become a user name to securely get you connected
    # to the service.
    export ENN_USER=ENN_USER_FROM_TEMPLATE
fi

# Additions to the standard PYTHONPATH variable
check_release=` echo $PYTHONPATH | grep PYTHONPATH_FROM_TEMPLATE `
if [ "x$check_release" == "x" ]
then
    export PYTHONPATH=PYTHONPATH_FROM_TEMPLATE:$PYTHONPATH
fi

echo "If this is your first time running ENN consider running these commands"
echo "next to run your first small ENN experiment that tests if everything"
echo "is set up correctly:"
echo "  cd experimenthost"
echo "  python session_server.py --config_file=../domain/omniglot/config/test_enn/test_config.hocon"
