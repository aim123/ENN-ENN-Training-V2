#!/bin/bash

# A script to assist in setting up prerequisites for ENN runs.
# This could be run from inside a Docker container if need be,
# but note we do not necessarily want to *require* Docker as
# part of the ENN setup.

PYTHON_VERSION=3.6

# Allow for development repo, when set
if [ "x${ENN_REPO}" == "x" ]
then
    ENN_REPO=enn-release
fi

# Allow for alternate data center configuration, when set
DATA_AWS_S3_CMD="aws s3"
if [ "x${DOMAIN_DATA_ENDPOINT}" != "x" ]
then
    # In some data center situations, we need to specify the
    # DOMAIN_DATA_ENDPOINT as s3://<address>>
    # That will not fly for aws CLI, so convert it to an http address
    USE_DOMAIN_DATA_ENDPOINT=`echo $DOMAIN_DATA_ENDPOINT | sed -e "s/s3:/http:/g"`
    DATA_AWS_S3_CMD="aws s3 --endpoint-url=${USE_DOMAIN_DATA_ENDPOINT}"
fi

# Allow for alternate data center configuration, when set
STUDIO_AWS_S3_CMD="aws s3"
STUDIO_AWS_S3API_CMD="aws s3api"
if [ "x${STUDIO_STORAGE_ENDPOINT}" != "x" ]
then
    STUDIO_AWS_S3_CMD="aws s3 --endpoint-url=${STUDIO_STORAGE_ENDPOINT}"
    STUDIO_AWS_S3API_CMD="aws s3api --endpoint-url=${STUDIO_STORAGE_ENDPOINT}"
fi

# Find sudo to apt-get install prerequisite linux packages.
# If running from inside a container, this might not exist,
# so attempt to work without it assuming root
SUDO=`which sudo`

#
# Parse arguments
#
use_apt_get=`echo $1 | grep "\--no-sudo"`


#
# Check Assumptions
#

# Everything here needs to be done from the top-level directory of the repo
working_dir=`pwd`
exec_dir=`basename $working_dir`
desired_dir=${ENN_REPO}
if [ "$exec_dir" != "$desired_dir" ]
then
    echo "This script must be run from the top-level $desired_dir directory"
    exit 1
fi


#
# Start to do work
#

echo
echo "Obtaining ENN_USER name..."
# First see if we already have a value from script environment
if [ "x$ENN_USER" == "x" ]
then
    # If that doesn't work, try getting it as an argument from the command line
    if [ "x$use_apt_get" == "x" ]
    then
        ENN_USER=$1
    fi
fi
# If that doesn't work, try getting it from the user's HOME directory name 
if [ "x$ENN_USER" == "x" ]
then
    ENN_USER=`basename $HOME`
fi

function is_valid_enn_user() {

    INVALID_USER_NAMES="root ubuntu guest enn enn_user ennuser leafenntraining leaf-enn-training ec2-user"

    is_valid="true"
    local_enn_user=$1
    if [ "x$local_enn_user" == "x" ]
    then
        is_valid="false"
    else
        for invalid in $INVALID_USER_NAMES
        do
            if [ "x$local_enn_user" == "x$invalid" ]
            then
                is_valid="false"
            fi
        done          
    fi

    echo $is_valid
}

# See if what we tried to get automatically is valid.
# If not, ask for valid username as user input
while [ `is_valid_enn_user $ENN_USER` == "false" ]
do
    echo "Please enter a valid value for ENN_USER"
    echo "This should be something that we can identify you uniquely"
    echo "on the service backend, such as <first_name>_<last_name>."
    read ENN_USER
done
echo "Using $ENN_USER as ENN_USER environment variable"


# Check for the OS we are installing on.
OS_RELEASE_FILE=/etc/os-release
os_name=""
os_version=""
if [ -f $OS_RELEASE_FILE ]
then
    # Likely linux
    os_name=`cat $OS_RELEASE_FILE | sed -e "s/\"//g" | grep ^NAME | sed -e "s/^NAME=//g"`
    os_version=`cat $OS_RELEASE_FILE | sed -e "s/\"//g" | grep ^VERSION_ID | sed -e "s/^VERSION_ID=//g"`
fi

if [ "x$os_name" != "xUbuntu" ]
then
    echo
    echo "WARNING:"
    echo "It's not clear that this is machine is running a supported Linux release." 
    echo "You might have problems later on."
    echo
fi


# Check for Ubuntu 16.04 to see if we need to add a PPA distro.
if [ "x$os_name" == "xUbuntu" ] \
    && [ "x$os_version" == "x16.04" ]
then
    echo "Found Ubuntu 16.04. Adding PPA distribution for Python ${PYTHON_VERSION}"
    $SUDO apt-get update

    # Get add-apt-repository if it doesn't exist already
    $SUDO apt-get install -y software-properties-common

    # See https://en.wikipedia.org/wiki/Python_(mythology)
    # ... as to the significance of the name "deadsnakes"
    $SUDO add-apt-repository -y ppa:deadsnakes/ppa
fi


# Assuming Ubuntu 18.04 from here on out
echo
if [ "x$use_apt_get" == "x" ]
then
    # Get python OS prerequisites
    echo "Obtaining linux dependencies ..."
    $SUDO apt-get update
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y \
        gawk \
        wget \
        tzdata \
        python-pip \
        python-setuptools \
        virtualenv \
        python${PYTHON_VERSION} \
        python3-dev \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python-pydot \
        graphviz
else
    echo "Skipping linux dependencies ..."
fi


# Some extra prerequisites if we are on a mac
BREW=`which brew`
if [ "x$BREW" != "x" ]
then
    echo
    echo "Obtaining linux dependencies for mac..."
    $BREW install gnu-tar
    ln -s /usr/local/bin/gtar /usr/local/bin/tar
fi


# Display the version of tar for potential debugging purposes.
# We want this to be GNU tar.
echo
echo "Tar version is ..."
tar --version

# Creating a virtualenv using correct python version for all
# the ENN dependencies to go into
venv=$HOME/venv/enn-${PYTHON_VERSION}
echo
echo "Creating and activating virtualenv $venv"
mkdir -p $venv
virtualenv --python=/usr/bin/python${PYTHON_VERSION} $venv
source $venv/bin/activate


echo
echo "Installing pip version that will work with Studio ML dependency..."
DESIRED_PIP_VERSION=19.0.3
pip install pip==${DESIRED_PIP_VERSION}
pip_version=`pip --version | gawk '{ print $2 }'`
if [ "x$pip_version" != "x${DESIRED_PIP_VERSION}" ]
then
    echo "New pip version did not install successfuly"
    exit 1
fi
echo "New pip version installed successfuly"


echo
echo "Fetching and installing ENN requirements..."
pip install -U -r requirements.txt
if [ -f requirements-local.txt ]
then
    pip install -U -r requirements-local.txt
fi
if [ "x$os_name" == "xUbuntu" ] \
    && [ "x$os_version" == "x16.04" ]
then
    # Don't know why, but errors show up in 16.04 installs
    # that go away if we install the requirements again.
    echo "Reinstalling requirements for Ubuntu 16.04 to avoid sort() errors above"
    pip install -U -r requirements.txt
fi


echo
STUDIO_ML_TAR=""
if [ "x$STUDIO_ML_TAR" != "x" ]
then 
    # The Studio ML convention is that any patch files located
    # in a dist directory where the main python script is run
    # will also be installed as part of the pip environment
    # on the Studio Worker side.
    echo "Fetching and installing latest Studio ML patch $STUDIO_ML_TAR ..."
    DIST_DIR=experimenthost/dist
    DIST_FILE=$DIST_DIR/$STUDIO_ML_TAR
    mkdir -p $DIST_DIR
    # Remove any patches that might have been in there before
    rm -rf $DIST_DIR/*
    wget https://s3-us-west-2.amazonaws.com/studioml-test/$STUDIO_ML_TAR -O $DIST_FILE
    pip install $DIST_FILE
else
    echo "No Studio ML patch needed"
fi


# See if we need to generate any files
echo
if [ -f setup/generate.sh ]
then
    echo "Generating files..."
    setup/generate.sh
else
    echo "No files to generate."
fi


echo
echo "Testing AWS read of test file..."
TEST_FILE=data_file.tar.gz
${DATA_AWS_S3_CMD} cp s3://ml-enn/deepbilevel_datafiles/omniglot/$TEST_FILE /tmp
if [ ! -f /tmp/$TEST_FILE ]
then
    echo "AWS read of test file failed."
    echo "Please check that you have valid AWS credentials in $HOME/.aws/credentials"
    echo "or in the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
    exit 1
fi
echo "AWS read of test file succeeded"


echo
echo "Attempting to create AWS buckets for ENN..."
# These are buckets that are referenced in the hocon config files for ENN
if [ "x${DATABASE_BUCKET}" == "x" ]
then
    DATABASE_BUCKET=${ENN_USER}-enn-studioml-database
fi

if [ "x${STORAGE_BUCKET}" == "x" ]
then
    STORAGE_BUCKET=${ENN_USER}-enn-studioml-storage
fi

REGION_EXTRAS=""
if [ "x${AWS_DEFAULT_REGION}" != "x" ]
then
    REGION_EXTRAS="--region ${AWS_DEFAULT_REGION} --create-bucket-configuration LocationConstraint=${AWS_DEFAULT_REGION}"
fi


${STUDIO_AWS_S3API_CMD} create-bucket --bucket ${DATABASE_BUCKET} ${REGION_EXTRAS}
ls_first=`${STUDIO_AWS_S3_CMD} ls ${DATABASE_BUCKET} | grep "NoSuchBucket"`
if [ "x$ls_first" != "x" ]
then
    echo "${STUDIO_AWS_S3API_CMD} create-bucket --bucket ${DATABASE_BUCKET} ${REGION_EXTRAS} failed."
    echo "Please check that you have valid AWS credentials in $HOME/.aws/credentials"
    echo "or in the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
    echo "and that the bucket mentioned above has correct permissions,"
    echo "and/or doesn't already exist in another region than the default"
    exit 1
fi
${STUDIO_AWS_S3API_CMD} create-bucket --bucket ${STORAGE_BUCKET} ${REGION_EXTRAS}
ls_first=`${STUDIO_AWS_S3_CMD} ls ${STORAGE_BUCKET} | grep "NoSuchBucket"`
if [ "x$ls_first" != "x" ]
then
    echo "${STUDIO_AWS_S3API_CMD} create-bucket --bucket ${STORAGE_BUCKET} ${REGION_EXTRAS} failed."
    echo "Please check that you have valid AWS credentials in $HOME/.aws/credentials"
    echo "or in the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
    echo "and that the bucket mentioned above has correct permissions,"
    echo "and/or doesn't already exist in another region than the default"
    exit 1
fi
echo "AWS creation of ENN buckets succeeded"


echo
echo "Setting up ENN Environment variables..."
mkdir -p $HOME/.enn
USER_ENN_ENV=$HOME/.enn/enn-${PYTHON_VERSION}.sh
cp setup/enn-env-template.sh $USER_ENN_ENV
sed -i -e "s/ENN_USER_FROM_TEMPLATE/${ENN_USER}/g" $USER_ENN_ENV
sed -i -e "s/PYTHONPATH_FROM_TEMPLATE/${working_dir//\//\\/}/g" $USER_ENN_ENV
sed -i -e "s/PYTHON_VERSION_FROM_TEMPLATE/${PYTHON_VERSION}/g" $USER_ENN_ENV
rm -rf $HOME/.enn/env.sh
ln -s ${USER_ENN_ENV} $HOME/.enn/env.sh

echo
echo "Your ENN setup is almost complete."
echo "To finish the last bit, please run this command to set up"
echo "essential environment variables and use the ENN python virtualenv $venv"
echo "      source ${USER_ENN_ENV}"
