# ENN Client Documentation

## Overview

ENN stands for the Evolution of Neural Networks.

The client provided here works by asking a service (source not included here)
for a population of neural network architectures in [Keras 2.2.4](https://keras.io/layers/about-keras-layers/)
JSON format.

Upon receving these network descriptions, the client then sends
out requests for each of the candidate networks to be trained for a small
number of epochs by a (possibly remotely distributed) worker process.

After measurements of each candidate network's performance come back,
fitness measurements are sent back to the same ENN service, where the 
candidate networks are used as a basis to create another new, evolved
generation of candidate networks.

## Setup and Installation

The goal of this section will be to get your environment set up enough
for you to run a first small sample experiment.

### Supported Environment

This document assumes:

* You are doing your setup on Ubuntu 18.04 or similar (enough).
* You have root access to the machine to install some basic linux packages.
* You have github credentials and have at least read permissions to clone this repo.
* You have valid AWS account credentials

#### Existing Development Environment

While setting up into a Docker container has the benefit of providing
a controlled environment, the ENN software at this time involves some
Python software development, and we on the project do not feel we should
be dictating what your Python development tooling environment should look like.
Options in a development environment can include anything from GPU hardware to speed up
local evaluations of neural networks to IDE tools for more efficient coding and
debugging.

If you are installing onto a development machine you use on a continuous daily basis,
you can skip to the "Clone ENN Repository" section, as you probably have the essence of 
the extra things required for installing into a docker container. Please note, however,
that we do testing of our own software on clean Ubuntu 18.04 Docker instances.

#### EC2 Instance on AWS

If you are installing onto an EC2 instance in AWS, you'll want to have at
least a t2small instance type and run the ubuntu 18.04 AWS image to start.
Follow the next steps for "Installing into a fresh Docker container".

#### Installing into a fresh Docker container

This section will get you a few minimal tools installed from a clean ubuntu 18.04:

* a text editor (vim in this example, but choose your own if you like)
* the git executable
* AWS credentials
* Credentials for github access


```shell
mkdir $HOME/.aws $HOME/enn-workspace
apt-get update -yqq
apt-get install -yqq vim git
```

After completing this step you will want to copy/paste your AWS credentials into the docker to
```shell
$HOME/.aws/credentials
```

### Clone ENN Repository

Create and clone the current github repository

```shell
cd $HOME/enn-workspace

# This repo is for data scientist release use.
# Developers to the ENN service will have a different repo and branch.
git clone https://github.com/leaf-ai/enn-release.git --branch master

# Get in the repo directory
cd enn-release
```

If you do not have access to the repo, contact Dan.Fink@cognizant.com
to get a username and password for read-only access to the repo. 
Please provide in the email your Slack login (if you have one)
so we can add you to the slack discussion
channels to help guide you through your ENN experience.

### Install Prerequisites

Note this install is for a python 3.6 virtualenv, so any other python
projects and dependencies you may have going in your development environment
will not be disturbed.

```shell
./setup/setup.sh |& tee /tmp/setup-errors.txt
```

This script will install linux and python packages that are prerequisites
for running the ENN client software.  It will also do some rudimentary tests
using your AWS credentials to separate out problems with accessing AWS
from problems with the code itself.

When running this script you might be asked for an ENN_USER name.
Please enter something unique enough for us to identfy you
should you have problems with the ENN service.

Note that you should only ever need to run this setup script
once per machine you install on.

### Set Up Python Environment

After you are done with the setup script, you will be asked to do this
to use the virtualenv and environment variables put together in setup.sh:

```shell
source $HOME/.enn/enn-3.6.sh
```

This sets up a PYTHONPATH for you and some environment variables
described below.

You should see an 'enn-3.6' at the command prompt indicating that you
are using the python virtualenv for enn using python 3.6.

Note that you will need to run this script once per command shell
you use.

### Run your first experiment

If you've had no errors up to this point, it's time to try running
a first small experiment.  

```shell
cd experimenthost
python session_server.py --config_file=../domain/omniglot/config/test_enn/test_config.hocon
```
This configuration runs a very small image classification experiment against
the omniglot domain.  It only runs 2 generations of 2 candidates each for a
very short time, so you will not be breaking any world-records off the bat,
but it will give you some confidence that the system is working.

Experiment results are stored by default inside the enn-release/results directory.
They include a list of checkpoints keys (which can be optionally used to
restart the experiment from a previous point), some visualization of the best
network, and various statistics for each generation.

By default a few nice things happen for you:

1. If you break your experiment host/session server in the middle of a run
    (or it dies) and you re-run with the same experiment name, the run
    will pick up at the last completed checkpoint without having to go
    digging for what the checkpoint id is.

2. Opaque data for checkpoints themselves are persisted on the service side
    beyond the lifetime of the experiment host, making it possible to pick
    up a productive line of inquiry days or weeks later.

These niceties do require you to be sure that your experiment names are
indeed distinct from one another, as it is entirely possible for you to
think you have started a new run from scratch, when really you are
resuming and old one.  When in doubt, clean out your enn-release/results
directory of old experiments.

Note: By default, experiment names consist of a user name + domain name + date + timestamp

### Some Other Domains to Try

In addition to the omniglot image classification domain,
we also provide examples for:

* a chestxray domain which uses a different kind of image classification
  on a publicly available data set of xray images.

    To run a first version of chestxray, do this:

```shell
python session_server.py --config_file=../domain/chestxray/config/test_enn/test_config.hocon
```

* a textclassifier domain which will do 1D text classification
  on a publicly available data set of posts on wikipedia data.

```shell
python session_server.py --config_file=../domain/textclassifier/config/test_enn/test_config.hocon
```
### Set up Errors

#### Errors beyond the scope of this document.

There are some setup situations which are too involved and are beyond the scope
of this document.

##### Problems with AWS

There are many things that can go wrong in trying to connect to AWS.
We make an effort in the setup.sh script to test out reading and writing
to AWS before using any of the ENN software to separate out the source
of these kinds of problems. Despite this, debugging AWS access problems
are beyond the scope of this document.

##### Problems with CUDA

If you see errors having to do with versions of CUDA libraries like ...

```shell
 ImportError: libcublas.so.8.0: cannot open shared object file: No such file or directory)
```

... setting up CUDA libraries and drivers properly on your ubuntu machine
is also its own can of worms and is beyond the scope of this document. 


## Some Details

### References

For more information regarding how LEAF evolves neural networks,
refer to the following papers:

 - [Evolving Deep Neural Networks](https://arxiv.org/pdf/1703.00548.pdf)
 - [Evolutionary Architecture Search for Deep Multitask Networks](https://arxiv.org/pdf/1803.03745.pdf)

An architectural diagram is provided below:
![Architecture Overview](https://s3-us-west-2.amazonaws.com/ml-enn/enn_system_diagram.png)


### Environment Variable Descriptions

The following variables are referenced by the HOCON config files for the domains
and are set up as part of the source-ing of the enn.sh script above:

```shell
# Tells where domain data should come from.
# Often this is S3, but can be your own Minio server for private data.
# Note there is no slash '/' at the end here.
export DOMAIN_DATA_ENDPOINT=http://s3.us-west-2.amazonaws.com

# Places where Studio ML output from distributed workers should go
# Often this is S3, but can be your own Minio server for private data.
# Note there is a slash '/' at the end here.
export STUDIO_DATABASE_ENDPOINT=http://s3.us-east-1.amazonaws.com/
export STUDIO_STORAGE_ENDPOINT=http://s3.us-east-1.amazonaws.com/

# For now this ENN_USER variable is fine to be some name unique enough for us
# to tell who you are on the service back-end when there are problems.
# A linux username could be good enough here (please not 'root' or 'guest').
# Eventually this will become a user name to securely get you connected
# to the service.
export ENN_USER=<your_enn_username>
```

During the course of your experimentations you might find the need to change
one or more of these, or if your situation gets complex enough, add your
own environment variables to your own configs.

It's worth noting that while the system does support config files in other
formats such as JSON and YAML, only the HOCON format has robust enough support
for comments and environment variables within structured data, which is why we favor it.

### StudioML

ENN uses the completion service from StudioML for distributed computing.
The installation of the Studio ML patch in the setup.sh script above should be
enough to get a basic configuration of Studio ML in place.

Optional: If you have never used StudioML before, run `studio ui` , go to `localhost:5000`
in a browser and log in to create an authentication cookie before running ENN server.
[For more information on StudioML, see this link.](https://www.studio.ml)

### Experiment Configuration
There is a single HOCON configuration file for each experiment.
You can think of HOCON as JSON with the ability to comment and reference environment variables.
There are five configuration sections which must be set properly in order for the experiment to run properly.
An example of the configuration files are found in
`enn-release/domain/config/test_enn/`. The sections are:

 - experiment_config: Parameters for running ENN server and workers, including:
    - number of generations to run
    - timeout settings for incomplete work,
    - whether to run workers locally or remotely,
    - configuration for distributing the work via Studio ML

 - domain_config:   Parameters for evaluation and loading data, including:
    - where to get the data
    - how long to train for
    - Domains can add their own configuration parameters here.

 - blueprint_config: Evolution parameters for blueprint coevolution, including:
    - overall population size for each generation

 - module_config: Evolution parameters for module coevolution

 - builder_config:  Parameters for how the neural networks get put together, including:
    - which Keras layers to use
    - specifications for hyperparameter evolution

Descriptions of the parameters for each file can be found within the files
themselves.
 

### Communication with the StudioML Workers

The SessionServer class in experimenthost/session_server.py is the main class
coordinating the communication between the LEAF ENN Service doing the evolution
and the StudioML workers doing the heavy lifting for evaluating a single
candidate.   For each candidate, a python dictionary describing the worker
request is sent to the StudioML worker node, along with a tarball of all related
code to execute on the worker.

#### Worker Request Dictionary

Keys for the dictionary are strings.
Value types are context dependent.

The following fields are germane to the evaluation code:

'config' -  a dictionary containing configuration information for the evaluation

    'domain' - the name of the domain being evaluated
    'domain_config' - the domain_config dictionary as specified in the hocon file.
                      Any domain-specific evaluation parameters will be in here.

            See the "Description of 'domain' Parameters" section above for
            contents.  For information as to how default values are put
            together, see the file framework/domain/domain_config.py

'id' - a string identifier for the candidate, unique to the experiment (at least)

'identity' - information pertaining to the execution of the candidate's evaluation
    'experiment_id' - the string experiment id used as a StudioML identifier

'interpretation' - a dictionary containing the domain-specific interpretation
                    of the candidate.  This dictionary has two main sub-fields:

    'model' - a description of the model to be tested.
              For ENN, this is always a JSON string describing a Keras neural network.

    'global_hyperparameters' - another dictionary containing the evolved global
              hyperparameters. By default, this contains a 'learning_rate' key
              with a float value.


### Code

The directories that get tarred up to be sent over to the workers include the
following:

 - worker
 - worker common
 - framework
 - domain/<domain_name>

Python dependencies and environment variable values are sent over by StudioML automatically
based on the current python packages installed on the machine running the server.

To setup a new domain, the following python code files are needed:

 - NetworkDomainConfig class implementation for loading the dataset (among other things),
        placed as domain/<yourdomain>/<yourdomain>_domain_config.py.
 - KerasNetworkEvaluator class implementation for training and evaluating candidate networks,
        placed under domain/<yourdomain>/<yourdomain>_evaluator.py.
 - New HOCON configuration files for the domain, placed under domain/<yourdomain>/config/<yourconfig>.hocon.

For examples, see existing files under the directories for the provided domains (omniglot, textclassier, etc)
mentioned above.

#### Evaluation on the StudioML worker

The main entry point for evaluation on the StudioML worker is the script found
in framework/client_script/client.py. This script is intended to be put together
in a domain-agnostic manner, so you should not find yourself modifying it.
If you do find yourself needing to modify it, please let us know so we can
work together to make the abstractions better.

In there, the worker request dictionary and code are unpacked from the initial
StudioML payload, evaluation code is unpacked into a local directory, and a
NetworkEvaluator subclass is created whose evaluate_model() method is invoked
to do the heavy lifting of training the candidate for a short number of epochs.

The Studio ML Completion Service is responsible for sending over the payload and also a
dictionary of filepaths mapped to their names. These can include data files and
any auxiliary code needed by the worker for training. The dictionary of filepaths
are translated from paths on the server to those local to the worker. The files
are uploaded by StudioML to cloud storage where the worker can then download them.
S3 urls are also accepted and the workers will automatically download the files from
S3 instead.

In the example code given here, the meat of the network evaluation begins in
framework/evaluator/network_evaluator.py 's evaluate_model() method.
This eventually calls each domain evaluator's evaluate_network() method,
whose implmenentations are specific to the context of each domain.

The evaluate_network() method takes as arguments:
1. candidate_id - a String id for the candidate being evaluated
2. training_model - a Keras model to train and evaluate
3. global_hyperparameters - the dictionary of evolved global hyperparameters
            that are specific to the candidate, but applied globally
            to the evaluation.
4. domain_config - The configuration dictionar for domain evaluation,
            as specified in the HOCON config file.
5. data_dict - a dictionary containing domain keys for each data file used.

Inside evaluate_network(), the train() and test() methods are called.
This begins to put some formalities on top of neural network evaluation,
but really from this point on, you can do what you need to to evaluate
your candidates in whatever your domain demands.

The evaluate_network() method eventually returns a dictionary whose key-value pairs
impart measurement information as to the performance of the model. 
You can measure anything and everything you want in here, but
at least one of these key-value pairs should impart a fitness scalar
which can be used to measure other candidates against each other.
By default, the system looks for a key called 'fitness' with a double value
that is to be maximized.

The dictionary returned by evaluate_network() is packed up into a
worker response dictionary in the client.py script's clientFunction which is
sent back to the experiment host/session server.

#### When Evaluation Goes Wrong

Alas, inevitably, there will be bugs in your evaluation code.
How do you find out what went wrong?

First off, know that any candidates whose evaluations had errors or timed out
are discarded by the system.  Evolution is robust against some of its
candidate population not making it any further than the current generation.

Any worker responses which are not parseable by the session server are shown
in the session server output, interspersed with the regular progress bar for
the generstion. This includes exceptions thrown in the evaluation code run on the
StudioML worker.

Sometimes the exception information isn't enough and you want to dive deeper.
This gets into the realm of StudioML.  To find studio task output ...

cd $HOME/.studioml

In here will be a long list of directories, each of which correspond to a
single StudioML worker task.  The task directories have the following name
structure under $HOME/.studioml/experiments/:

completion_service_<experiement-name>-<unique-studio-job-id>

Note: job ids are not tied to candidate ids, to find the job id corresponding
to a candidate id, grepping is required.

Inside one of these directories you will see a number of files, the most
important of which are:

1. retval -- this is what was returned to the session server from the worker.
    1.  If you don't see this, your job hasn't finished.
    2.  If you do see this, and it looks like gobbledygook, it's probably
        a pickled dictionary with valid results.
    3.  If you see this file and you can read it, it's probably the
        stacktrace describing where your evaluation code failed.

2. output -- this is basically the logged contents of standard out from the
            evaluation task itself.  You can use this to get a feel for what
            happened before your code died.
