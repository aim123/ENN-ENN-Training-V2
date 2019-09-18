
# Execution Flow and Output Files

## Execution Flow

The execution flow for ENN has a few different phases:

1. Evolution

    Evolution is run for a number of generations. Each generation evaluates
    the fitness of a number of candidates.  For the most part all apart from
    a very small handful of candidates we do not care about.  These are
    referred to as the "cattle".  The cattle candidates are each only trained
    for a small number of epochs, so when the evolution is done, we still
    do not yet have a fully trained model.   Note that this evolution phase
    can take days to weeks to complete for real runs of several (~100)
    candidates per generation and will cost real money for the compute time
    ($100s to low $1000s of aws EC2 machine time).

    To find the best model in evolution, it is a reasonable thing to do to
    go to the last generation of evolution and pick out the best candidate.
    (See gen_NN/best_fitness_candidate.json description below.)  Technically
    speaking, however, the candidate with the highest fitness might have occurred
    in a previous generation if the fitness happened to have converged earlier.

2.  Full Training

    Once you have found one or more candidates which have the best architecture,
    you must fully train these candidates.  By fully training the networks,
    we graduate the candidates from being "cattle" -- architecture descriptions
    whose weights we do not care about -- to being "pets" -- models whose
    weights we do care about.  Instead of training for a small number of
    epochs, we want to train the "pets" until convergence.  After full
    training, the "pets" are generally models that we would want to use.

    As of 6/2019 This full training process is a manual one.

3.  Model Deployment

    Now that we have a model, it is useful to deploy it in a customer's own
    ecosystem.  This, however, is generally not something that the LEAF team is
    well equipped to help with, as details of deployment are very customer
    specific.


## Output Files

### Evolution

#### Results Directory

    There are a number of files that are output by the system as an evolution
    run progresses.  The first thing to know is that all files from all
    experiments are put in a single folder.  By default this folder is under
    the top-level directory of the repo and is called...

    ```
    <repo/base>/results
    ```

#### Experiment Results Directory

    Each experiment gets its own folder under the results directory.
    By default, experiment names and their corresponding results folders
    are of the form:

        <enn_user>_<domain>_<config_name>_<timestamp>

    A number of files get put in this directory.

    Whenever possible, by default we try to output files in standard file
    formats so that they could be parsed by tools in potentially different
    languages than Python, and/or file formats that could be included in
    a web front-end.  To this end, you will tend to see files with suffixes
    like .csv for tables, .json for structured data, .png for images, and
    .txt for log files.

##### checkpoint_ids.json

    This file stores the checkpoint ids for each generation seen so far by
    the experiment.  If the run stops for any reason (error or ctrl-c), this
    file is one of the files read to recover the experiment where it left off.

##### completion_service_log.txt

    Log of activity from the Studio ML Completion Service.

##### experiment_host_log.txt

    This is the log of the output of the Experiment Host for the entire
    experiment.  Theoretically anything that goes to stdout or stderr should be
    captured.

##### experiment_host_stats.json

    A structured data file for various domain-agnostic operational statistics
    gathered during the course of a run by the Experiment Host
    (aka session_server.py).

##### experiment_host_stats.png

    A rudimentary visualization of some of the key stats provided in 
    the experiment_host_stats.json file (above).

##### fitness.csv

    A table of fitness data where there is a single row for each generation
    processed by the experiment, and columns representing things like the
    generation number, a time stamp of when the generation completed,
    and the best and average fitness values for that generation.
    The intent is that this file contain the raw data for progress graphs.

##### master_config.json

    A JSON file containing the configuration as the system sees it, after all
    default values, overlay values, and environment variables are considered.
    This is intended to be used for debugging your .hocon configurations.

##### studio_config.yaml

    This is a config file to which the ENN Session Server will point StudioML
    to get its configuration for the workers.  Studio ML requires this to be in
    YAML format.  This is extracted from the experiment .hocon config file by
    the fully resolved key:
    ```
    experiment_config.completion_service.studio_ml_config
    ```

##### worker_code_snapshot.tar.gz

    This file is the tarball of all the code that gets set to a remote
    Studio ML worker.  If you are having confounding evaluation problems 
    where you suspect you do not have the right code getting to the worker,
    you can take this apart to help debug your problem. First thing to try
    to see what is in there is:

    ```
    tar tvf worker_code_snapshot.tar.gz
    ```

#### gen_NN (directories)

    These are a series of directories for each generation. The NN above is
    filled in with a zero-padded 2-digit number for easy sorting.
    (It is our observation that most ENN runs do not go beyond 100 generations
     before converging, hence only 2 digits.)

    All files in each directory pertain to a specific generation, with the
    exception of error files which go in their own errors directory (see below).

##### best_fitness_candidate.json

    After all candidates have been evaluated, the system determines the best
    candidate for a particular generation based on fitness. This JSON file
    completely describes the data representing that best candidate of the
    generation.

    This file will actually have duplicate contents with respect to of one of
    the candidate_XXX.json files (see below), but it remains as an easy way
    to quickly find the most salient results of a generation.

##### best_fitness_candidate png files

    After all candidates have been evaluated, the system determines the best
    candidate for a particular generation based on fitness. These files are
    each a rendering of the architecture for the best candidate described in
    best_fitness_candidate.json (above).  Each file highlights different
    aspects of the *same* candidate.

###### best_fitness_candidate_see_nn_nested.png

    This is probably the first rendering you will want to take a look at
    and it is often the most impressive, especially for networks in later
    generations.  It shows all the connections between all the layers of the
    entire network in a nice colorful format.  On display in this rendering is
    all the complexity that ENN can deliver.

###### best_fitness_candidate_see_nn_blueprint.png

    This rendering is one where the higher-level inter-module connectivity
    evolution of the blueprint for the candidate is highlighted.  The contents
    of each co-evolved module is collapsed so as to highlight only the module
    connectivity.  This isn't so interesting in the early generations, but in
    later generations such a bird's-eye view helps put the complexity of
    candidates into context.

###### best_fitness_candidate_see_nn_modules.png

    This rendering is one where the lower-level inter-layer connectivity
    evolution of each modules in the candidate is highlighted.  The layer
    contents and connectivity of each co-evolved module are laid out one by
    one, so as to highlight the repeated elements that are used throughout
    the candidate.  Again, this isn't so interesting in earlier generations,
    but in later generations the repeated module details begin to help in
    the intelligibility of what is really happening in the candidate network.

##### candidate_XXX.json

    The XXX in the file name refers to the id of the candidate whose
    data is saved.

    The major components of the data include:
###### id field
        This is a string identifier unique (enough) to at least the experiment
###### identity field
        This is a dictionary that describes data about the birth circumstances
        of the candidate
###### metrics field
        These are the metrics gathered by evaluation of the candidate.
###### interpretation field
        The interpretation itself is the meat of the candidate.
        In any application, this is what has been evolved, but turned into a
        format that an evaluator can digest.
        For ENN, this consists of two primary evolved fields:

####### global_hyperparameters field

    These are the evolved hyperparameters whose evolutionary boundaries are set
    by the .hocon config key "builder_config.evaluation_hyperparameters_spec".
    Most domains at least evolve a learning_rate field which is a double,
    but you can change what gets evolved in the hocon to be any structured
    data (see README-specs.md for details). 

####### model field

    This is a description of the Keras JSON that describes the architecture of
    the candidate.  It's worth noting that there are *no* weights referred to
    in this description, as what ENN is trying to address is the problem of
    architecture search.

##### population_results.json

    This is JSON describing the list of candidates as they are sent back to the
    ENN service including all of the candidates metrics.  This file is useful in
    debugging situations.

##### results_dict.json

    This JSON contains a dictionary of candidate ids to loose descriptions of
    candidates so far.  It gets updated every time a new evaluation result is
    returned and is used for recovery of state when restarting an experiment.
    By the end of the generation, the contents of this file are largely
    redundant with what is in the population_results.json file.

#### errors (directory)

    This directory is the place for specific errors which are not necessarily
    fatal to the experiment run.  All errors will have a generation and a
    timestamp associated with them, but they differ in their scope ...

##### experiment_host_error_gen_NN_TTTTTTT.txt

    Contains errors from the SessionServer itself in the form of text from
    Python Tracebacks.

    The NN in the filename refers to the generation the error was seen and the
    TTTTTTT refers to a timestamp generated when the error was first encountered.
  
##### evaluation_error_gen_NN_candidate_XX_TTTTTTT.json

    The NN in the filename refers to the generation the error was seen and the
    TTTTTTT refers to a timestamp generated when the error was first encountered.
    The XX in the filename refers to the candidate's id which is unique to the
    experiment.

    This file contains the full JSON description of the candidate that had
    errors.  (See candidate_XXX.json above for a description of high-level
    fields.)  Any single evaluation error is not fatal to the evolution
    process, as the system simply discards candidates with errors as being
    not elligble for the selection process (their fitness is set to None)
    on the server.

    Most important in this file is the "eval_error" field in the metrics
    which gives Python Traceback information as to where in the evaluation
    code the error occurred when it was running on the remote StudioML worker.

###### Debugging Evaluation Errors

    When you have the checkpoint id and candidate id handy, you can invoke the
    SessionServer to run evaluation locally in such a fashion that the work for
    the single candidate is not distributed via StudioML/Completion Service but
    is instead initiated in-line in the Session Server thread of execution.
    When you have this, you can use a debugger like PyCharm to see where
    things are hanging up.  To invoke the session server this way do this:
 
    ```
    python session_server.py \
        --experiment_name=<your_previously_run_experiment_name> \
        --config_name=<your_config_path> \
        --checkpoint_id checkpoint_<generation_number> \
        --evaluate_locally <candidate_id>

    ```
