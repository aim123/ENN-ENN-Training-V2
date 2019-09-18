
## Fitness Specification for Evolution

In general, if you can measure something about your network,
you can use it as a fitness metric.
 
There are a few places you need to add information for new fitness metrics:
 
###  Domain Evaluator
 
    In the <domain>_evaluator.py code, there is a method called
    evaluate_network().  The return value of this method is a
    dictionary with string keys.  Anything you want to measure
    about your candidate networks you can put in this dictionary.
    It doesn't have to be a scalar.
 
    If you want to use one of these measurements as a fitness
    metric, it *does* need to be a scalar float.  By default, all our
    example domains return a value for a 'fitness' key in that metrics
    dictionary.
 
    When evaluation information is sent back to the ENN service,
    that entire metrics dictionary is sent with its corresponding
    candidate.
 
    In a concrete example, you can see omniglot_evaluator.evaluate_network()
    filling in the value for 'fitness' in the metrics dictionary with the
    computed value of overall_acc[uracy] returned by the
    omniglot_evaluator.test() method.
 
###  The HOCON config file
 
    Recall from the ENN presentations that ENN is doing a coevolution
    of populations of modules (NN layer-level connectivity evolution) and
    blueprints (module-level connectivity) evolution.  Each of these
    populations gets the same fitness information, but you only have
    to specify your fitness information in one place on the blueprint_config.
 
    By default, the assumed fitness specification looks like this:
 
```
    "blueprint_config": {
        ...

        "fitness": [
            {
                "metric_name": "fitness",
                "maximize": true
            } 
        ],
        ...
    }
```
 
    The value for the "metric_name" key is always a string and corresponds to the
    string key in metrics structure you are returning for each candidate.
 
    The value for the "maximize" key is a boolean.
    "true" means greater values of the metric are considered better,
    and "false" means lesser values of the metric are considered better.

    You can change the name of the fitness metric to anything metric you
    are already measuring in your evaluator. Let's say you had a fitness
    metric for "Area Under the Receiver Operating Characteristics" curve,
    also known as "auroc" and let's say you wanted to minimze that for your
    optimization.  You don't have to change your evaluation code, because
    you are already measuring these things.  Your new specification would be: 

```
    "blueprint_config": {
        ...

        "fitness": [
            {
                "metric_name": "auroc",
                "maximize": false
            } 
        ],
        ...
    }
```
 
### Putting it All Together
 
    Most folks in their own domains will likely want to leave the default
    values for these fitness metric keys and simply change the calculation
    for the value that goes into the 'fitness' key of the metrics dictionary
    that is returned from the domain evaluators evaluate_network() method.
    It's the least amount of work.
 
    Some folks will want to experiment with different fitness metrics
    to see which one works best.  The nice thing about this LEAF style of
    fitness specification is that for every candidate you evaluate, you can
    compute (and therefore record) all values for any kind of fitness
    you can think of and do apples-to-apples comparisons across different
    experiments, regardless of whether or not these experiments actually
    used the same fitness metric or not for their optimization.

### Multi-objective Fitness

    Currently ENN supports up to 2 fitness objectives to be optimized at the
    same time.  When more than one fitness objective is desired, a Pareto front
    selection policy is used.

    You might have noticed that the cannonical fitness specification above
    is an array.  To add fitness objectives, just add components to the array:

```
    "blueprint_config": {
        ...

        "fitness": [
            {
                "metric_name": "fitness",
                "maximize": true
            },
            {
                "metric_name": "auroc",
                "maximize": false
            } 
        ],
        ...
    }
```
