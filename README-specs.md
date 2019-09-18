
## Evolvable Parameter JSON Specs

In various places throughout the config files it is possible to specify a JSON
string that describes schema for structures that will be present in some
portion of the evolved output.

In principle, you should be able to describe any kind of data you want from
these config strings/files. While we require that this is specified in the
JSON format described here, there are really very few restrictions on what
you can do within that JSON.

There are two places where these specifications come into play.
1. in the specification of global hyperparameters
2. in the specification of layer parameters to be used within the network.

It is easiest to explain how this works by example.

### Simple Hyperparameters JSON

The purpose of the JSON string/file is to specify the fields and types that
the ENN Service should be evolving.

Say you want the dictionary of evolved global hyperparameters to look like
this:
~~~~
{
    "learning_rate": 0.0005,
    "optimizer": "SGD"
}
~~~~

A corresponding JSON spec would look like this:
~~~~
{
    "type": {
        "learning_rate": { 
            "type": "Double",
            "lowerBound": 1e-4,
            "upperBound": 1e-3,
            "scale": "log"
        },
        "optimizer": { 
            "type": "String",
            "choice": [ "Adam", "SGD" ]
        }
    }
}
~~~~

Each JSON spec has a top-level **_type _** entry which tells the system to
expect a type definition of an object/structure.  The top-level JSON object
sent to the ENN Service must always be a structure, so it always has a type
entry.

Each field specification has as its key the name of the field as you would
like to see it returned to you if the field were in a dictionary.
The value associated with the field name key is itself another JSON object
which contains the **_type_** definition of the field.

Given the example, you can see that how you specify the type can be a little
shifty. We run down the examples here:

### Scalar Types

As illustrated in the example, the value of type can be a simple String.
When this is the case, this implies that the field should be a simple scalar
value. We currently support Double, Float, Integer, Boolean, and String scalar
data types in the structure for valid values of the **_type_** field.

#### Boolean Types

Boolean types need no further specification other than their type information.

#### Numeric Types

Double and Float are the same thing, since we are using python here.

##### Ranges

For Integer, Double and Float types, each entry can express a range of
possibilities for an independent variable via the "lowerBound" and "upperBound"
spec fields.

Scales can be specified as "linear" (default) or "log".

##### Discrete Choices

Numeric types also offer the possibility of picking from a number of
discrete choices with the "choice" specification. The value for this field
is an array of the available choices to pick uniformly from.

#### String Types

You can thing of String types as being enumerated types.
Strings only support the "choice" specification.


### Nested Structures

It is also possible to specify nested structures of data.  Say you wanted
to express that your dictionary looks like this

~~~~
{
    "optimizer": "SGD",
    "opt_params": {
        "learning_rate": 0.0005
    }
}
~~~~

To express this, the value of the **_type_** field in the JSON spec can be
another JSON structure with more structure definition instead of a string
describing a scalar type.  

And here is an example of the JSON spec for the nested structure above:

~~~~
{
    "type": {
        "optimizer": { 
            "type": "String",
            "choice": [ "Adam", "SGD" ]
        },
        "opt_params": { 
            "type": {
                "learning_rate": {
                    "type" : "Double",
                    "lowerBound": 1e-4,
                    "upperBound": 1e-3,
                    "scale": "log"
            }
        }
    }
}
~~~~

## Layer Parameters JSON File

The Layer Parameters JSON file allows you to set the boundaries for the evolved
parameters for each layer type.  The file format itself is based on the 
same format used for evolved hyperparameters (see above).

### Example layer_parameters JSON spec

~~~~
{
    'type': {
        'Conv2D' : {
            'type' : {
                'filters' : {
                    'type' : 'Integer',
                    'lowerBound': 16,
                    'upperBound' : 96,
                    'scale': 'log'
                },
                'kernel_size' : {
                    'type' : 'Integer',
                    'choice' : [ 1, 3 ]
                },
                'activation' : {
                    'type' : 'String',
                    'choice': [ 'relu', 'linear', 'elu', 'selu' ]
                },
                'kernel_initializer' : {
                    'type' : 'String',
                    'choice': [ 'glorot_normal', 'he_normal',
                                'glorot_uniform', 'he_uniform' ]
                },
                'kernel_regularizer' : {
                    'type' : {
                        'regularizer' : {
                            'type' : 'String',
                            'choice': [ 'l2' ]
                        },
                        'penalty' : {
                            'type' : 'Double',
                            'lowerBound': 1e-9,
                            'upperBound' : 1e-3,
                            'scale': 'log'
                        }
                    }
                },
                'padding' : {
                    'type' : 'String',
                    'choice': [ 'same' ]
                }
            }
        },
        'Dropout' : {
            'type' : {
                'rate' : {
                    'type' : 'Double',
                    'lowerBound': 0.0,
                    'upperBound' : 0.7,
                    'scale': 'linear'
                }
            }
        }
    }
}
~~~~

You can think of this file as the definition of nested structures, as described
in the section above.

There is an outer structure definition for the entire definition which defines
one field per layer type that can be picked at random.  The field names for
each layer must match exactly those that are defined in the 
[Keras 2.2.4 documentation](https://keras.io/layers/about-keras-layers/)
for layer types you are going to use. 

For each layer type, there is an inner structure definition for the layer
parameters. Each field name and type is particular to the layer, however
you can set the boundaries for how the value of that field is evolved.
Be careful as the names for the fields must match exactly to what is outlined
in the Keras documentation for the layer. Any fields that are not specified
use the Keras default values.  For values which are to have a single
non-default value, prefer the "choice" description with a single option.
And, no, if you add extra fields here, the python intepreter will not
magically take them up and interpret them.


### Supported Layers and Regularizer Types

We support definitions for all documented fields from the following layer types:

    * BatchNormalization
    * Conv1D
    * Conv2D
    * Dense
    * Dropout
    * GlobalAveragePooling2D
    * GRU
    * LSTM
    * MaxPooling1D
    * MaxPooling2D
    * SpatialDropout1D
    * SpatialDropout2D
    * SpatialDropout3D
    * UpSampling1D
    * UpSampling2D
    * ZeroPadding1D
    * ZeroPadding2D

We support definitions for the following regularizers:

    * l1
    * l2
    * l1_l2

If you let us know which layers you need that are not listed here, we can
add support for them.
