Model specification
"""""""""""""""""""

DMpy makes it simple to design and create models in Python code. Each model includes a separate learning and observation
model. The learning model (e.g. a Rescorla-Wagner model) learns the value of stimuli, and the observation model
(e.g. a softmax function) translates the estimated value into behaviour.

Both learning and observation models are defined as Python functions, however the requirements of these functions differ
between the two types of model.

Learning models
---------------

Learning model functions should take both dynamic and static parameters, defined as follows:

* Dynamic parameters: Inputs whose values vary during the task. For example, one such input could be the estimated value of a stimulus, which is updated at each step. For each step, the learning model function would take the previous value as an input.

* Static parameters: Parameters which take fixed values at all time steps. For example, a Rescorla-Wagner model has a fixed learning rate across all trials.

When specifying models, the function should take arguments in the following order:
trial outcome, dynamic parameters, static parameters

The function must return any values that are entered into the observation model, such as the estimated value of a stimulus
at the current time step, and any other estimated values that are re-entered into the model at the next step. It is
also possible to return any other values from the function, which will be givenas outputs when simulating models.
For example, it might be desirable to return trial-by-trial prediction errors to combine with neuroimaging.

The function must return a tuple of the form:
(any value entered into the observation model, re-entered value 1, re-entered value 2,... other value 1, other value 2...)

The code in the model itself should calculate the value for the current trial, based on whatever inputs are provided. This
code is then executed at each step of the task.

For example, the following code implements a simple Rescorla-Wagner model, with value at each step updated according to
a prediction error, weighted by a learning rate.

.. code-block:: python

        def rescorla_wagner(o, v, alpha):

            pe = o - v
            value = v + alpha * pe

            return (value, pe)

Here the function takes the output on the current trial, the value from the previous trial, and the static alpha parameter.
The prediction error is assigned to a variable within the function, which then allows this value to be returned from the function
and used, along with the value.

Model code can generally use standard Python/Numpy functions, however as
DMpy relies on PyMC3 for model estimation, which in turn relies on Theano for underlying calculations, some aspects of
the learning function may need to use Theano functions. A couple of notable particular examples of this are if/else statements and equality tests; standard
python if/else statements (``if``, ``else``, ``elif``) and equality tests (``==``, ``!=``, ``>=``, ``<=``) will not work and it is necessary to use a Theano equivalent, such as switch and T.eq. This is a fairly simple exercise, for example:

.. code-block:: python

        import theano.tensor as T

        value = T.switch(T.eq(v, o), 1, 0)

In place of:

.. code-block:: python

        if v == o:
            value = 1
        else:
            value = 0

Although translating from python/numpy to Theano is a little awkward, the theano documentation is clear and it's typically not difficult to find the desired function.

It currently isn't possible to refer to values prior to t-1.


Observation models
------------------

Observation model functions take a similar form to learning model functions. They should take as arguments any values that are provided by the learning model, followed by any static parameters.

They should return the probability of choosing a given action, along with any other values (such as intermediate variables calculated by the function) in a tuple of the form: (probability, [other variable1, other variable2...]).

Note that this is slightly different to the learning model return specification; it's currently not possible to re-enter values calculated in the observation model into the model at the next step - this is because observtation models (for now) are not treated as trial-by-trial models that are updated one step at a time, and are instead evaluated across an entire value array produced by a learning model. Additionally, outputs other than the value have to be returned as a list, rather than simply as other values in the tuple. This inconsistency across learning and observation models will probably get fixed at some point in the future.

For example, this is a slightly complex softmax model that takes as inputs the value estimated by the learning model (``v``), a second variable estimated by the learning model (``c``), and two static parameters (``b`` and ``m``). It returns the estimated probability of choosing an action, along with a variable calculated by the model (in this case, the modulated inverse temperature, ``b_m``).

.. code-block:: python

        def softmax_c(v, c, b, m):
            b_m = b / (1 - m * c)
            return ((b_m * v).exp() / ((b_m * v).exp() + (b_m * (1 - v)).exp()), [b_m])


Debugging
---------

When coding a model, it is common to encounter problems in the update step, particularly with complex models. To make it simpler to discover problems, a utility function (``model_check``) is provided that takes a dictionary of parameter values and prints the output of each step in the update function, allowing the location of errors in code.

For example:

.. code-block:: python

        >>> from DMpy.utils import model_check
        >>>from DMpy.learning import rescorla_wagner

        >>> model_check(rescorla_wagner, {'o': 1, 'v':0.5, 'alpha':0.3})

        o 1
        v 0.5
        alpha 0.3
        Running code:
        import theano.tensor as T
        Running code:
        import numpy as np
        Running code:
        alpha=0.3
        0.3

        Running code:
        v=0.5
        0.5

        Running code:
        o=1
        1

        Running code:
        pe=o-v
        0.5

        Running code:
        value=v+alpha*pe
        0.65

        RETURNS
        value
        0.65
        pe
        0.5

