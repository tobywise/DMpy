Parameter initialisation
""""""""""""""""""""""""

The first step in the model fitting process is parameter initialisation, which simply involves telling DMpy the parameters of your model and providing values for fixed parameters and priors (or not) for free parameters. This is achieved using the ``Parameter`` class and is the same for learning and observation model parameters.

Each parameter should be defined as an instance of the ``Parameter`` class, for example:

.. code-block:: python

        alpha = Parameter('alpha', 'normal', lower_bound=0, upper_bound=1, mean=0.5, variance=0.1)

Here we've defined the parameter alpha; the first argument is the name of the parameter (this will be used to identify parameters in the model fitting output later) and the second argument specifies the prior distribution of the parameter. Currently DMpy recognises ``normal``, ``uniform`` and ``flat`` distributions.

The later arguments specify bounds (used if the estimated value of the parameter should be limited to a certain range), while the ``mean`` and ``variance`` arguments are used to provide information about the prior distribution. Means and variances are only necessary for distributions that require these parameters (such as a normal distribution) and can be ignored for others. Additionally, distribution information is ignored by MLE estimation; any parameters without a uniform or flat distribution will be converted to one of these (depending on the bounds specified) before estimation if MLE is chosen as the estimation method.

It is also simple to fix parameter values (i.e. set them to a desired value rather than estimating their value) by changing the ``distribution`` argument to ``'fixed'``. The ``mean`` argument can then be used to indicate the desired value for the parameter to take. For example:

.. code-block:: python

        beta = Parameter('beta', 'fixed', mean=3)


Model initialisation
""""""""""""""""""""

The next step is to tell DMpy which learning and observation models. This can be done using the ``RLModel`` class:

.. code-block:: python

    from DMpy.learning import rescorla-wagner

    rw_model = RLModel(rescorla_wagner, [value, alpha_static], softmax, [b])

The first argument given to RLModel is the learning model function. This can either be imported from the included learning models or defined separately (see the model specification documentation). The next argument indicates the parameters to be provided to the learning model (as defined earlier). The observation model is then specified (again this should be a python function) followed by its parameters. If no observation model is being used, the last two arguments can be set to ``None``.


Model fitting
"""""""""""""

DMpy provides a range of methods for model fitting, based on the functions provided by PyMC3.

Once a model is defined using the ``RLModel`` class, fitting is performed by simply calling the model's ``.fit()`` method and supplying trial outcomes (e.g. rewards), subjects' responses (the options chosen by subjects on each trial), and a fitting method. For example:

.. code-block:: python

        model.fit(data, fit_method='MAP')

Data should be provided in a .txt or .csv file with the columns named "Outcome", indicating the outcome of the trial, "Response", representing the subjects' response, and "Subject", representing the subject ID. All subjects should be provided in the same response file. It is possible for different subjects to have different outcome series, as long as they are the same length. Missing responses should be coded as NaN, and these will be ignored by the model fitting procedures.

The fitting method can be one of four options:

* Maximum likelihood estimation (``'MLE'`` or ``'mle'``)
* Maximum a posteriori (``'MAP'`` or ``'map'``)
* Variational (``'variational'``)
* Monte-Carlo Markov Chain (``'MCMC'``)

Detailed descriptions of these methods are beyond the scope of this documentation, and are described with far more detail and competence in other sections of the internet. In brief, MLE and MAP methods provide point estimates of parameter values using relatively simple methods. The main difference between these two is that MAP estimation takes into account priors on parameters (e.g. mean and variance) while MLE does not (except for simple upper/lower bounds on parameter values).

Variational inference and MCMC sampling provide additional information about parameter distributions, and can provide more accurate parameter estimates. However, these methods can be slow, particularly MCMC.

The fitting process aims to find parameter values that minimise the a likelihood function. Two options are provided here, either the log likelihood (for binary responses) or the R:sup:2 value (for continuous responses). The log-likelihood is calculated as follows:

.. math::

   \sum_{t=1}^{T} log(P(A_{t}|C_{t}))

Where :math:`A` is the action action predicted by the model on trial :math:`t` and :math:`C` is the subject's choice on that trial. It is important to note that DMpy minimiseds this function for all subjects together, rather than independently for each subject. By default, the log-likelihood is calculated, but this can be changed using the ``logp_method`` argument of the ``fit()`` method.

..note::The first time the fitting procedure is run it can be quite slow due to Theano having to set up various things.

Hierarchical model fitting
""""""""""""""""""""""""""

DMpy is able to fit models in a hierarchical manner (i.e. where a group-level prior is provided to constrain individual-level parameter estimates). This can be achieved by setting the ``hierarchical`` argument of the ``fit()`` method to ``True`` (note that this will only work when using variational inference or MCMC as the estimation method). If this option is chosen, the distribution provided when initialising the parameters will be used for the group-level prior rather than the individual-level priors. For example, if a parameter is given a normal distribution with a mean of 0.5 and a variance of 0.1, the prior means of the individual-level parameter estimates will be drawn from a distribution with a mean of 0.5 and variance of 0.1. The standard distributions of the individual-level parameters are set to 1, but I need to find a way of easily allowing the user to modify these too...


Fitting output
""""""""""""""

Once the model has been fit, a table of estimated parameter values will be provided. This will give estimates of each free parameter in the model for each subject. If variational or MCMC was chosen as the fitting method, the mean and standard deviation of the posterior distribution will be given. This table is stored in the ``parameter_table`` attribute of the model instance.


Multiple runs per subject
"""""""""""""""""""""""""

It is possible to fit models to data where each subject has multiple runs on the task by adding an additional column to the response data file with the name "Run". Each run should be given a number, starting from zero, and each subject should have the same number of runs. When the model is fit, this will result in the same parameter values being used across every run for each subject. For example, we might assume that a subject will use the same learning rate across every run of a task, and this allows our estimated learning rate parameter to be consistent across runs. If we had reason to think that the value of this parameter would vary across runs, this could be achieved by inputting each run as a different "subject", which would then allow for different parameter estimates for each run. To summarise - parameter estimates are equal across runs, but differ across subjects.

It is also possible to use the "Run" column to fit models for tasks where behaviour in response to multiple stimuli is measured. For example, in a task featuring two stimuli with independent value it may be desirable to fit a separate model for each stimulus, assuming the subject is using the same e.g. learning rate across stimuli. Where there is no interdependence between the value of stimulus A and stimulus B, it is possible to fit this model to the two stimuli by specifying them as two separate "runs".


Parameter starting values
"""""""""""""""""""""""""

Some quantities used by the model will need to have a starting value provided (for example the estimated value of a stimulus), even though this is not necessarily of interest. This can be acheived by simply setting the parameter to the desired starting value using the mean argument of the Parameter class. An advantage of this is that it is also possible to estimate the best fitting starting value by providing a distribution rather than a fixed point as the starting value.


