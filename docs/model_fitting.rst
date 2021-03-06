Model fitting
"""""""""""""

Parameter initialisation
------------------------

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
--------------------

The next step is to tell DMpy which learning and observation models. This can be done using the ``DMModel`` class:

.. code-block:: python

    from DMpy.learning import rescorla-wagner

    rw_model = DMModel(rescorla_wagner, [value, alpha_static], softmax, [b])

The first argument given to DMModel is the learning model function. This can either be imported from the included learning models or defined separately (see the model specification documentation). The next argument indicates the parameters to be provided to the learning model (as defined earlier). The observation model is then specified (again this should be a python function) followed by its parameters. If no observation model is being used, the last two arguments can be set to ``None``.


Model fitting
-------------

DMpy provides a range of methods for model fitting, based on the functions provided by PyMC3.

Once a model is defined using the ``DMModel`` class, fitting is performed by simply calling the model's ``.fit()`` method and supplying trial outcomes (e.g. rewards), subjects' responses (the options chosen by subjects on each trial), and a fitting method. For example:

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

The ``fit()`` method takes additional arguments that can be used to alter how the model fitting is performed. The ``fit_kwargs`` argument takes a dictionary of argument names and values to be supplied to the underlying `PyMC3 variational fitting function used for variational inference`_ or the `PyMC3 MAP fitting function for MAP and MLE estimation`_ while the ``sample_kwargs`` argument takes a dictionary of keyword arguments to be supplied to `PyMC3's sampling function for MCMC sampling`_.

.. _Variational fitting: http://docs.pymc.io/api/inference.html#pymc3.variational.inference.fit
.. _MAP and MLE fitting: http://docs.pymc.io/notebooks/getting_started.html#Maximum-a-posteriori-methods
.. _MCMC: http://docs.pymc.io/api/inference.html#pymc3.sampling.sample


Specific subjects can be easily excluded from fitting by supplying their ID (as given in the response file) to the ``exclude`` argument. Multiple subjects can be excluded by providing a list of subject IDs.

Additional model inputs (such as outcomes for other stimuli) can be provided by including them as columns in the response file provided to the fitting method. The ``model_inputs`` argument can then be used to specify which columns will be used. This should be provided as a list, even if only one column is provided. For example:

.. code-block:: python

        model.fit(response_file, fit_method='MAP', model_inputs=['stimulus_B_outcomes', 'stimulus_C_outcomes'])


Hierarchical model fitting
--------------------------

DMpy is able to fit models in a hierarchical manner (i.e. where a group-level prior is provided to constrain individual-level parameter estimates). This can be achieved by setting the ``hierarchical`` argument of the ``fit()`` method to ``True`` (note that this will only work when using variational inference or MCMC as the estimation method). If this option is chosen, the distribution provided when initialising the parameters will be used for the group-level prior rather than the individual-level priors. For example, if a parameter is given a normal distribution with a mean of 0.5 and a variance of 0.1, the prior means of the individual-level parameter estimates will be drawn from a distribution with a mean of 0.5 and variance of 0.1. The standard distributions of the individual-level parameters are set to 1, but I need to find a way of easily allowing the user to modify these too...


Fitting output
--------------

Once the model has been fit, a table of estimated parameter values will be provided. This will give estimates of each free parameter in the model for each subject. If variational or MCMC was chosen as the fitting method, the mean and standard deviation of the posterior distribution will be given. This table is stored in the ``parameter_table`` attribute of the model instance.

If either variational or MCMC fitting methods are used and the ``plot`` argument is set to true, DMpy will also produce trace plots to illustrate the posterior distributions of each estimated parameter.


Multiple runs per subject
-------------------------

It is possible to fit models to data where each subject has multiple runs on the task by adding an additional column to the response data file with the name "Run". Each run should be given a number, starting from zero, and each subject should have the same number of runs. When the model is fit, this will result in the same parameter values being used across every run for each subject. For example, we might assume that a subject will use the same learning rate across every run of a task, and this allows our estimated learning rate parameter to be consistent across runs. If we had reason to think that the value of this parameter would vary across runs, this could be achieved by inputting each run as a different "subject", which would then allow for different parameter estimates for each run. To summarise - parameter estimates are equal across runs, but differ across subjects.

It is also possible to use the "Run" column to fit models for tasks where behaviour in response to multiple stimuli is measured. For example, in a task featuring two stimuli with independent value it may be desirable to fit a separate model for each stimulus, assuming the subject is using the same e.g. learning rate across stimuli. Where there is no interdependence between the value of stimulus A and stimulus B, it is possible to fit this model to the two stimuli by specifying them as two separate "runs".


Parameter starting values
-------------------------

Some quantities used by the model will need to have a starting value provided (for example the estimated value of a stimulus), even though this is not necessarily of interest. This can be achieved by simply setting the parameter to the desired starting value using the mean argument of the Parameter class. An advantage of this is that it is also possible to estimate the best fitting starting value by providing a distribution rather than a fixed point as the starting value.


Debugging errors during model fitting
-------------------------------------

Errors during model fitting are sometimes encountered when the model produces illegal values (e.g. ``NaN``) during fitting. This may be because of an error in the code for the model function, which can be checked using the ``utils.model_check`` function (see model specification documentation).

Alternatively, it may be that the model produces invalid values when its parameters are within certain ranges. This will typically lead to an optimization error occurring, e.g.

.. code-block:: python

        >>> model.fit(oo, b, 'variational')

        Loading data
        Loading multi-subject data with 180 subjects
        Loaded data, 180 subjects with 120 trials

        -------------------Fitting model using ADVI-------------------

        Performing hierarchical model fitting for 180 subjects

          0%|          | 0/100 [00:00<?, ?it/s]
        Average Loss = 12,100:   3%|▎         | 3/100 [00:00<00:04, 22.39it/s]
        Traceback (most recent call last):
          File "Anaconda\lib\site-packages\IPython\core\interactiveshell.py", line 2882, in run_code
            exec(code_obj, self.user_global_ns, self.user_ns)
          File "<ipython-input-39-7415b86a0365>", line 30, in <module>
            model.fit(oo, b, 'variational', fit_kwargs={'n':100}, sample_kwargs={'draws': 100}, hierarchical=True)
          File "DMpy\model.py", line 306, in fit
            fit_stats=fit_stats, fit_kwargs=fit_kwargs, sample_kwargs=sample_kwargs)
          File "DMpy\model.py", line 421, in _fit_variational
            approx = fit(model=rl, **fit_kwargs)
          File "pymc3\variational\inference.py", line 885, in fit
            return inference.fit(n, **kwargs)
          File "pymc3\variational\inference.py", line 131, in fit
            self._iterate_with_loss(n, step_func, progress, callbacks)
          File "pymc3\variational\inference.py", line 172, in _iterate_with_loss
            raise FloatingPointError('NaN occurred in optimization.')
        FloatingPointError: NaN occurred in optimization.


The best way to avoid such errors is to ensure that priors over free parameters are specified properly. For example, if a parameter value of less than zero would lead to the model producing strange results, it is best to set a lower bound of zero for the parameter, using ``lower_bound=0`` when defining the parameter using the ``Parameter`` class.

It can also be useful to test whether certain values will lead to errors, rather than finding this out some way through fitting. For this purpose it is possible to pass a test value to the parameter instance using the ``testval`` argument (this is a keyword argument that gets passed to the underlying PyMC3 distribution class, for more information see the PyMC3 documentation). This will begin fitting at this parameter value, allowing errors to be detected at the start of fitting. For example, if you suspected that negative values of an ``alpha`` parameter was causing fitting to fail, you may wish to initialise the parameter as follows, so that model fitting will begin at a negative ``alpha`` value and make it clear whether this is indeed the case.

.. code-block:: python

        alpha = Parameter('alpha', 'uniform', lower_bound=0, upper_bound=1, testval=-1)