Model fitting
"""""""""""""

DMpy provides a range of methods for model fitting, based on the functions provided by PyMC3.

Once a model is defined using the ``RLModel`` class, fitting is performed by simply calling the model's ``.fit()`` method and supplying trial outcomes (e.g. rewards), subjects' responses (the options chosen by subjects on each trial), and a fitting method. For example:

.. code-block:: python

        model.fit(outcomes, responses, 'MAP')

Outcomes should be provided as a series of zeros and ones.

Responses should be provided in a .txt or .csv file with the columns named "Response", representing the subjects' response, and "Subject", representing the subject ID. All subjects should be provided in the same response file.

The fitting method can be one of four options:

* Maximum likelihood estimation (``'MLE'`` or ``'mle'``)
* Maximum a posteriori (``'MAP'`` or ``'map'``)
* Variational (``'Variational'``)
* Monte-Carlo Markov Chain (``'MCMC'``)

Detailed descriptions of these methods are beyond the scope of this documentation, and are decribed with far more detail and competence in other sections of the internet. In brief, MLE and MAP methods provide point estimates of parameter values using relatively simple methods. The main difference between these two is that MAP estimation takes into account priors on parameters (e.g. mean and variance) while MLE does not (except for simple upper/lower bounds on parameter values).

Variational inference and MCMC sampling provide additional information about parameter distributions, and can provide more accurate parameter estimates. However, these methods can be slow, particularly MCMC.

The fitting process aims to find parameter values that minimise the log-likelihood of the data given the model, according to the following equation:

.. math::

   \sum_{t=1}^{T} log(P(A_{t}|C_{t}))

Where :math:`A` is the action action predicted by the model on trial :math:`t` and :math:`C` is the subject's choice on that trial. It is important to note that DMpy estimates the log-likelihood for all subjects together, rather than independently for each subject.


Multiple runs per subject
"""""""""""""""""""""""""

It is possible to fit models to data where each subject has multiple runs on the task by adding an additional column to the response data file with the name "Run". Each run should be given a number, starting from zero, and each subject should have the same number of runs. When the model is fit, this will result in the same parameter values being used across every run for each subject. For example, we might assume that a subject will use the same learning rate across every run of a task, and this allows our estimated learning rate parameter to be consistent across runs. If we had reason to think that the value of this parameter would vary across runs, this could be achieved by inputting each run as a different "subject", which would then allow for different parameter estimates for each run. To summarise - parameter estimates are equal across runs, but differ across subjects.

It is also possible to use the "Run" column to fit models for tasks where behaviour in response to multiple stimuli is measured. For example, in a task featuring two stimuli with independent value it may be desirable to fit a separate model for each stimulus, assuming the subject is using the same e.g. learning rate across stimuli. Where there is no interdependence between the value of stimulus A and stimulus B, it is possible to fit this model to the two stimuli by specifying them as two separate "runs".


Multiple outcome series
"""""""""""""""""""""""

Outcomes should be provided in an array of shape (trials, subjects).

The first time fitting is run it can be quite slow due to Theano having to set up various things.


Parameter starting values
"""""""""""""""""""""""""

Some quantities used by the model will need to have a starting value provided (for example the estimated value of a stimulus).


Parameter value inheritance
"""""""""""""""""""""""""""

It is possible for parameters to inherit their value from other parameters. For example, in Sutton's K1 (1992) model, the starting value of the beta parameter is defined as log(Rhat), where Rhat is another parameter in the model. It is possible to specify this in DMpy