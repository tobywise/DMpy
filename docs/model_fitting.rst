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