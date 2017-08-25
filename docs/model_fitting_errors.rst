Debugging errors during model fitting
"""""""""""""""""""""""""""""""""""""

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

It can also be useful to test whether certain values will lead to errors, rather than finding this out some way through fitting. For this purpose it is possible to pass a test value to the parameter instance using the ``testval`` argument (this is a keyword argument that gets passed to the underlying PyMC3 distribution class, for more information see the PyMC3 documentation). This will begin fitting at this parameter value, allowing errors to be detected at the start of fitting.