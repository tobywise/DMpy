Simulation
""""""""""

Simulation is an important step to check that models produce expected patterns of behaviour. The process involves taking a series of task stimuli and feeding them into a learning model and observation model with a given set of parameter values to produce behaviour predicted by these particular parameter values.

To facilitate this process, the ``RLModel`` class has a ``simulate`` method, that provides the ability to simulate behaviour for a given model across parameter values, and produces outputs necessary to evaluate the simulated behaviour.


Parameter recovery
""""""""""""""""""

Parameter recovery tests the ability to determine parameter values from observed behavioural data simulated with known parameter values. For example, if we simulated behaviour using a Rescorla-Wagner learning model with an alpha value of 3, our model fitting procedures should return an alpha value of 3 when we attempt to fit a Rescorla-Wagner model to this data. DMpy provides a couple of functions to assist with this process. Firstly, when data is simulated using the ``RLModel`` class ``simulate`` method and saved to a csv file, the generated response file will include columns containing the values of each parameter used to simulate the data.

If a response file including these columns is fed into any of the ``RLModel`` class's fit methods, it will automatically include these values in its output table for each subject and estimate Pearson correlations between the true and recovered parameter values (saved in the ``.recovery_correlations`` attribute of the model), along with producing plots to illustrate these.

In order for this to work properly, it's advisable to simulate across a range of parameter values. This is simple to do by providing lists of values in the parameter dictionary rather than single values. If multiple values are provided for more than one parameter, all possible combinations of these parameter values will be simulated. Be aware that if many parameter values are provided, this can rapidly lead to a *very* large number of simulated datasets and cause problems!

This is an example of parameter recovery with a Rescorla-Wagner model, where we're assessing our ability to recover the alpha parameter. To simulate across a range of values of alpha, we provide a range from 0.1 to 0.9 in steps of 0.1 using ``np.arange(0.1, 1, 0.1)``.

.. code-block:: python

        >>> sim_rw = model_rw.simulate(outcomes, n_subjects=50,
                                       response_file='parameter_recovery.csv',
                                       learning_parameters={'value': 0.5, 'alpha0': np.arange(0.1, 1, 0.1)},
                                       observation_parameters={'b':3})
        Finished simulating
        Saving simulated responses to parameter_recovery.csv

The simulation plots also now plot estimated probabilities and other values across the range of parameter values provided.

.. image:: rw_sim.png
        :align: center

If we now fit our model to this data, we can see whether the alpha parameter is recovered successfully.

.. code-block:: python

        >>> model_rw.fit_MAP(outcomes, sim_rw)
        Loading data
        Loading multi-subject data with 450 subjects
        Loaded data, 450 subjects with 120 trials

        -------------------Finding MAP estimate-------------------

        Performing model fitting for 450 subjects

        Optimization terminated successfully.
         Current function value: 21874.952537
         Iterations: 45
         Function evaluations: 72
         Gradient evaluations: 72

         Performing parameter recovery tests...
                   alpha0                   Subject  alpha0_sim  value_sim
            0    0.172732    alpha0.0.1.value.0.5_0         0.1        0.5
            1    0.129099    alpha0.0.1.value.0.5_1         0.1        0.5
            2    0.146754   alpha0.0.1.value.0.5_10         0.1        0.5
            3    0.111058   alpha0.0.1.value.0.5_11         0.1        0.5
            4    0.127479   alpha0.0.1.value.0.5_12         0.1        0.5

        Finished model fitting in 30.8701867692 seconds

The parameter table has our simulated values in addition to the estimated values for each subject, and these are saved in the model's ``.parameter_table`` attribute.

Additionally, the fitting method produced two figures: a scatter plot showing the relationship between the true and estimated alpha values, and a correlation matrix showing the correlation between every estimated parameter in the model (in this case there is only a single value so it's a pretty uninteresting matrix).

.. image:: rw_pr1.png
        :align: center

.. image:: rw_pr2.png
        :align: center

To illustrate this more clearly, let's look at an example of a more complex model for which parameters aren't recovered so accurately...

.. code-block:: python

        >>> model_1lr.fit_MAP(outcomes, complex_model)

        Finished model fitting in 61.4955701763 seconds

.. image:: complex_pr1.png
        :align: center

.. image:: complex_pr2.png
        :align: center

We can see from the plots that it doesn't look good. The a parameter is estimated successfully, as shown by the scatter plot and a correlation of .96 between the true and estimated values in the correlation matrix. However, the other two parameters show poor correlations between true and estimated values, indicating that we're not able to recover them successfully.

