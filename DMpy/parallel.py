import sys

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import os
import time
import dill
import itertools


def RL_parallel_fit_func(model, outcomes_responses, fit_method='MLE', **kwargs):

    import copy
    import time
    import os
    import dill

    if not isinstance(outcomes_responses, list):
        outcomes_responses = [outcomes_responses]

    if isinstance(model, list):
        model = model[0]  # TODO need to sort this out properly

    with open(model, 'rb') as f:
        model = dill.load(f)

    fits = []

    if fit_method == 'MLE':
        for i in outcomes_responses:
            temp_model = copy.copy(model)
            temp_model.outcomes_responses = i
            temp_model.fit_MLE(i[0], i[1], **kwargs)
            temp_model.parameter_table['Outcomes_responses'] = str(temp_model.outcomes_responses)
            temp_model.parameter_table['Model'] = temp_model.name
            fits.append(temp_model)

    elif fit_method == 'MAP':
        for i in outcomes_responses:
            temp_model = copy.copy(model)
            temp_model.outcomes_responses = i
            temp_model.fit_MLE(i[0], i[1], **kwargs)
            temp_model.parameter_table['Outcomes_responses'] = str(temp_model.outcomes_responses)
            temp_model.parameter_table['Model'] = temp_model.name
            fits.append(temp_model)

    elif fit_method == 'variational':
        raise NotImplementedError

    elif fit_method == 'MCMC':
        for i in outcomes_responses:
            temp_model = copy.copy(model)
            temp_model.outcomes_responses = i
            temp_model.fit_NUTS(i[0], i[1], **kwargs)
            temp_model.parameter_table['Outcomes_responses'] = str(temp_model.outcomes_responses)
            temp_model.parameter_table['Model'] = temp_model.name
            fits.append(temp_model)

    else:
        raise ValueError("Fit method must be MLE, MAP, variational, or NUTS")

    datetime = time.strftime("%Y%m%d-%H%M%S")
    fits_fname = 'model_fits_{0}'.format(datetime)
    with open(fits_fname, 'wb') as f:
        dill.dump(fits, f)

    return os.path.abspath(fits_fname)


def RL_parallel_sim_func(model, outcomes, learning_parameters, observation_parameters,
                         n_subjects=1, *args, **kwargs):

    import sys
    sys.path.insert(0, '/home/k1327409/rl/')
    import copy
    import time
    import os
    import dill

    if not isinstance(outcomes, list):
        outcomes = [outcomes]

    if not isinstance(model, list):
        outcomes = [model]

    fits = []
    response_fnames = []

    for n, i in enumerate(outcomes):

        with open(model[n], 'rb') as f:
            model = dill.load(f)

        if len(i) > 2:
            i = (i, '')

        temp_model = copy.copy(model)
        datetime = time.strftime("%Y%m%d-%H%M%S")
        response_fname = '{0}_simulated_{1}.csv'.format(i[0], datetime)
        temp_model.simulate(i[0], n_subjects=n_subjects, response_file=response_fname,
                            learning_parameters=learning_parameters, observation_parameters=observation_parameters)
        fits.append(temp_model)
        response_fnames.append(os.path.abspath(response_fname))

    datetime = time.strftime("%Y%m%d-%H%M%S")
    fits_fname = 'model_fits_{0}'.format(datetime)
    with open(fits_fname, 'wb') as f:
        dill.dump(fits, f)

    return os.path.abspath(fits_fname), response_fnames


def combine_fits_func(fits, dir, sim=False, responses=None):

    import sys
    sys.path.insert(0, '/home/k1327409/rl/')
    import pandas as pd
    import dill
    import os
    import time

    unpickled_fits = []
    fname = ''

    for n, f in enumerate(fits):
        print "Loading fit chunk {0} of {1}".format(n+1, len(fits))
        with open(f, 'rb') as p:
            unpickled_fits.append(dill.load(p))

    print "Loaded fits"

    fits = sum(unpickled_fits, [])  # collapse list
    if sim:
        responses = sum(responses, [])

    datetime = time.strftime("%Y%m%d-%H%M%S")

    if not sim:
        out_df = pd.concat([f.parameter_table for f in fits])
        fname = os.path.join(dir, 'parameter_table_{0}.csv'.format(datetime))
        out_df.to_csv(fname)
        print "Written csv"

    fits_fname = os.path.join(dir, 'model_fits_{0}'.format(datetime))
    with open(fits_fname, 'wb') as f:
        dill.dump(fits, f)
    print "Saved fits"

    return fits_fname, fname


class ParallelFit(object):

    def __init__(self, models, outcomes_responses, out_dir, method='MLE', chunk_size=1, model_dir='',
                 all_combinations=False, **kwargs):
        self.models = models
        self.outcomes_responses = outcomes_responses
        self.method = method
        self.chunk_size = chunk_size
        self.model_dir = model_dir
        self.out_dir = out_dir
        self.all_combinations = all_combinations

        if self.method not in ['MLE', 'MAP', 'variational', 'MCMC']:
            raise ValueError("Method must be either MLE, MAP, variational or MCMC")

        self.pickled_models = []

        if not self.model_dir:
            datetime = time.strftime("%Y%m%d-%H%M%S")
            self.model_dir = os.path.join(self.out_dir, 'models_{0}'.format(datetime))

        os.makedirs(self.model_dir)

        # check for names and pickle using dill (pickle doesn't like nested functions)
        for i in self.models:
            if not len(i.name):
                raise AttributeError("Models should all have names for identification")
            model_path = os.path.join(self.model_dir, i.name)
            with open(model_path, 'wb') as f:
                dill.dump(i, f)
            self.pickled_models.append(model_path)

        if self.all_combinations:
            combinations = itertools.product(self.pickled_models, self.outcomes_responses)
            self.pickled_models_combined = []
            self.outcomes_responses_combined = []
            for m, o_r in combinations:
                self.pickled_models_combined.append(m)
                self.outcomes_responses_combined.append(o_r)
        else:
            self.pickled_models_combined = self.pickled_models
            self.outcomes_responses_combined = self.pickled_models

        if self.chunk_size > 1:
            self.outcomes_responses_combined = [self.outcomes_responses_combined[i:i + self.chunk_size] for i in
                                       xrange(0, len(self.outcomes_responses_combined), self.chunk_size)]
            self.pickled_models_combined = [self.pickled_models_combined[i:i + self.chunk_size] for i in
                                       xrange(0, len(self.pickled_models_combined), self.chunk_size)]


    def run(self, **kwargs):

        RL_parallel_fit = pe.MapNode(name='RL_parallel_fit',
                                 interface=util.Function(input_names=['model', 'outcomes_responses', 'fit_method', 'kwargs'],
                                                         output_names=['fits'],
                                                         function=RL_parallel_fit_func),
                                 iterfield=['model', 'outcomes_responses'])
        print "ITERFIELDS"
        print len(self.pickled_models_combined)
        print len(self.outcomes_responses_combined)
        RL_parallel_fit.inputs.model = self.pickled_models_combined
        RL_parallel_fit.inputs.outcomes_responses = self.outcomes_responses_combined
        RL_parallel_fit.inputs.fit_method = self.method
        RL_parallel_fit.inputs.kwargs = kwargs

        combine_fits = pe.Node(name='combine_fits',
                               interface=util.Function(input_names=['fits', 'dir'],
                                                       output_names=['fits', 'parameters'],
                                                       function=combine_fits_func))
        combine_fits.inputs.dir = self.out_dir

        wf = pe.Workflow(name="RL_modelling")
        wf.base_dir = self.out_dir
        # wf.config = {'execution': {'remove_node_directories': False}}

        wf.connect([(RL_parallel_fit, combine_fits, [('fits', 'fits')])])

        wf.run(**kwargs)


class ParallelSimulate(ParallelFit):

    def __init__(self, parameters, *args, **kwargs):
        super(ParallelSimulate, self).__init__(*args, **kwargs)

        self.parameters = parameters

        if self.all_combinations:
            combinations = itertools.product(self.pickled_models, self.outcomes_responses)
            self.pickled_models_combined = []
            self.outcomes_responses_combined = []
            self.parameters_combined = []
            for m, o_r, p in combinations:
                self.pickled_models_combined.append(m)
                self.outcomes_responses_combined.append(o_r)
                self.parameters_combined.append(p)
        else:
            self.pickled_models_combined = self.pickled_models
            self.outcomes_responses_combined = self.pickled_models
            self.parameters_combined = self.parameters


    def run(self, **kwargs):

        RL_parallel_sim = pe.MapNode(name='RL_parallel_sim',
                                     interface=util.Function(input_names=['model', 'outcomes', 'parameters'],
                                                             output_names=['fits', 'responses'],
                                                             function=RL_parallel_sim_func),
                                     iterfield=['model', 'outcomes_responses'])
        RL_parallel_sim.inputs.model = self.pickled_models_combined
        RL_parallel_sim.inputs.outcomes = self.outcomes_responses_combined
        RL_parallel_sim.inputs.fit_method = self.method

        combine_fits = pe.Node(name='combine_fits',
                               interface=util.Function(input_names=['fits', 'dir', 'sim'],
                                                       output_names=['fits', 'parameters'],
                                                       function=combine_fits_func))
        combine_fits.inputs.dir = self.out_dir
        combine_fits.inputs.sim = True

        wf = pe.Workflow(name="RL_modelling")
        wf.base_dir = self.out_dir
        # wf.config = {'execution': {'remove_node_directories': False}}

        wf.connect([(RL_parallel_sim, combine_fits, [('fits', 'fits')])])

        wf.run(**kwargs)

