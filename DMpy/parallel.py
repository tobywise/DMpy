import sys

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

sys.path.insert(0, '/home/k1327409/rl/')
import os
import time
import dill
import itertools


def RL_parallel_fit_func(model, outcomes_responses, fit_method='MLE', **kwargs):

    import sys
    sys.path.insert(0, '/home/k1327409/rl/')
    import copy
    import time
    import os
    import dill

    if not isinstance(outcomes_responses, list):
        outcomes_responses = [outcomes_responses]

    with open(model, 'rb') as f:
        model = dill.load(f)

    fits = []

    if fit_method == 'MLE':
        for i in outcomes_responses:
            temp_model = copy.copy(model)
            temp_model.outcomes_responses = outcomes_responses
            temp_model.fit_MLE(i[0], i[1], **kwargs)
            temp_model.parameter_table['Outcomes_responses'] = str(temp_model.outcomes_responses)
            temp_model.parameter_table['Model'] = temp_model.name
            fits.append(temp_model)

    elif fit_method == 'MAP':
        for i in outcomes_responses:
            temp_model = copy.copy(model)
            temp_model.outcomes_responses = outcomes_responses
            temp_model.fit_MLE(i[0], i[1], **kwargs)
            temp_model.parameter_table['Outcomes_responses'] = temp_model.outcomes_responses
            temp_model.parameter_table['Model'] = temp_model.name
            fits.append(temp_model)

    elif fit_method == 'variational':
        raise NotImplementedError

    elif fit_method == 'NUTS':
        for i in outcomes_responses:
            temp_model = copy.copy(model)
            temp_model.outcomes_responses = outcomes_responses
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


def RL_parallel_sim_func(model, outcomes, parameters, *args, **kwargs):  # sim choices is important here

    import sys
    sys.path.insert(0, '/home/k1327409/rl/')
    import copy
    import time
    import os
    import dill

    if not isinstance(outcomes, list):
        outcomes_responses = [outcomes]

    with open(model, 'rb') as f:
        model = dill.load(f)

    fits = []

    for i in outcomes_responses:
        temp_model = copy.copy(model)
        temp_model.outcomes_responses = outcomes_responses
        temp_model.fit_MLE(i[0], i[1])
        temp_model.parameter_table['Outcomes_responses'] = str(temp_model.outcomes_responses)
        temp_model.parameter_table['Model'] = temp_model.name
        fits.append(temp_model)

    datetime = time.strftime("%Y%m%d-%H%M%S")
    fits_fname = 'model_fits_{0}'.format(datetime)
    with open(fits_fname, 'wb') as f:
        dill.dump(fits, f)

    return os.path.abspath(fits_fname)


def combine_fits_func(fits, dir):

    import sys
    sys.path.insert(0, '/home/k1327409/rl/')
    import pandas as pd
    import dill
    import os
    import time

    unpickled_fits = []

    for f in fits:
        with open(f, 'rb') as p:
            unpickled_fits.append(dill.load(p))

    fits = sum(unpickled_fits, [])  # collapse list

    datetime = time.strftime("%Y%m%d-%H%M%S")
    fits_fname = os.path.join(dir, 'model_fits_{0}'.format(datetime))
    with open(fits_fname, 'wb') as f:
        dill.dump(fits, f)

    out_df = pd.concat([f.parameter_table for f in fits])
    fname = os.path.join(dir, 'parameter_table_{0}.csv'.format(datetime))
    out_df.to_csv(fname)

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

        if self.method not in ['MLE', 'MAP', 'variational', 'NUTS']:
            raise ValueError("Method must be either MLE, MAP, variational or NUTS")

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

        # break responses/outcomes into chunks
        # self.outcomes_responses = [("/home/k1327409/rl/outcomes.txt",
        #                        "/home/k1327409/rl/simulated_choices_30.txt"),
        #                       ("/home/k1327409/rl/outcomes.txt",
        #                        "/home/k1327409/rl/simulated_choices_10.txt")
        #                       ]

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


    def run(self, **kwargs):

        RL_parallel_fit = pe.MapNode(name='RL_parallel_fit',
                                 interface=util.Function(input_names=['model', 'outcomes_responses', 'fit_method', 'kwargs'],
                                                         output_names=['fits'],
                                                         function=RL_parallel_fit_func),
                                 iterfield=['model', 'outcomes_responses'])
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
                                                             output_names=['fits'],
                                                             function=RL_parallel_sim_func),
                                     iterfield=['model', 'outcomes_responses'])
        RL_parallel_sim.inputs.model = self.pickled_models_combined
        RL_parallel_sim.inputs.outcomes_responses = self.outcomes_responses_combined
        RL_parallel_sim.inputs.fit_method = self.method

        combine_fits = pe.Node(name='combine_fits',
                               interface=util.Function(input_names=['fits', 'dir'],
                                                       output_names=['fits', 'parameters'],
                                                       function=combine_fits_func))
        combine_fits.inputs.dir = self.out_dir

        wf = pe.Workflow(name="RL_modelling")
        wf.base_dir = self.out_dir
        # wf.config = {'execution': {'remove_node_directories': False}}

        wf.connect([(RL_parallel_sim, combine_fits, [('fits', 'fits')])])

        wf.run(**kwargs)

