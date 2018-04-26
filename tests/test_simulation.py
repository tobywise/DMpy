import unittest
from DMpy.model import DMModel, Parameter
from DMpy.learning import rescorla_wagner
from DMpy.observation import softmax
import numpy as np
from DMpy.utils import _check_column, load_data_for_simulation, parameter_check, simulated_dataframe
import pandas as pd
import pytest


@pytest.fixture()
def outcomes():

    return [0, 1, 0, 1]

@pytest.fixture()
def obs_model():

    v = Parameter('value', 'fixed', mean=0.5, dynamic=True)
    alpha = Parameter('alpha', 'fixed', mean=0.3)
    beta = Parameter('beta', 'fixed', mean=3)

    obs_model = DMModel(rescorla_wagner, [v, alpha], softmax, [beta])

    return obs_model

@pytest.fixture()
def no_obs_model():

    v = Parameter('value', 'fixed', mean=0.5, dynamic=True)
    alpha = Parameter('alpha', 'fixed', mean=0.3)

    no_obs_model = DMModel(rescorla_wagner, [v, alpha], None, None)

    return no_obs_model

@pytest.fixture()
def example_dataframe_loaded():

    return pd.read_csv('example_outcome_df.csv')

@pytest.fixture()
def simulated_model(obs_model, outcomes):
    obs_model.simulate(outcomes=np.vstack([outcomes, outcomes]).T, learning_parameters=dict(value=0.5, alpha=[0.3, 0.4]),
                       observation_parameters=dict(beta=[3, 4]), n_subjects=2, combinations=True)
    return obs_model

@pytest.fixture()
def fit_model(obs_model, responses):

    # Better to pickle this and reload?
    obs_model.fit(responses, fit_method='variational', hierarchical=True, fit_kwargs={'n': 5},
                  fit_stats=False, plot=False, suppress_table=True)

    return obs_model

def test_check_missing_parameters(obs_model, outcomes):

    """One parameter has no value provided"""

    with pytest.raises(ValueError):

        obs_model.simulate(outcomes=np.vstack([outcomes, outcomes]), learning_parameters=dict(alpha=[0.3, 0.4]),
                           observation_parameters=dict(beta=[3, 4]), n_subjects=2)

class TestParameterCombinations(object):

    """ Test method that creates combinations of parameter values"""

    def test_combination_shape(self, obs_model):

        combinations = obs_model._create_parameter_combinations(True, [[0.1, 0.2], [3, 4]], 1, 1, False)

        assert combinations.shape == (4, 2)

    def test_combination_shape_multiple_runs(self, obs_model):

        combinations = obs_model._create_parameter_combinations(True, [[0.1, 0.2], [3, 4]], 2, 1, False)

        assert combinations.shape == (8, 2)

    def test_combination_shape_multiple_subject(self, obs_model):

        combinations = obs_model._create_parameter_combinations(True, [[0.1, 0.2], [3, 4]], 1, 2, False)

        assert combinations.shape == (8, 2)

    def test_pair_shape(self, obs_model):

        combinations = obs_model._create_parameter_combinations(False, [[0.1, 0.2], [3, 4]], 1, 1, False)

        assert combinations.shape == (2, 2)

    def test_pair_shape_multiple_runs(self, obs_model):

        combinations = obs_model._create_parameter_combinations(False, [[0.1, 0.2], [3, 4]], 2, 1, False)

        assert combinations.shape == (4, 2)

    def test_pair_shape_multiple_subjects(self, obs_model):

        combinations = obs_model._create_parameter_combinations(False, [[0.1, 0.2], [3, 4]], 1, 2, False)

        assert combinations.shape == (4, 2)

    def test_value_label_order(self, obs_model, outcomes):
        # TODO test again when things aren't broken
        obs_model.simulate(outcomes=outcomes, learning_parameters=dict(value=0.5, alpha=[0.3, 0.4]),
                           observation_parameters=dict(beta=[3, 4]), n_subjects=1, combinations=True)

        assert ((obs_model.sim_learning_parameters['alpha'] == np.array([0.3, 0.3, 0.4, 0.4]))
               and (obs_model.sim_learning_parameters['value'] == np.array([0.5, 0.5, 0.5, 0.5]))
               and (obs_model.sim_observation_parameters['beta'] == np.array([3, 4, 3, 4])))

    def test_parameter_combinations_fit_model(self, fit_model):

        ##


class TestSimulatedDataFrame(object):

    """
    Test function that creates a dataframe from simulation results
    """

    def test_simulated_dataframe_subject_values(self, simulated_model):

        assert len(simulated_model.simulation_results.Subject.unique()) == 8

    def test_simulated_dataframe_run_values(self, simulated_model):

        assert len(simulated_model.simulation_results.Run.unique()) == 2

    def test_simulated_dataframe_runs_nested_within_subjects(self, simulated_model):

        df = simulated_model.simulation_results

        assert len(df.Run[df.Subject == df.Subject.unique()[0]].unique()) == 2

    def test_simulated_dataframe_same_parameter_values_within_subject(self, simulated_model):

        df = simulated_model.simulation_results

        tests = []

        for i in df.columns:
            if '_sim' in i:
                tests.append(len(df[i][df.Subject == df.Subject.unique()[0]].unique()) == 1)
        print tests
        assert all(tests)

    def test_all_parameters_in_dataframe(self, simulated_model):

        parameter_names = simulated_model.sim_learning_parameters.keys() + simulated_model.sim_observation_parameters.keys()

        for p in parameter_names:
            assert any([p in i for i in simulated_model.simulation_results])

    def test_correct_parameter_values_in_dataframe(self, simulated_model):

        for p, v in simulated_model.sim_learning_parameters.items():
            assert(np.all(simulated_model.simulation_results[p + '_sim'].unique() == np.unique(v)))

        for p, v in simulated_model.sim_observation_parameters.items():
            assert(np.all(simulated_model.simulation_results[p + '_sim'].unique() == np.unique(v)))

    def test_simulated_dataframe_fit_model(self):

        simulated_dataframe(forgetful_beta_asymmetric_model._simulation_results_dict,
                            forgetful_beta_asymmetric_model.outcomes.eval(),
                            forgetful_beta_asymmetric_model.model_inputs,
                            forgetful_beta_asymmetric_model.n_runs.eval(),
                            forgetful_beta_asymmetric_model.n_subjects,
                            forgetful_beta_asymmetric_model.subjects,
                            forgetful_beta_asymmetric_model.sim_learning_parameters,
                            forgetful_beta_asymmetric_model.sim_observation_parameters,
                            forgetful_beta_asymmetric_model.fit_complete)


def test_multiple_runs_have_same_simulated_values_combinations(simulated_model):

    """Check that multiple runs have the same values if using repeated outcome"""

    df = simulated_model.simulation_results

    # check that all runs produce the same value
    assert(np.all(df.prob[df.Run == 0] == df.prob[df.Run == 1]))



class TestSimulationOutcomeLoading(object):

    """ Test function for loading outcomes """

    def test_correct_number_of_outputs(self):

        assert len(load_data_for_simulation(np.ones(3))) == 5

    def test_convert_to_numpy_array(self):

        """ Try providing a list of pandas dataframes, this should fail as it can't be converted to an array"""

        with pytest.raises(TypeError):
            load_data_for_simulation([pd.DataFrame({'a': [1, 2]})])

    def test_array_with_model_inputs(self):

        with pytest.raises(ValueError):
            load_data_for_simulation(np.array([1, 2, 3]), ['a'])

    def test_zero_length_array(self):

        with pytest.raises(AttributeError):
            load_data_for_simulation(np.array([]))

    def test_one_length_array(self):

        with pytest.raises(AttributeError):
            load_data_for_simulation(np.array([1]))

    def test_more_than_2d_array(self):

        with pytest.raises(AttributeError):
            load_data_for_simulation(np.ones((2, 3, 4)))

    def test_nans_in_outcomes(self):

        with pytest.raises(ValueError):
            load_data_for_simulation(np.array([0, 1, np.nan]))

    def test_infs_in_outcomes(self):

        with pytest.raises(ValueError):
            load_data_for_simulation(np.array([0, 1, np.inf]))

    def test_numpy_array_outcome_shape_1D(self):

        """Tests whether a 1D array has been successfully converted to shape (n_trials, 1)"""

        assert load_data_for_simulation(np.ones(3))[0].shape == (3, 1)

    def test_numpy_array_n_runs(self):

        assert load_data_for_simulation(np.ones((3, 4)))[2] == 4

    def test_column_nans(self, example_dataframe_loaded):

        df = example_dataframe_loaded
        df.Run[0] = np.nan

        with pytest.raises(ValueError):
            _check_column(df, 'Run', equal='')

    def test_column_equal_length(self, example_dataframe_loaded):

        df = example_dataframe_loaded
        df = pd.concat([df, df[:2]])

        with pytest.raises(ValueError):
            _check_column(df, 'Subject', equal='Outcome')

    def test_dataframe_load_from_string(self):

        """ Check that loading from path produces a numpy array"""

        assert isinstance(load_data_for_simulation('example_outcome_df.csv')[0], np.ndarray)

    def test_dataframe_has_outcome_column(self, example_dataframe_loaded):

        df = example_dataframe_loaded
        df.columns = [i.replace('Outcome', 'abc') for i in df.columns]

        with pytest.raises(AttributeError):
            load_data_for_simulation(df)

    def test_dataframe_has_rows(self, example_dataframe_loaded):

        with pytest.raises(AttributeError):
            load_data_for_simulation(example_dataframe_loaded.iloc[:1, :])

    def test_dataframe_nans(self, example_dataframe_loaded):

        df = example_dataframe_loaded
        df.Run[0] = np.nan

        with pytest.raises(ValueError):
            load_data_for_simulation(df)

    def test_dataframe_n_runs(self, example_dataframe_loaded):

        assert load_data_for_simulation(example_dataframe_loaded)[2] == 2

    def test_dataframe_n_subjects(self, example_dataframe_loaded):

        assert load_data_for_simulation(example_dataframe_loaded)[3] == 2

    def test_incorrect_input_names(self, example_dataframe_loaded):

        with pytest.raises(AttributeError):
            load_data_for_simulation(example_dataframe_loaded, ['Input2'])

    def test_inputs_not_strings(self, example_dataframe_loaded):

        with pytest.raises(TypeError):
            load_data_for_simulation(example_dataframe_loaded, [1])

    def test_model_input_returns_list_of_arrays(self, example_dataframe_loaded):

        assert isinstance(load_data_for_simulation(example_dataframe_loaded, ['Input1'])[1][0], np.ndarray)


def test_parameter_check_sim_dict():
    """ Try passing a list instead of a dict """

    with pytest.raises(TypeError):
        parameter_check([], sim=True)

def test_parameter_check_sim_parameters():
    """ Test with incorrect parameter types as values in parameter dict"""

    with pytest.raises(TypeError):
        parameter_check({'a': 'a'}, sim=True)

def test_parameter_check_fit():

    """ Test with a dict instead of list of parameters """

    with pytest.raises(TypeError):
        parameter_check({}, sim=False)

def test_parameter_check_fit_parameters():

    """ Test with a list of non-Parameters class instances """

    with pytest.raises(TypeError):
        parameter_check(['a'], sim=False)










