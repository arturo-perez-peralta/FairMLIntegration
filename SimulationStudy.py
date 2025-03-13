#!/usr/bin/python3

#============================================================================
#==========---------------------------------------------=====================
#==========                  IMPORTS                    =====================
#==========---------------------------------------------=====================
#============================================================================

# Standard libraries
import pickle
import os
import threading
import logging

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.AUTO_REUSE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# aif360
# German dataset
from aif360.datasets import GermanDataset

# Pre processors
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover

# In processors
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing

# Post processors
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing import RejectOptionClassification

# Custom imports
import utils

# Dictionary that will store all the results
resultsDict = dict()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')


def worker(filler):

    #============================================================================
    #==========---------------------------------------------=====================
    #==========                  SIMULATION                 =====================
    #==========---------------------------------------------=====================
    #============================================================================
    
    
    #=========================================================================
    #                          SIMULATION DATASET
    #=========================================================================
    
    #-------------------------------------------------------------------------
    #                          One variable
    #-------------------------------------------------------------------------
    
#    i = threading.get_native_id()
    
    def simul1V(seed: int = 12345, N: int = 5000, p1: float = 0.5, p2: float = 0.5):
        """
        Obtain a simulated dataset from the toy model for the case of one sensitive variable
        ====================================================================================
        Inputs:
            seed (int): seed needed to ensure reproductibility.
            N (int): number of individuals in the dataset.
            p1 (float, between 0.0 and 1.0): probability for a binomial distribution from which to draw the first sensitive variable.
            p2 (float, between 0.0 and 1.0): probability for a binomial distribution from which to draw the second sensitive variable.
            
        Outputs:
            data_train (aif360.StandardDataset): Train dataset obtained from the simulation.
            data_val (aif360.StandardDataset): Validation dataset obtained from the simulation.
            data_test (aif360.StandardDataset): Test dataset obtained from the simulation.
            sensitive_attribute (str): Name of the sensitive attribute .
            privileged_groups (list): list that stores a dictionary with the sensitive attribute and the privileged label.
            unprivileged_groups (list): list that stores a dictionary with the sensitive attribute and the unprivileged label.
        """
        # Set the seed
        np.random.seed(seed)
    
        # Create variables
        vars = dict()
    
        # Sensitive variables (drawn from a binomial distribution)
        vars['sens1'] = np.random.binomial(n = 1, p = p1, size = N)
        vars['sens2'] = np.random.binomial(n = 1, p = p2, size = N)
    
        # v1, v2 (noisy measurements of the sensitive variables) and their sum
        vars['v1'] = np.random.normal(loc = vars['sens1'], scale = 1.0, size = N)
        vars['v2'] = np.random.normal(loc = vars['sens2'], scale = 1.0, size = N)
        vars['mean'] = np.mean(vars['v1'] + vars['v2'])
    
        # Noisy measurements of the sum of v1 and v2
        vars['indirect'] = np.random.normal(loc = vars['mean'], scale = 1.0, size = N)
        vars['weight_response'] = np.random.normal(loc = vars['mean'], scale = 1.0, size = N)
    
        # Response variable
        vars['response'] = vars['weight_response'] > 0.0
    
        # Create the dataset with the correct variables
        final_vars = ['sens1', 'sens2', 'indirect', 'response']
        df = dict()
        for name in final_vars:
            df[name] = vars[name]
        
        # Transform the sensitive variables to boolean
        df['sens1'] = df['sens1'] == 1
        df['sens2'] = df['sens2'] == 1
    
        # Create the dataset from the dictionary
        df = pd.DataFrame(df)
    
        # Convert to standard dataset
        data = utils.convert_to_standard_dataset(
            df=df,
            target_label_name = 'response',
            sensitive_attribute = ['sens1'],
            priviledged_classes = [lambda x: x == 1],
            favorable_target_label = [1],
            features_to_keep = [],
            categorical_features = ['sens2']
        )
    
        # train, val, test split
        data_train, vt = data.split([0.7], shuffle=True, seed=seed)
        data_val, data_test = vt.split([0.5], shuffle=True, seed=seed)
    
        # Obtain sensitive attributes and privileged groups
        sensitive_attribute = data.protected_attribute_names[0] 
        privileged_groups, unprivileged_groups = utils.get_privileged_groups(data)
    
        return data_train, data_val, data_test, sensitive_attribute, privileged_groups, unprivileged_groups
    
    #-------------------------------------------------------------------------
    #                          Two variables
    #-------------------------------------------------------------------------
    
    
    def simul2V(seed: int = 12345, operation: str = "OR", N: int = 5000, p1: float = 0.5, p2: float = 0.5):
        """
        Obtain a simulated dataset from the toy model for the case of two sensitive variables
        =====================================================================================
        Inputs:
            seed (int): seed needed to ensure reproductibility.
            operation (str): bitwise operation that we apply to the sensitive variables.
                             Allowed values: "OR", "AND", "XOR".
            N (int): number of individuals in the dataset.
            p1 (float, between 0.0 and 1.0): probability for a binomial distribution from which to draw the first sensitive variable.
            p2 (float, between 0.0 and 1.0): probability for a binomial distribution from which to draw the second sensitive variable.
            
        Outputs:
            data_train (aif360.StandardDataset): Train dataset obtained from the simulation with a bitwise operation applied to two sensitive variables.
            data_val (aif360.StandardDataset): Validation dataset obtained from the simulation with a bitwise operation applied to two sensitive variables.
            data_test (aif360.StandardDataset): Test dataset obtained from the simulation with a bitwise operation applied to two sensitive variables.
            sensitive_attribute (str): Name of the sensitive attribute.
            privileged_groups (list): list that stores a dictionary with the sensitive attribute and the privileged label.
            unprivileged_groups (list): list that stores a dictionary with the sensitive attribute and the unprivileged label.
            data_val_single (aif360.StandardDataset): Validation dataset with just one sensitive variable.
            data_test_single (aif360.StandardDataset): Test dataset with just one sensitive variable.
        """
        # Set the seed
        np.random.seed(seed)
    
        # Create variables
        vars = dict()
    
        # Sensitive variables (drawn from a binomial distribution)
        vars['sens1'] = np.random.binomial(n = 1, p = p1, size = N)
        vars['sens2'] = np.random.binomial(n = 1, p = p2, size = N)
    
        # v1, v2 (noisy measurements of the sensitive variables) and their sum
        vars['v1'] = np.random.normal(loc = vars['sens1'], scale = 1.0, size = N)
        vars['v2'] = np.random.normal(loc = vars['sens2'], scale = 1.0, size = N)
        vars['mean'] = np.mean(vars['v1'] + vars['v2'])
    
        # Noisy measurements of the sum of v1 and v2
        vars['indirect'] = np.random.normal(loc = vars['mean'], scale = 1.0, size = N)
        vars['weight_response'] = np.random.normal(loc = vars['mean'], scale = 1.0, size = N)
    
        # Response variable
        vars['response'] = vars['weight_response'] > 0.0
    
        # Create the dataset with the correct variables
        final_vars = ['sens1', 'sens2', 'indirect', 'response']
        df = dict()
        for name in final_vars:
            df[name] = vars[name]
        
        df['sens1'] = df['sens1'] == 1
        df['sens2'] = df['sens2'] == 1
    
        # Apply bitwise operation
        if operation == 'OR':
            df['prot_attr'] = np.logical_or(df['sens1'], df['sens2'])
    
        elif operation == 'AND':
            df['prot_attr'] = np.logical_and(df['sens1'], df['sens2'])
    
        elif operation == 'XOR':
            df['prot_attr'] = np.logical_xor(df['sens1'], df['sens2'])
    
        df = pd.DataFrame(df)
    
        # Convert to standard datasets
        data_single = utils.convert_to_standard_dataset(
            df=df,
            target_label_name = 'response',
            sensitive_attribute = ['sens1'],
            priviledged_classes = [lambda x: x == 1],
            favorable_target_label = [1],
            features_to_keep = [],
            categorical_features = []
        )
    
        data = utils.convert_to_standard_dataset(
            df=df,
            target_label_name = 'response',
            sensitive_attribute = ['prot_attr'],
            priviledged_classes = [lambda x: x == 1],
            favorable_target_label = [1],
            features_to_keep = [],
            categorical_features = []
        )
    
        # train, val, test split
        data_train, vt = data.split([0.7], shuffle=True, seed=seed)
        data_val, data_test = vt.split([0.5], shuffle=True, seed=seed)
    
        _, vt_single = data_single.split([0.7], shuffle=True, seed=seed)
        data_val_single, data_test_single = vt_single.split([0.5], shuffle=True, seed=seed)
    
        # Obtain sensitive attributes and privileged groups
        sensitive_attribute = data.protected_attribute_names[0] 
        privileged_groups, unprivileged_groups = utils.get_privileged_groups(data)
    
        return data_train, data_val, data_test, sensitive_attribute, privileged_groups, unprivileged_groups, data_val_single, data_test_single
    
    
    #============================================================================
    #==========---------------------------------------------=====================
    #==========                  AUXILIARY FUNCTIONS        =====================
    #==========---------------------------------------------=====================
    #============================================================================
    
    def ObtainPrelDataSingle() -> tuple[dict]:
        """
        Compute the results dictionary in the univariate case
        ====================================================================================
        Inputs:
            None
            
        Outputs:
            modelsNames (list): name of the models to whom we will apply a pre processor on post processor.
            modelsTrain (dictionary): dictionary that relates the previous names with their functions.
            modelsArgs (dictionary): dictionary that stores for each of the previous names the corresponding keyword arguments.
        """
    
        # names 
        modelsNames = [
            'logreg',
            'xgboost'
        ]
    
        modelsTrain = {
            'logreg': LogisticRegression,
            'xgboost': XGBClassifier
        }
    
        modelsArgs = {
            'logreg': {
                'solver': 'liblinear',
                'random_state': seed
            },
            'xgboost': {
                'eval_metric': 'error',
                'eta':0.1,
                'max_depth':6,
                'subsample':0.8
            }
        }
    
        return modelsNames, modelsTrain, modelsArgs
    
    
    
    
    def ObtainPrelDataMultiple(sensitive_attribute, privileged_groups, unprivileged_groups) -> tuple[dict]:
        """
        Compute the results dictionary in the univariate case
        ====================================================================================
        Inputs:
            None
            
        Outputs:
            modelsNames (list): name of the models to whom we will apply a pre processor on post processor.
            modelsBenchmark (list): name of the models that are not fairness processors themselves.
            modelsPost (list): name of the models to whom we will apply a post processor.
            modelsTrain (dictionary): dictionary that relates the previous names with their functions.
            modelsArgs (dictionary): dictionary that stores for each of the previous names the corresponding keyword arguments.
        """
    
    
        # Names of the models 
        modelsNames = [
            'logreg',
            'xgboost',
            'adversarial',
            'metafair',
            'pir'
        ]
    
        # Which models are previous benchmarks
        modelsBenchmark = [
            'logreg',
            'xgboost'
        ]
    
        # Which models are fairness processors
        modelsFair = [
            'adversarial',
            'metafair_sr',
            'metafair_fdr',
            'pir'
        ]
    
        # We obtain the names of pre processors + benchmarks (later we will apply a post processor)
        modelsPre = [
            prefix + '_' + model_name for prefix in ['RW', 'DI'] for model_name in modelsBenchmark
        ]
    
        # modelsPost is a list with the names of the models to whom we need to apply a post processor later
        # (i.e. pre processors or in processors)
        modelsPost = modelsPre + modelsFair
    
        # Names of the models with their functions
        modelsTrain = {
            'logreg': LogisticRegression,
            'xgboost': XGBClassifier,
            'adversarial': AdversarialDebiasing,
            'metafair': MetaFairClassifier,
            'pir': PrejudiceRemover
        }
    
        # Dictionary of kwargs
        modelsArgs = {
            'logreg': {
                'solver': 'liblinear',
                'random_state': seed
            },
            'xgboost': {
                'eval_metric': 'error',
                'eta':0.1,
                'max_depth':6,
                'subsample':0.8
            },
            'adversarial': {
                'privileged_groups': privileged_groups,
                'unprivileged_groups': unprivileged_groups,
                'scope_name': 'debiased_classifier',
                'debias': True,
                'num_epochs': 80
            },
            'metafair': {
                'tau': 0.8,
                'sensitive_attr': sensitive_attribute,
                'type': 'sr',
                'seed': seed
            },
        #    'metafair_fdr': {
        #        'tau': 0.8,
        #        'sensitive_attribute': sensitive_attribute,
        #        'type': 'fdr',
        #        'seed': seed
        #    },
            'pir': {
                'sensitive_attr': sensitive_attribute,
                'eta': 50.0
            }
        }
    
        return modelsNames, modelsBenchmark, modelsPost, modelsTrain, modelsArgs
    
    
    def results(val: pd.DataFrame, test: pd.DataFrame, method: str) -> None:
        """
        Compute the results dictionary in the univariate case
        ====================================================================================
        Inputs:
            val (pd.DataFrame): validation data set with LP.
            test (pd.DataFrame): validation data set with LP. 
            method (str): name of the model whose results we want to compute.
            
        Outputs:
            None    (it  modifies the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
        
        # Evaluate the model in a range of thresholds
        metrics_sweep[method] = utils.metrics_threshold_sweep(
            dataset=val,
            model=methods[method],
            thresh_arr=thresh_sweep
        )
    
        # Evaluate the metrics for the best threshold
        metrics_best_thresh_validate[method] = utils.describe_metrics(
            metrics_sweep[method],
            measurement,
            combination
            )
    
        # Compute the metrics in test using the best threshold for validation
        metrics_best_thresh_test[method] = utils.compute_metrics(
            dataset=test, 
            model=methods[method], 
            threshold=metrics_best_thresh_validate[method]['best_threshold'])
        
    
    
    def results_mult(val: pd.DataFrame, val_single: pd.DataFrame, test: pd.DataFrame, test_single: pd.DataFrame, method: str) -> None:
        """
        Compute the results dictionary in the multivariate case
        ====================================================================================
        Inputs:
            val (pd.DataFrame): validation data set with LP.
            val_single (pd.DataFrame): validation data set with a single sensitive variable.
            test (pd.DataFrame): validation data set with LP.
            test_single (pd.DataFrame): validation data set with a single sensitive variable.
            method (str): name of the model whose results we want to compute.
            
        Outputs:
            None    (it  modifies the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
        # Evaluate the model in a range of thresholds
        metrics_sweep[method] = utils.metrics_threshold_sweep_mult(
            dataset = val,
            dataset_single = val_single,
            model = methods[method],
            thresh_arr = thresh_sweep
        )
    
        # Evaluate the metrics for the best threshold
        metrics_best_thresh_validate[method] = utils.describe_metrics(metrics_sweep[method])
    
        # Compute the metrics in test using the best threshold for validation
        metrics_best_thresh_test[method] = utils.compute_metrics_mult(
            dataset = test, 
            dataset_single = test_single,
            model = methods[method], 
            threshold = metrics_best_thresh_validate[method]['best_threshold'])
    
    
    
    
    
    #============================================================================
    #==========---------------------------------------------=====================
    #==========                  PROCESSORS                 =====================
    #==========---------------------------------------------=====================
    #============================================================================
    
    
    def BenchmarkLogistic():
        """
        Training and validation of a logistic regression model
        ====================================================================================
        Inputs:
            None
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
        # Assign the correct name
        model_name = 'logreg'
    
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
    
        # Model parameters
        fit_params = {'sample_weight': train.instance_weights}
    
        # Introduce the model in the model dict
        methods[model_name] = LogisticRegression(
            solver='liblinear',
            random_state=seed
        )
    
        # Train the model
        methods[model_name] = methods[model_name].fit(train.features, train.labels.ravel(), **fit_params)
    
        # Obtain results
        if nvar == 1:
            results(val, test, model_name)
    
        elif nvar == 2:
            val_single, test_single = data_val_single.copy(deepcopy = True), data_test_single.copy(deepcopy = True)
            results_mult(val, val_single, test, test_single, model_name)
    
    
    
    
    def BenchmarkXGB():
        """
        Training and validation of a XGBoost model
        ====================================================================================
        Inputs:
            None
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
        # Assign the correct name
        model_name = 'xgboost'
    
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
    
        # Model parameters
        fit_params = {'eval_metric': 'error', 'eta':0.1, 'max_depth':6, 'subsample':0.8}
    
        # Assign the correct dict
        methods[model_name] = XGBClassifier(**fit_params)
    
        # Train the model
        methods[model_name] = methods[model_name].fit(train.features, train.labels.ravel())
    
        # Obtain results
        if nvar == 1:
            results(val, test, model_name)
    
        elif nvar == 2:
            val_single, test_single = data_val_single.copy(deepcopy = True), data_test_single.copy(deepcopy = True)
            results_mult(val, val_single, test, test_single, model_name)
    
    
    
    #=========================================================================
    #                          REWEIGHTING
    #=========================================================================
    
    
    def PreprocRW(model, do_results = True):
        """
        Implement the reweighting processor and then applies a given model
        ====================================================================================
        Inputs:
            model (sklearn or aif360 model): The model to whom we are going to apply reweighting
            do_results (boolean): If true, it modifies the results dictionaries in place. 
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
        
        # Assign the correct name
        method = "RW"
        model_name = method + "_" + model
    
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
        
        # Call the processor
        PreProcessor = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
    
        # Transform the data
        PreProcessor.fit(train)
        trainRW = PreProcessor.transform(train)
        valRW = PreProcessor.transform(test)
        testRW = PreProcessor.transform(val)
    
        # Train the model
        if model == 'adversarial':
            tf.compat.v1.reset_default_graph()
            modelsArgs[model]['sess'] = tf.Session()
    
        Algorithm = modelsTrain[model](**modelsArgs[model])
    
        if model in modelsBenchmark:
            if model == 'logreg':
                fit_params = {'sample_weight': trainRW.instance_weights}
                methods[model_name] = Algorithm.fit(trainRW.features, trainRW.labels.ravel(), **fit_params)
            else:
                methods[model_name] = Algorithm.fit(trainRW.features, trainRW.labels.ravel())
        else:
            methods[model_name] = Algorithm.fit(trainRW)
                
        # Obtain results
        if do_results:
            if nvar == 1:
                results(valRW, testRW, model_name)
    
            elif nvar == 2:
                val_single, test_single = data_val_single.copy(deepcopy = True), data_test_single.copy(deepcopy = True)
                results_mult(valRW, val_single, testRW, test_single, model_name)
    
        if model == 'adversarial':
            modelsArgs[model]['sess'].close()
    
    
    
    #=========================================================================
    #                          DISPARATE IMPACT REMOVER
    #=========================================================================
    
    
    def PreprocDI(repair_level, model, do_results = True):
        """
        Implement the reweighting processor and then applies a given model
        ====================================================================================
        Inputs:
            repair_level (float between 0 and 1): Parameter that controls the level of repair.
                The closer it is to one, the fairer the data set.
            model (sklearn or aif360 model): The model to whom we are going to apply reweighting.
            do_results (boolean): If true, it modifies the results dictionaries in place. 
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
        
        # Assign the correct name
        method = "DI"
        model_name = method + "_" + model
    
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
    
        # Initialize the processor
        PreProcessor = DisparateImpactRemover(
            repair_level=repair_level,
            sensitive_attribute=sensitive_attribute
        )
        # Transform the data
        PreProcessor.fit_transform(train)
        trainDI = PreProcessor.fit_transform(train)
        valDI = PreProcessor.fit_transform(val)
        testDI = PreProcessor.fit_transform(test)
    
        # Train the model
        # If we are training adversarial debiasing we need a tf session.
        if model == 'adversarial':
            tf.compat.v1.reset_default_graph()
            modelsArgs[model]['sess'] = tf.Session()
    
        Algorithm = modelsTrain[model](**modelsArgs[model])
    
        # This logic handles whether or not the model is a sklearn model or a aif360 model.
        if model in modelsBenchmark:
            if model == 'logreg':
                fit_params = {'sample_weight': trainDI.instance_weights}
                methods[model_name] = Algorithm.fit(trainDI.features, trainDI.labels.ravel(), **fit_params)
            else:
                methods[model_name] = Algorithm.fit(trainDI.features, trainDI.labels.ravel())
        else:
            methods[model_name] = Algorithm.fit(trainDI)
    
        # Obtain results
        if do_results:
            if nvar == 1:
                results(valDI, testDI, model_name)
    
            elif nvar == 2:
                val_single, test_single = data_val_single.copy(deepcopy = True), data_test_single.copy(deepcopy = True)
                results_mult(valDI, val_single, testDI, test_single, model_name)
    
        # If we are dealing with adversarial debiasing we close the session
        if model == 'adversarial':
            modelsArgs[model]['sess'].close()
    
    
    
    def InprocMeta(quality: str,  tau: float = 0.8, do_results: bool = True):
        """
        Implement the meta fair in processor
        ====================================================================================
        Inputs:
            quality (str): "fdr" for false discovery ratio, "sr" for statistical rate.
            tau (float): penalty parameter of the fairness constraint.
            do_results (boolean): If true, it modifies the results dictionaries in place. 
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
    
        # assign the correct name
        model_name = "metafair"
        model_name_quality = '{}_{}'.format(model_name, quality)
    
        # Initialize the model and store it in the dictionary
        methods[model_name_quality] = MetaFairClassifier(
            tau=tau,
            sensitive_attr=sensitive_attribute,
            type=quality,
            seed=seed
            )
    
        # Train the model
        methods[model_name_quality] = methods[model_name_quality].fit(train)
    
        # Obtain scores
        methods[model_name_quality].scores_train = methods[model_name_quality].predict(train).scores
        methods[model_name_quality].scores_val = methods[model_name_quality].predict(val).scores
        methods[model_name_quality].scores_test = methods[model_name_quality].predict(test).scores
    
        # Obtain results
        if do_results:
            if nvar == 1:
                results(val, test, model_name_quality)
    
            elif nvar == 2:
                val_single, test_single = data_val_single.copy(deepcopy = True), data_test_single.copy(deepcopy = True)
                results_mult(val, val_single, test, test_single, model_name_quality)
    
    
    
    def InprocPI(eta = 50.0, do_results = True):
        """
        Implement the prejudice index regularizer in processor
        ====================================================================================
        Inputs:
            eta (float): parameter that weights the importance given to the regularizer (similar to lambda in lasso regression).
            do_results (boolean): If true, it modifies the results dictionaries in place. 
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
        # Assign the correct name
        model_name = 'pir'
        
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
        
        # Initialize the model and store it in the dictionary
        methods[model_name] = PrejudiceRemover(
            sensitive_attr=sensitive_attribute,
            eta=eta
            )
        
        # Train the model
        methods[model_name] = methods[model_name].fit(train)
        
        # Obtain scores
        methods[model_name].scores_train = methods[model_name].predict(train).scores
        methods[model_name].scores_val = methods[model_name].predict(val).scores
        methods[model_name].scores_test = methods[model_name].predict(test).scores
    
        # Obtain results
        if do_results:
            if nvar == 1:
                results(val, test, model_name)
    
            elif nvar == 2:
                val_single, test_single = data_val_single.copy(deepcopy = True), data_test_single.copy(deepcopy = True)
                results_mult(val, val_single, test, test_single, model_name)
    
    
    
    
    def InprocAdvs(do_results = True):
        """
        Implement the adversarial debiasing in processor
        ====================================================================================
        Inputs:
            do_results (boolean): If true, it modifies the results dictionaries in place. 
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
        
        # Assign the correct name
        model_name = 'adversarial'
        
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
        
        #We train the model
        methods[model_name] = AdversarialDebiasing(
            privileged_groups = privileged_groups,
            unprivileged_groups = unprivileged_groups,
            scope_name = 'debiased_classifier',
            debias=True,
            sess=sess,
            num_epochs=80
        )    
        methods[model_name].fit(train)
    
        # Obtain results
        if do_results:
            if nvar == 1:
                results(val, test, model_name)
    
            elif nvar == 2:
                val_single, test_single = data_val_single.copy(deepcopy = True), data_test_single.copy(deepcopy = True)
                results_mult(val, val_single, test, test_single, model_name)
    
    
    
    def PosprocPlatt(model_name):
        """
        Implement the Platt scaling by groups post processor
        ====================================================================================
        Inputs:
            model_name (str): Name of the model we want to do post processing to. 
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
    
        # Assign the correct name
        fairness_method = '_Platt'
    
        # Validation
        #---------------
    
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy = True), data_val.copy(deepcopy = True), data_test.copy(deepcopy = True)
    
        # Copy the predictions
        model_thresh = metrics_best_thresh_validate[model_name]['best_threshold']
        val_preds = utils.update_dataset_from_model(val, methods[model_name], class_thresh = model_thresh)
    
        ## Platt Scaling:
        #---------------
        #1. Split training data on sensitive attribute
        val_preds_priv, val_preds_unpriv, priv_indices, unpriv_indices = utils.split_dataset_on_sensitive_attribute(
            dataset = val_preds,
            privileged_group_label = list((privileged_groups[0].values()))[0]
        )
        
        #2. Copy validation data predictions
        val_preds2 = val_preds.copy(deepcopy = True)
        
        #3. Make one model for each group
        sensitive_groups_data = {'priv': [val_preds_priv, priv_indices],
                                 'unpriv': [val_preds_unpriv, unpriv_indices]}
        for group, data_group_list in sensitive_groups_data.items():
            # Assign the correct name
            model_name_group = '{}_{}_{}'.format(model_name, fairness_method, group)
            # Initialize the model, store it in the dict
            methods[model_name_group] = LogisticRegression()
            # Train the model using the validation data divided by group
            methods[ model_name_group ] = methods[model_name_group].fit(
                data_group_list[0].scores,   # data_group_list[0] -> data_val_preds_priv or data_val_preds_unpriv
                val.subset(data_group_list[1]).labels.ravel()
            ) # data_group_list[1] -> priv_indices or unpriv_indices
    
            # predict group probabilities, store in val_preds2
            # Platt scores are given by the predictions of the posterior probabilities
            scores_group = methods[model_name_group].predict_proba(data_group_list[0].scores)
            pos_ind_group = np.where(methods[model_name_group].classes_ == data_group_list[0].favorable_label)[0][0]
            val_preds2.scores[data_group_list[1]] = scores_group[:, pos_ind_group].reshape(-1,1)
       
        # Evaluate the model in a range of values
        thresh_sweep_platt = np.linspace(np.min(val_preds2.scores.ravel()),
                                         np.max(val_preds2.scores.ravel()),
                                         50)
    
        # Obtain the metrics for the val set
        metrics_sweep[model_name+fairness_method] = utils.metrics_postprocessing_threshold_sweep_from_scores(
                dataset_true = val,
                dataset_preds = val_preds,
                thresh_arr = thresh_sweep_platt
            )
    
        # Evaluate metrics and obtain the best thresh
        metrics_best_thresh_validate[model_name+fairness_method] = utils.describe_metrics(metrics_sweep[model_name+fairness_method])
    
        # Test
        #---------------
    
        model_thresh = metrics_best_thresh_validate[model_name]['best_threshold']
        test_preds = utils.update_dataset_from_model(test, methods[model_name], class_thresh = model_thresh)
    
        ## Plat Scaling:
        #---------------
        
        # 1. Divide test set using sensitive varaible's groups
        test_preds_priv, test_preds_unpriv, priv_indices, unpriv_indices = utils.split_dataset_on_sensitive_attribute(
            dataset = test_preds,
            privileged_group_label = list((privileged_groups[0].values()))[0]
        )
        # 2. Copy test data
        if nvar == 1:
            test_preds2 = test_preds.copy(deepcopy = True)
        elif nvar == 2:
            test_single = data_test.copy(deepcopy = True)
            test_preds2 = data_test.copy(deepcopy = True)
            test_single.scores = np.zeros_like(test_single.labels)
    
        # 3. Predict for each group
        sensitive_groups_data_test = {'priv': [test_preds_priv, priv_indices],
                                      'unpriv': [test_preds_unpriv, unpriv_indices]}
        
    
        for group, data_group_list in sensitive_groups_data_test.items():    
            # We assign the correct name
            model_name_group = '{}_{}_{}'.format(model_name, fairness_method, group)
    
            # Predict in each group, store the result in data_val_preds2
            # The probabilities are the Platt scores
            scores_group = methods[model_name_group].predict_proba(data_group_list[0].scores)
            pos_ind_group = np.where(methods[model_name_group].classes_ == data_group_list[0].favorable_label)[0][0]
            test_preds2.scores[data_group_list[1]] = scores_group[:, pos_ind_group].reshape(-1,1)
    
    
        if nvar == 1:    
            # Obtain metrics
            metrics_best_thresh_test[model_name+fairness_method] = utils.compute_metrics_from_scores(
                dataset_true = test,
                dataset_pred = test_preds2,
                threshold = metrics_best_thresh_validate[model_name+fairness_method]['best_threshold']
            )
    
        elif nvar == 2:
            # Obtain metrics
            metrics_best_thresh_test[model_name+fairness_method] = utils.compute_metrics_from_scores(
                dataset_true = test_single,
                dataset_pred = test_preds2,
                threshold = metrics_best_thresh_validate[model_name+fairness_method]['best_threshold']
            )
    
    
    def PosprocEqoddsLABELS(model_name):
        """
        Implement the Equald Odds post processor given prediction labels
        ====================================================================================
        Inputs:
            model_name (str): Name of the model we want to do post processing to.
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
        # Assign the correct name
        fairness_method = '_eqOdds' 
    
        # Copy the dataset
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
    
        # Copy the predictions of the base model
        train_preds = utils.update_dataset_from_model(train, methods[model_name])
        val_preds = utils.update_dataset_from_model(val, methods[model_name])
        test_preds = utils.update_dataset_from_model(test, methods[model_name])
    
        # Initialize the model and store the predictions
        methods[model_name+fairness_method] = EqOddsPostprocessing(
            privileged_groups = privileged_groups,
            unprivileged_groups = unprivileged_groups, 
            seed = seed)
    
        # Train the model
        methods[model_name+fairness_method] = methods[model_name+fairness_method].fit(train, train_preds)
    
        # Evaluate the model in a range of thresholds
        metrics_sweep[model_name+fairness_method] = utils.metrics_postprocessing_threshold_sweep(
            dataset_true=val,
            dataset_preds=val_preds,
            model=methods[model_name+fairness_method],
            thresh_arr=thresh_sweep,
            scores_or_labels='labels'
        )
    
        # Evaluate the model for the best threshold
        metrics_best_thresh_validate[model_name+fairness_method] = utils.describe_metrics(metrics_sweep[model_name+fairness_method])
    
        if nvar == 1:
    
            # We use the best threshold to obtain predicitions for test
            metrics_best_thresh_test[model_name+fairness_method] = utils.compute_metrics_postprocessing(
                dataset_true=test,
                dataset_preds=test_preds,
                model=methods[model_name+fairness_method], 
                threshold=metrics_best_thresh_validate[model_name+fairness_method]['best_threshold'], 
                scores_or_labels='labels'
            )
    
        elif nvar == 2:
    
            test_single = data_test_single.copy(deepcopy=True)
            # We use the best threshold to obtain predicitions for test
            metrics_best_thresh_test[model_name+fairness_method] = utils.compute_metrics_postprocessing_mult(
                dataset_true=test,
                dataset_preds=test_preds,
                dataset_true_single = test_single,
                model=methods[model_name+fairness_method], 
                threshold=metrics_best_thresh_validate[model_name+fairness_method]['best_threshold'], 
                scores_or_labels='labels'
            )
    
    
    
    
    def PosprocEqoddsSCORES(model_name, quality):
        """
        Implement the Equald Odds post processor given a score card.
        ====================================================================================
        Inputs:
            model_name (str): Name of the model we want to do post processing to. 
            quality (str): "fpr" (false positive rate), "fnr" (false negative rate) or "weighted" (weighted combination of both).
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
         # Assign the correct name
        fairness_method = '_eqOdds'
    
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
    
        # Copy the model's predictions
        train_preds = utils.update_dataset_from_model(train, methods[model_name])
        val_preds = utils.update_dataset_from_model(val, methods[model_name])
        test_preds = utils.update_dataset_from_model(test, methods[model_name])
    
        # Assign the correct name
        model_name_metric = model_name + fairness_method + '_' + quality
        
        # Initialize the model 
        methods[model_name_metric] = CalibratedEqOddsPostprocessing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            cost_constraint=quality,
            seed=seed)
        
        # Train the model
        methods[model_name_metric] = methods[model_name_metric].fit(train, train_preds)
    
        # Evaluate the model for a range of thresholds
        metrics_sweep[model_name_metric] = utils.metrics_postprocessing_threshold_sweep(
            dataset_true = val,
            dataset_preds = val_preds,
            model = methods[model_name_metric],
            thresh_arr = thresh_sweep,
            scores_or_labels = 'scores'
        )
    
        # Evaluate in best thresh
        metrics_best_thresh_validate[model_name_metric] = utils.describe_metrics(metrics_sweep[model_name_metric])
    
        if nvar == 1:
    
            # Using the best thresh, evaluate in test
            metrics_best_thresh_test[model_name_metric] = utils.compute_metrics_postprocessing(
                dataset_true=test,
                dataset_preds=test_preds,
                model=methods[model_name_metric], 
                threshold=metrics_best_thresh_validate[model_name_metric]['best_threshold'], 
                scores_or_labels='scores'
            )
    
        elif nvar == 2:
            test_single = data_test_single.copy(deepcopy=True)
    
            # We use the best threshold to obtain predicitions for test
            metrics_best_thresh_test[model_name+fairness_method] = utils.compute_metrics_postprocessing_mult(
                dataset_true=test,
                dataset_preds=test_preds,
                dataset_true_single = test_single,
                model=methods[model_name+fairness_method], 
                threshold=metrics_best_thresh_validate[model_name+fairness_method]['best_threshold'], 
                scores_or_labels='labels'
            )
    
    
    
    def PosprocReject(model_name, key_metric):
        """
        Implement the Option rejection post processor
        ====================================================================================
        Inputs:
            model_name (str): Name of the model we want to do post processing to. 
            key_metric (str): 'spd' (Statistical parity difference), 'aod' (Average odds difference) or 'eod' ("Equal opportunity difference").
            
        Outputs:
            None    (it  may modify the metrics_sweep, metrics_best_thresh_validate and 
                     metrics_best_thresh_test dictionaries in place)
        """
    
        # Assign the correct name
        fairness_method = '_RejOpt'
        model_name_metric = model_name + fairness_method + '_' + key_metric
    
        # Copy the datasets
        train, val, test = data_train.copy(deepcopy=True), data_val.copy(deepcopy=True), data_test.copy(deepcopy=True)
    
        # Copy predictions
        train_preds = utils.update_dataset_from_model(train, methods[model_name])
        val_preds = utils.update_dataset_from_model(val, methods[model_name])
        test_preds = utils.update_dataset_from_model(test, methods[model_name])
    
        # Train the model
        methods[model_name_metric] = RejectOptionClassification(
            unprivileged_groups=unprivileged_groups, 
            privileged_groups=privileged_groups, 
            metric_name=fair_metrics_optrej[key_metric],
            metric_lb=-0.01,
            metric_ub=0.01
            )
    
        # Train the model
        methods[model_name_metric] = methods[model_name_metric].fit(train, train_preds)
    
    
        if nvar == 1:
            # Obtain best threshold in val
            metrics_best_thresh_validate[model_name_metric] = utils.compute_metrics_postprocessing(
                dataset_true=val, 
                dataset_preds=val_preds, 
                model=methods[model_name_metric], 
                required_threshold=False)
            
            # Obtain it in test
            metrics_best_thresh_test[model_name_metric] = utils.compute_metrics_postprocessing(
                dataset_true=test, 
                dataset_preds=test_preds, 
                model=methods[model_name_metric], 
                required_threshold=False)
            
        elif nvar == 2:
            val_single, test_single = data_val_single.copy(deepcopy=True), data_test_single.copy(deepcopy=True)
            # Obtain best threshold in val
            metrics_best_thresh_validate[model_name_metric] = utils.compute_metrics_postprocessing_mult(
                dataset_true=val, 
                dataset_preds=val_preds,
                dataset_true_single=val_single, 
                model=methods[model_name_metric], 
                required_threshold=False)
            
            # Obtain it in test
            metrics_best_thresh_test[model_name_metric] = utils.compute_metrics_postprocessing_mult(
                dataset_true=test, 
                dataset_preds=test_preds, 
                dataset_true_single=val_single,
                model=methods[model_name_metric], 
                required_threshold=False)
    
    
    
    #============================================================================
    #==========---------------------------------------------=====================
    #==========                  MAIN LOOP FUNCTION         =====================
    #==========---------------------------------------------=====================
    #============================================================================
    
    
    #============================================================================
    #==========                  PARAMETERS                 =====================
    #============================================================================
    
    
    # DI remover
    repair_level = 0.5                      
    
    
    # MetaFair classifier
    quality_constraints_meta = ['sr', 'fdr']
    tau = 0.8   
    
    # MetaFair classifier
    quality_constraints_meta = ['sr', 'fdr']
    
    # Equal odds
    quality_constraints_eqodds = ["weighted", 'fnr', 'fpr']
    
    # Reject option
    fair_metrics_optrej = {
        'spd': "Statistical parity difference",
        'aod': "Average odds difference",
        'eod': "Equal opportunity difference"
    }
    
    
    # Change this list if you want to use multiple 
    datasets = ['Simulation']
    nvars = ['1', '2']
    
    # What operations to consider for the LPs
    # operations = ['OR', 'AND', 'XOR']
    operations = ['XOR']
    
    # ind = individual case, com = use of multistage processors
    cases = ['ind', 'com']
    
    # Functions to load the data sets
    loadDatasets = {
        'Simulation1V': simul1V,
        'Simulation2V': simul2V
    }
    
    # What measure of performance should be optimized when selecting the threshold
    measurement = 'bal_acc'
    combination = []

    seed = np.random.randint(10000)
    np.random.seed(seed)
    i = seed
    thread_name = 'Thread ' + str(i)

    logger.info('Starting.')

    for data in datasets:
        for nvar in nvars:
            # Select name of the data set
            dataset = data + nvar + 'V'

                        # Univariate case 
            if nvar == '1':
                # Arguments for the iteration
                argumentsLoadData = {
                    'seed': seed
                }
                nvar = 1

                # Load data
                data_train, data_val, data_test, \
                sensitive_attribute, privileged_groups, \
                unprivileged_groups = loadDatasets[dataset](**argumentsLoadData)

                for case in cases:
                    # No multistage processor
                    if case == 'ind': 

                        # Obtain benchmarks
                        modelsNames, modelsTrain, modelsArgs = ObtainPrelDataSingle()
                        modelsBenchmark = modelsNames


                        # Initialize dicts
                        methods = dict()

                        # Range of thresholds to evaluate our models
                        thresh_sweep = np.linspace(0.01, 1.0, 50)
                        metrics_sweep = dict()

                        # Store results from validation and test
                        metrics_best_thresh_validate = dict()
                        metrics_best_thresh_test = dict()

                        # Benchmarks
                        BenchmarkLogistic()
                        BenchmarkXGB()
                        
                        # Pre processing
                        for model in modelsNames:
                            PreprocRW(model, do_results = True)
                            PreprocDI(repair_level, model, do_results = True)
                        
                        # In processing
                        for quality in quality_constraints_meta:
                            InprocMeta(quality, tau = 0.8, do_results = True)
                        InprocPI(eta = 50.0, do_results = True)
                        
                        tf.compat.v1.reset_default_graph()
                        sess = tf.compat.v1.Session()
                        InprocAdvs(do_results = True)
                        sess.close()
                        
                        # Post processing
                        for model in modelsNames:
                            PosprocPlatt(model)
                            PosprocEqoddsLABELS(model)
                            for quality in quality_constraints_eqodds:
                                PosprocEqoddsSCORES(model, quality)
                            for key_metric in fair_metrics_optrej:
                                PosprocReject(model, key_metric)

                        # Name of the training instance
                        file = dataset + '_' + case + '_' + str(i)
                        
                        # Store the results in a dictionary
                        resultsDict[file] = dict()
                        resultsDict[file]['methods'] = methods
                        resultsDict[file]['best_thresh_test'] = pd.DataFrame(metrics_best_thresh_test).T
                        resultsDict[file]['metrics_sweep'] = metrics_sweep

                        # Use pickle to save the results
                        with open('results/best/' + data + '/' + file + '_best.pickle', 'wb') as handle:
                            pickle.dump(resultsDict[file]['best_thresh_test'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        with open('results/sweep/' + data + '/' + file + '_sweep.pickle', 'wb') as handle:
                            pickle.dump(resultsDict[file]['metrics_sweep'], handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # Multistage processor
                    elif case == 'com':
                        # Obtain benchmarks and in proncessing models
                        modelsNames, modelsBenchmark, modelsPost, \
                        modelsTrain, modelsArgs = ObtainPrelDataMultiple(sensitive_attribute, privileged_groups, unprivileged_groups)

                        # Initialize dicts
                        methods = dict()

                        # Range of thresholds to evaluate our models
                        thresh_sweep = np.linspace(0.01, 1.0, 50)
                        metrics_sweep = dict()

                        # Store results from validation and test
                        metrics_best_thresh_validate = dict()
                        metrics_best_thresh_test = dict()

                        # Benchmarks
                        BenchmarkLogistic()
                        BenchmarkXGB()

                        # Pre processing + In processing
                        for model in modelsNames:
                            if model == 'adversarial':
                                tf.compat.v1.reset_default_graph()
                                sess = tf.compat.v1.Session()
                            PreprocRW(model, do_results = True)
                            if model == 'adversarial':
                                sess.close()
                                tf.compat.v1.reset_default_graph()
                                sess = tf.compat.v1.Session()
                            PreprocDI(repair_level, model, do_results = True)
                            
                            if model == 'adversarial':
                                sess.close()

                        # Pre/In processing + Post processing
                        for quality in quality_constraints_meta:
                            InprocMeta(quality, tau = 0.8, do_results = True)
                        InprocPI(eta = 50.0, do_results = True)

                        tf.compat.v1.reset_default_graph()
                        sess = tf.compat.v1.Session()
                        InprocAdvs(do_results = True)

                        for model in modelsPost:
                            PosprocPlatt(model)
                            PosprocEqoddsLABELS(model)
                            for quality in quality_constraints_eqodds:
                                PosprocEqoddsSCORES(model, quality)
                            for key_metric in fair_metrics_optrej:
                                PosprocReject(model, key_metric)
                                
                        sess.close()
                        
                        # Name of the training instance
                        file = dataset + '_' + case + '_' + str(i)

                        # Store the results in a dictionary
                        resultsDict[file] = dict()
                        resultsDict[file]['methods'] = methods
                        resultsDict[file]['best_thresh_test'] = pd.DataFrame(metrics_best_thresh_test).T
                        resultsDict[file]['metrics_sweep'] = metrics_sweep
                        
                        # Save them with pickle
                        with open('results/best/' + data + '/' + file + '_best.pickle', 'wb') as handle:
                            pickle.dump(resultsDict[file]['best_thresh_test'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        with open('results/sweep/' + data + '/' + file + '_sweep.pickle', 'wb') as handle:
                            pickle.dump(resultsDict[file]['metrics_sweep'], handle, protocol=pickle.HIGHEST_PROTOCOL)

            
            # Multivariate case
            if nvar == '2':
                for operation in operations:
                        # The arguments of the function that loads the data are now different
                        argumentsLoadData = {
                            'seed': seed,
                            'operation': operation
                        }
                        nvar = 2

                        resultsDict[dataset + '_' + operation] = dict()

                        # Load the data
                        data_train, data_val, data_test, \
                        sensitive_attribute, privileged_groups, unprivileged_groups, \
                        data_val_single, data_test_single = loadDatasets[dataset](**argumentsLoadData)
            
                        for case in cases:
                            if case == 'ind': 
                                logger.info('Starting Individual Multivariate ' + operation + ' case.')
                                # Initialize dicts
                                methods = dict()

                                # Obtain benchmarks
                                modelsNames, modelsTrain, modelsArgs = ObtainPrelDataSingle()
                                modelsBenchmark = modelsNames

                                # Range of thresholds to evaluate our models
                                thresh_sweep = np.linspace(0.01, 1.0, 50)
                                metrics_sweep = dict()

                                # Store results from validation and test
                                metrics_best_thresh_validate = dict()
                                metrics_best_thresh_test = dict()

                                # Benchmarks
                                logger.info('Starting benchmarks.')
                                BenchmarkLogistic()
                                logger.info('Logistic finished.')
                                BenchmarkXGB()
                                logger.info('XGB finished.')
                                
                                # Pre processing
                                logger.info('Starting pre.')
                                for model in modelsNames:
                                    PreprocRW(model, do_results = True)
                                    logger.info('RW of ' + model + ' finished.')
                                    PreprocDI(repair_level, model, do_results = True)
                                    logger.info('DI of ' + model + ' finished.')
                                logger.info('pre finished.')
                                
                                # In processing
                                logger.info('Starting in.')
                                for quality in quality_constraints_meta:
                                    InprocMeta(quality, tau = 0.8, do_results = True)
                                logger.info('Meta fair finished.')
                                InprocPI(eta = 50.0, do_results = True)
                                logger.info('PI finished.')
                                
                                tf.compat.v1.reset_default_graph()
                                sess = tf.compat.v1.Session()
                                InprocAdvs(do_results = True)
                                logger.info('Adversarial finished.')
                                sess.close()
                                logger.info('in finished.')
                                
                                # Post processing
                                logger.info('Starting post.')
                                for model in modelsNames:
                                    PosprocPlatt(model)
                                    logger.info('Platt of ' + model + ' finished.')
                                    PosprocEqoddsLABELS(model)
                                    for quality in quality_constraints_eqodds:
                                        PosprocEqoddsSCORES(model, quality)
                                    logger.info('EqOdds of ' + model + ' finished.')
                                    for key_metric in fair_metrics_optrej:
                                        PosprocReject(model, key_metric)
                                    logger.info('OptRej of ' + model + ' finished.')

                                file =  dataset + '_' + operation + '_' + case + '_' + str(i)

                                # Store results in a dictionary
                                resultsDict[file] = dict()
                                resultsDict[file]['methods'] = methods
                                resultsDict[file]['best_thresh_test'] = pd.DataFrame(metrics_best_thresh_test).T
                                resultsDict[file]['metrics_sweep'] = metrics_sweep

                                # Save results with pickle
                                with open(os.getcwd() + '/results/simulation/' + file + '_best.pickle', 'wb') as handle:
                                    pickle.dump(resultsDict[file]['best_thresh_test'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                                logger.info('Finishing Individual Multivariate ' + operation + ' case.')


                            # Multistage processors
                            elif case == 'com':
                                logger.info('Starting Combination Multivariate ' + operation + ' case.')
                                
                                # Obtain benchmarks and in proncessing models
                                modelsNames, modelsBenchmark, modelsPost, \
                                modelsTrain, modelsArgs = ObtainPrelDataMultiple(sensitive_attribute, privileged_groups, unprivileged_groups)

                                # Initialize dicts
                                methods = dict()
                                # Range of thresholds to evaluate our models
                                thresh_sweep = np.linspace(0.01, 1.0, 50)

                                # Store results from validation and test
                                metrics_best_thresh_validate = dict()
                                metrics_best_thresh_test = dict()

                                # Benchmarks
                                logger.info('Benchmarks starting.')
                                BenchmarkLogistic()
                                logger.info('Logistic finished.')
                                BenchmarkXGB()
                                logger.info('XGB finished.')
                                logger.info('Benchmarks finished.')

                                # Pre processing + In processing
                                logger.info('Pre + In starting.')
                                for model in modelsNames:
                                    if model == 'adversarial':
                                        tf.compat.v1.reset_default_graph()
                                        sess = tf.compat.v1.Session()
                                    PreprocRW(model, do_results = True)
                                    logger.info('RW of ' + model + ' finished.')
                                    if model == 'adversarial':
                                        sess.close()
                                        tf.compat.v1.reset_default_graph()
                                        sess = tf.compat.v1.Session()
                                    PreprocDI(repair_level, model, do_results = True)
                                    logger.info('DI of ' + model + ' finished.')
                                    
                                    if model == 'adversarial':
                                        sess.close()

                                # Pre/In processing + Post processing
                                logger.info('pre/in + post starting.')
                                logger.info('Loading in.')
                                for quality in quality_constraints_meta:
                                    InprocMeta(quality, tau = 0.8, do_results = True)
                                logger.info('Meta fair finished.')
                                InprocPI(eta = 50.0, do_results = True)
                                logger.info('PI finished.')

                                tf.compat.v1.reset_default_graph()
                                sess = tf.compat.v1.Session()
                                InprocAdvs(do_results = True)
                                logger.info('Advs finished.')
                                logger.info('in finished.')

                                for model in modelsPost:
                                    PosprocPlatt(model)
                                    logger.info('Platt of ' + model + ' finished.')
                                    PosprocEqoddsLABELS(model)
                                    for quality in quality_constraints_eqodds:
                                        PosprocEqoddsSCORES(model, quality)
                                    logger.info('EqOdds of ' + model + ' finished.')
                                    for key_metric in fair_metrics_optrej:
                                        PosprocReject(model, key_metric)
                                    logger.info('OptReject of ' + model + ' finished.')
                                        
                                sess.close()

                                file = dataset + '_' + operation + '_' + case + '_' + str(i)

                                # Store results in a dictionary                            
                                resultsDict[file] = dict()
                                resultsDict[file]['methods'] = methods
                                resultsDict[file]['best_thresh_test'] = pd.DataFrame(metrics_best_thresh_test).T
                                resultsDict[file]['metrics_sweep'] = metrics_sweep
                            
                                # Save results with pickle
                                with open(os.getcwd() + '/results/simulation/' + file + '_best.pickle', 'wb') as handle:
                                    pickle.dump(resultsDict[file]['best_thresh_test'], handle, protocol=pickle.HIGHEST_PROTOCOL)

                                logger.info('Finishing Combination Multivariate ' + operation + ' case.')



#============================================================================
#==========                  MAIN LOOP                  =====================
#============================================================================



n_simuls = 50

# Create a list to hold the thread objects 
threads = []

for i in range(n_simuls):
    thread = threading.Thread(target=worker, args=(i,)) 
    threads.append(thread) 
    thread.start() 

exit()
    