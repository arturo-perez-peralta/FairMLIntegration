from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# aif360 libraries
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import OneHotEncoder

# for GermanDataset multi_attribute
from aif360.datasets import GermanDataset

# HELPER FUNCTIONS -----------------------------------------------------------------------------------------------------

def convert_to_standard_dataset(df, target_label_name, sensitive_attribute, priviledged_classes=[[0]],
                                favorable_target_label=[1], features_to_keep=[], categorical_features=[], **kwargs):
    """
    Transforms a dataframe `df` into the format required by aif360 (a StandardDataset object), which contains all the useful information about sensitive attributes and target variables.
    Args:
        df (pandas.dataframe): dataset containing features, sensitive attribute(s), and target variables.
        target_label_name (str): Column name of the target variables
        sensitive_attribute (str, list): Column name(s) of the sensitive attribute(s)
        priviledged_classes (list, optional): Value of the privileged classes of the sensitive attribute(s). Defaults to [[0]].
        favorable_target_label (int, list, optional): Value of the positive target. Defaults to [1].
        features_to_keep (list, optional): Column names of the features of `df` to keep. Defaults to [], in which case it keeps every feature.
        categorical_features (list, optional): Column names of the features to apply one hot encoding. Defaults to [], in which case it oncodes every non-numeric feature.
    Returns:
        StandardDataset: dataset constructed from `df` in the required aif360 format.
    """

    """Converts dataframe to the required format for AIF360 models.
    **kwargs are passed into the StandardDataset class."""

    if type(sensitive_attribute) != list:
        sensitive_attribute = [sensitive_attribute]

    if type(priviledged_classes) == int:
        priviledged_classes = [[priviledged_classes]]

    if type(favorable_target_label) == int:
        favorable_target_label = [favorable_target_label]

    if categorical_features == []:  # == 'infer'
        if features_to_keep == []:
            features_to_keep = df.columns.drop(target_label_name).to_list()  # sensitive_attribute

            coltypes = df[features_to_keep].astype(int, errors='ignore').dtypes
            categorical_features = coltypes[coltypes == 'object'].index.values

    # create the `StandardDataset` object
    standard_dataset = StandardDataset(df=df,
                                       label_name=target_label_name,
                                       favorable_classes=favorable_target_label,
                                       protected_attribute_names=sensitive_attribute,
                                       privileged_classes=priviledged_classes,
                                       categorical_features=categorical_features,
                                       features_to_keep=features_to_keep,
                                       **kwargs)
    return standard_dataset


def get_privileged_groups(dataset, sens_attr_ix=0):
    """
    Helper function to privileged and unprivileged group dictionaries from `dataset`
    Args:
        dataset (StandardDataset): dataset in the aif360 format.
        sens_attr_ix (int, optional): Index of the dataset.privileged_protected_attributes pointing to the sensitive attribute. Defaults to 0.
    Returns:
        tuple(list, list): privileged group and unprivileged group
    """

    sens_attr = dataset.protected_attribute_names[sens_attr_ix]
    priviledged_groups = [{sens_attr: v} for v in dataset.privileged_protected_attributes[sens_attr_ix]]
    unpriviledged_groups = [{sens_attr: v} for v in dataset.unprivileged_protected_attributes[sens_attr_ix]]

    return priviledged_groups, unpriviledged_groups


def get_privileged_groups_multi(dataset):
    """
    Helper function to privileged and unprivileged groups dictionaries from `dataset`
    Args:
        dataset (StandardDataset): dataset in the aif360 format with more than one sensitive attribute.
        
    Returns:
        tuple(list, list): privileged group and unprivileged group
    """
    
    tuple_priv = zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)
    tuple_unpriv = zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)
    result_unpriv = [{k: v} for k, v in tuple_unpriv]
    return [dict(tuple_priv)], result_unpriv
    

def split_dataset_on_sensitive_attribute(dataset, privileged_group_label):
    """
    """
    
    # find indices of previleged and unprivileged group
    group_test = np.asarray(dataset.protected_attributes.ravel() == privileged_group_label)
    privileged_indices = group_test.nonzero()[0]
    unprivileged_indices = (~group_test).nonzero()[0]
    
    # subset data based on indices of groups
    dataset_priv = dataset.subset(privileged_indices).copy(deepcopy = True)
    dataset_unpriv = dataset.subset(unprivileged_indices).copy(deepcopy = True)
    
    # return splits and indices    
    return dataset_priv, dataset_unpriv, privileged_indices, unprivileged_indices


# UPDATE DATASETS ------------------------------------------------------------------------------------------------------

def update_dataset_from_model(dataset, model, class_thresh=0.5):
    """
    Returns a copy of `dataset` with updated scores and labels predicted by `model`
    Args:
        dataset (StandardDataset): A StandardDataset
        model: must have `predict_proba()` and `classes_` instances.
        class_thresh (float): A numeric threshold for prediction
    Returns:
        Dataset with updated scores and labels
    TODO: Obtain model with best `class_thresh` automatically (only relevant for EqOddsPostprocessing).
    """

    # Create copy of dataset
    dataset_pred = dataset.copy(deepcopy=True)

    # obtain the position of the favorable index
    pos_ind = np.where(model.classes_ == dataset_pred.favorable_label)[0][0]

    # Obtain prob. scores from model for positive index
    y_pred_prob = model.predict_proba(dataset_pred.features)[:, pos_ind]
    # Obtain predicted labels according to class_threshold
    y_pred = np.zeros_like(dataset_pred.labels)
    y_pred[y_pred_prob >= class_thresh] = dataset_pred.favorable_label
    y_pred[~(y_pred_prob >= class_thresh)] = dataset_pred.unfavorable_label

    # Update dataset scores and labels
    dataset_pred.scores = y_pred_prob.reshape(-1, 1)
    dataset_pred.labels = y_pred

    return dataset_pred


def update_dataset_from_scores(dataset, scores, thresh):
    """
    Returns a dataset with updated labels of `dataset` based on the passed scores. This doesn't modify `dataset`.
    Args:
        dataset (StandardDataset): dataset in the aif360 format.
        scores (list): list of scores matching the scores in `dataset`.
        thresh (float): threshold
    Returns:
        StandardDataset: dataset with predictions and labels based on `scores`.
    """
    labels_raw = (scores > thresh).astype(np.float64)

    label_mapping_dict = {0: dataset.unfavorable_label, 1: dataset.favorable_label}
    dataset_updated = dataset.copy(deepcopy=True)

    labels = np.vectorize(label_mapping_dict.__getitem__)(labels_raw)

    dataset_updated.scores = np.vstack(scores)
    dataset_updated.labels = np.vstack(labels)
    return dataset_updated
    

def update_german_dataset_from_multiple_protected_attributes(dataset, operation = "OR"):
    """
    Returns a copy of the german dataset with updated protected attribute based on an OR summation of `age` and `sex`. This doesn't modify `dataset`.
    Args:
        dataset (StandardDataset): The German dataset in the aif360 format, containing only one protected attribute.
    Returns:
        StandardDataset: Another German dataset whose protected attribute is an OR summation of `age` and `sex`.
    """
    # create new dataset and get the shape of the current protected attr
    dataset_updated = dataset.copy(deepcopy=True)
    shape_protected_attribute = dataset_updated.protected_attributes.shape
    
    # load the german dataset with both age and sex and protected attr
    dataset_german_multi_protected_attr = GermanDataset(
        favorable_classes = lambda x: x == 1,
        features_to_drop=['personal_status'],
        metadata = {'label_maps': {1.0: 'Good Credit', 0.0: 'Bad Credit'},
                    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
                                           {1.0: 'Old', 0.0: 'Young'}]})
    
    if operation == "OR":
        # do an OR summation of age and sex
        new_protected_attrs = ~(np.sum(dataset_german_multi_protected_attr.protected_attributes, axis = 1) == 0)
        new_protected_attrs = new_protected_attrs.astype(int).reshape(shape_protected_attribute)

    elif operation == "AND":
        # do an AND product of age and sex
        new_protected_attrs = ~(np.prod(dataset_german_multi_protected_attr.protected_attributes, axis = 1) == 0)
        new_protected_attrs = new_protected_attrs.astype(int).reshape(shape_protected_attribute)

    elif operation == "XOR":
        # do XOR of age and sex
        orSum = ~(np.sum(dataset_german_multi_protected_attr.protected_attributes, axis = 1) == 0)
        andProd = ~(np.prod(dataset_german_multi_protected_attr.protected_attributes, axis = 1) == 0)
        new_protected_attrs = orSum - andProd
        new_protected_attrs = new_protected_attrs.astype(int).reshape(shape_protected_attribute)
    
    # add protected attribute to datacopy 
    dataset_updated.protected_attributes = new_protected_attrs
    
    # change the name of protected attrs
    sensitive_attr = dataset.protected_attribute_names[0] # dataset should have only one protected attr
    ind = np.where(np.array(dataset_updated.feature_names) == sensitive_attr)[0][0] # find column index of sensitive attr
    dataset_updated.protected_attribute_names[0] = 'the_protected_attr'
    dataset_updated.feature_names[ind] = 'the_protected_attr'
    return dataset_updated
    

# METRICS-RELATED ------------------------------------------------------------------------------------------------------

def compute_metrics(dataset, model, threshold):
    """
    Computes a pletora of perfomance metrics of a (fairness) model on a dataset over an specific threshold.
    Args:
        dataset (StandardDataset): dataset with classes in the aif360 format.
        model: aif360 or sklearn trained classifier.
        threshold (float): threshold for the metric computation
    Returns:
        dict: Dictionary of metrics where keys are metrics
    """
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0


    # Remove nans and turn them into score of random prediction (R=0.5)
    y_val_pred_prob = np.nan_to_num(y_val_pred_prob, nan=0.5)

    privileged_groups, unprivileged_groups = get_privileged_groups(dataset)

    metric_dict = defaultdict()
    # include threshol into metric_dict
    metric_dict['best_threshold'] = threshold

    y_pred_probs = y_val_pred_prob[:, pos_ind]
    y_true = dataset.labels

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs, pos_label=dataset.favorable_label)

    dataset_pred = update_dataset_from_scores(dataset, y_val_pred_prob[:, pos_ind], threshold)

    metric = ClassificationMetric(
        dataset, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    metric_dict['bal_acc'] = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    metric_dict['acc'] = metric.accuracy()
    metric_dict['independence'] = np.abs(metric.statistical_parity_difference())
    metric_dict['separation'] = np.abs(
        metric.false_positive_rate_difference() + metric.false_negative_rate_difference()) / 2
    metric_dict['sufficiency'] = np.abs(
        metric.positive_predictive_value(True) - metric.positive_predictive_value(False))
    metric_dict['auc'] = auc(fpr, tpr)
    return metric_dict


def compute_metrics_from_scores(dataset_true, dataset_pred, threshold):
    """
    Computes a pletora of perfomance metrics of a (fairness) model using only the predicted scores over an specific threshold.
    This function doesn't require the model argument as in other `compute_metrics` functions
    Relevant for Platt scaling
    Args:
        dataset_true (StandardDataset): dataset with true classes in the aif360 format.
        dataset_pred (StandardDataset): dataset with predicted socres in the aif360 format.
        threshold (float): threshold for the metric computation
    Returns:
        dict: Dictionary of metrics where keys are metrics
    """
    privileged_groups, unprivileged_groups = get_privileged_groups(dataset_true)
    
    fav_inds = dataset_pred.scores > threshold
    
    dataset_pred.labels[fav_inds] = dataset_true.favorable_label
    dataset_pred.labels[~fav_inds] = dataset_true.unfavorable_label

    metric_dict = defaultdict()
    
    metric_dict['best_threshold'] = threshold
    
    fpr, tpr, _ = roc_curve(dataset_true.labels, dataset_pred.scores, pos_label = dataset_true.favorable_label)

    metric = ClassificationMetric(dataset_true,
                                  dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)

    metric_dict['bal_acc'] = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    metric_dict['acc'] = metric.accuracy()
    metric_dict['independence'] = np.abs(metric.statistical_parity_difference())
    metric_dict['separation'] = np.abs(metric.false_positive_rate_difference() + metric.false_negative_rate_difference()) / 2
    metric_dict['sufficiency'] = np.abs(metric.positive_predictive_value(True) - metric.positive_predictive_value(False))
    metric_dict['auc'] = auc(fpr, tpr)
    
    return metric_dict


def metrics_threshold_sweep(dataset, model, thresh_arr):
    """
    Computes a pletora of perfomance metrics of a (fairness) model on a dataset over an array of score thresholds.
    Args:
        dataset (StandardDataset): dataset with classes in the aif360 format.
        model: aif360 or sklearn trained classifier.
        thresh_arr (list): List of thresholds to sweep the metrics through.
    Returns:
        dict: Dictionary of metrics where keys are metrics and values are arrays with the metric value over
        `thresh_arr`.
    """
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    # Remove nans and turn them into score of random prediction (R=0.5)
    y_val_pred_prob = np.nan_to_num(y_val_pred_prob, nan=0.5)

    privileged_groups, unprivileged_groups = get_privileged_groups(dataset)

    metric_arrs = defaultdict(list)
    # include threshol array and predictions scores into metric_arrs
    metric_arrs['thresh_arr'] = thresh_arr
    metric_arrs['y_pred_probs'] = y_val_pred_prob[:, pos_ind]
    metric_arrs['y_true'] = dataset.labels

    FPR, TPR, _ = roc_curve(metric_arrs['y_true'], metric_arrs['y_pred_probs'], pos_label=dataset.favorable_label)
    metric_arrs['auc'] = auc(FPR, TPR)

    for thresh in thresh_arr:
        dataset_pred = update_dataset_from_scores(dataset, y_val_pred_prob[:, pos_ind], thresh)

        metric = ClassificationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        independence = np.abs(metric.statistical_parity_difference())
        separation = np.abs(metric.false_positive_rate_difference() + metric.false_negative_rate_difference()) / 2
        sufficiency = np.abs(metric.positive_predictive_value(True) - metric.positive_predictive_value(False))
        metric_arrs['acc'].append(metric.accuracy())
        metric_arrs['bal_acc'].append((metric.true_positive_rate() + metric.true_negative_rate()) / 2)
        metric_arrs['independence'].append(independence)
        metric_arrs['separation'].append(separation)
        metric_arrs['sufficiency'].append(sufficiency)
    return metric_arrs


def compute_metrics_postprocessing(dataset_true, dataset_preds, model, threshold=None, required_threshold=True,
                                   scores_or_labels='scores'):
    """
    Computes a pletora of perfomance metrics of a postprocessing model on a dataset over an array of score thresholds.
    Args:
        dataset_true (StandardDataset): dataset with classes in the aif360 format.
        dataset_preds (StandardDataset): dataset with classes in the aif360 format.
        model: aif360 or sklearn trained classifier.
        threshold (float): List of thresholds to sweep the metrics through.
        required_threshold (Boolean): Boolean indicating if threshold is required. Methodologies lke RejectOption do not
                                      require thresholds because these are already implemented within the methodology
        scores_or_labels (string): string indicating if sweep should be done considering scores or labels
    Returns:
        dict: Dictionary of metrics where keys are metrics and values are floats with the metric value for
        `threshold`.
    """

    privileged_groups, unprivileged_groups = get_privileged_groups(dataset_true)

    # define metric_arrs
    metrics_best_thresh = defaultdict()

    metrics_best_thresh['best_threshold'] = threshold

    FPR, TPR, _ = roc_curve(dataset_true.labels, dataset_preds.scores, pos_label=dataset_true.favorable_label)
    metrics_best_thresh['auc'] = auc(FPR, TPR)

    if scores_or_labels == 'scores' and required_threshold is True:
        dataset_preds_trans = dataset_preds.copy()
        dataset_preds_trans.labels = model.predict(dataset_preds, threshold).labels
    elif scores_or_labels == 'labels' and required_threshold is True:
        dataset_preds = update_dataset_from_scores(dataset_preds, dataset_preds.scores, threshold)
        dataset_preds_trans = dataset_preds.copy()
        dataset_preds_trans.labels = model.predict(dataset_preds).labels
    elif required_threshold is False:
        metrics_best_thresh['best_threshold'] = model.classification_threshold
        dataset_preds_trans = dataset_preds.copy()
        dataset_preds_trans.labels = model.predict(dataset_preds).labels
    else:
        raise ValueError('Invalid value for parameter scores_or_labels or required_threshold.')

    metric = ClassificationMetric(
        dataset_true, dataset_preds_trans,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    metrics_best_thresh['acc'] = metric.accuracy()
    metrics_best_thresh['bal_acc'] = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    metrics_best_thresh['independence'] = np.abs(metric.statistical_parity_difference())
    metrics_best_thresh['separation'] = np.abs( metric.false_positive_rate_difference() + metric.false_negative_rate_difference()) / 2
    metrics_best_thresh['sufficiency'] = np.abs(metric.positive_predictive_value(True) - metric.positive_predictive_value(False))

    return metrics_best_thresh


def metrics_postprocessing_threshold_sweep(dataset_true, dataset_preds, model, thresh_arr, scores_or_labels='scores'):
    """
    Computes a pletora of perfomance metrics of a postprocessing model on a dataset over an array of score thresholds.
    Args:
        dataset_true (StandardDataset): dataset with classes in the aif360 format.
        dataset_preds (StandardDataset): dataset with classes in the aif360 format.
        model: aif360 or sklearn trained classifier.
        thresh_arr (list): List of thresholds to sweep the metrics through.
        scores_or_labels (string): string indicating if sweep should be done considering scores or labels
    Returns:
        dict: Dictionary of metrics where keys are metrics and values are arrays with the metric value over
        `thresh_arr`.
    """
    privileged_groups, unprivileged_groups = get_privileged_groups(dataset_true)

    # define metric_arrs
    metric_arrs = defaultdict(list)
    metric_arrs['thresh_arr'] = thresh_arr

    FPR, TPR, _ = roc_curve(dataset_true.labels, dataset_preds.scores, pos_label=dataset_true.favorable_label)
    metric_arrs['auc'] = auc(FPR, TPR)

    for t in thresh_arr:
        if scores_or_labels == 'scores':
            dataset_preds_trans = dataset_preds.copy()
            dataset_preds_trans.labels = model.predict(dataset_preds, t).labels
        elif scores_or_labels == 'labels':
            dataset_preds = update_dataset_from_scores(dataset_preds, dataset_preds.scores, t)
            dataset_preds_trans = dataset_preds.copy()
            dataset_preds_trans.labels = model.predict(dataset_preds).labels
        else:
            raise ValueError('Invalid value for parameter scores_or_labels. Valid values are: scores; labels')

        metric = ClassificationMetric(
            dataset_true, dataset_preds_trans,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )

        independence = np.abs(metric.statistical_parity_difference())
        separation = np.abs(metric.false_positive_rate_difference() + metric.false_negative_rate_difference()) / 2
        sufficiency = np.abs(metric.positive_predictive_value(True) - metric.positive_predictive_value(False))
        metric_arrs['acc'].append(metric.accuracy())
        metric_arrs['bal_acc'].append((metric.true_positive_rate() + metric.true_negative_rate()) / 2)
        metric_arrs['independence'].append(independence)
        metric_arrs['separation'].append(separation)
        metric_arrs['sufficiency'].append(sufficiency)

    return metric_arrs


def metrics_postprocessing_threshold_sweep_from_scores(dataset_true, dataset_preds, thresh_arr):
    """
    Computes a pletora of perfomance metrics on a predicted dataset with scores over an array of score thresholds.
    Main difference with other `metrics_postprocessing_*` functions is that this doesn't require a model to sweep.
    Relevant only for Platt Scaling since it doesn't use a single model (there is a model for each sensitive group)
    Args:
        dataset_true (StandardDataset): dataset with true classes in the aif360 format.
        dataset_preds (StandardDataset): dataset with predicted scores in the aif360 format.
        thresh_arr (list): List of thresholds to sweep the metrics through.
    Returns:
        dict: Dictionary of metrics where keys are metrics and values are arrays with the metric value over
        `thresh_arr`.
    """
    privileged_groups, unprivileged_groups = get_privileged_groups(dataset_true)

    # define metric_arrs
    metric_arrs = defaultdict(list)
    metric_arrs['thresh_arr'] = thresh_arr

    # TODO: Do this better
    FPR, TPR, _ = roc_curve(dataset_true.labels, dataset_preds.scores, pos_label=dataset_true.favorable_label)
    metric_arrs['auc'] = auc(FPR, TPR)

    for thresh in thresh_arr:
        dataset_preds_updated = dataset_preds.copy(deepcopy=True)

        # create predicted dataset
        fav_inds = dataset_preds.scores > thresh
        dataset_preds_updated.labels[fav_inds] = dataset_true.favorable_label
        dataset_preds_updated.labels[~fav_inds] = dataset_true.unfavorable_label

        metric = ClassificationMetric(dataset_true, dataset_preds_updated,
                                      unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)

        metric_arrs['acc'].append(metric.accuracy())
        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)

        independence = np.abs(metric.statistical_parity_difference())
        separation = np.abs(metric.false_positive_rate_difference() + metric.false_negative_rate_difference()) / 2
        sufficiency = np.abs(metric.positive_predictive_value(True) - metric.positive_predictive_value(False))

        metric_arrs['independence'].append(independence)
        metric_arrs['separation'].append(separation)
        metric_arrs['sufficiency'].append(sufficiency)
        
        # other metrics of interest
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())  # Separation
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())  # Sufficiency
        
    return metric_arrs


def describe_metrics(metrics):
    """
    Prints and returns the metrics for the best threshold on `thresh_arr`.
    Args:
        metrics (dictionary): Output of `metrics_threshold_sweep`
        thresh_arr (list): Threshold array passed to `metrics_threshold_sweep`.
    Return:
        Dictionary where metrics are keys and values are the best values for the threshold sweep
    """
    best_ind = np.argmax(metrics['bal_acc'])
    best_metrics = dict()
    # PER THRESHOLD METRICS
    best_metrics['best_threshold'] = metrics['thresh_arr'][best_ind]
    best_metrics['bal_acc'] = metrics['bal_acc'][best_ind]
    best_metrics['acc'] = metrics['acc'][best_ind]
    best_metrics['independence'] = metrics['independence'][best_ind]
    best_metrics['separation'] = metrics['separation'][best_ind]
    best_metrics['sufficiency'] = metrics['sufficiency'][best_ind]
    # AGGREGATE METRICS
    best_metrics['auc'] = metrics['auc']
    return best_metrics


def print_metrics(best_metrics):
    """
    Prints the metrics within the dictionary or aif360 ClassificationMetric object.
    Args:
        best_metrics (dictionary or aif360 ClassificationMetric) : dictionary containing the values associated to metrics or output of aif360 `ClassificationMetric`
    Return:
        None
    """
    if(isinstance(best_metrics, ClassificationMetric)):
        bal_acc = (best_metrics.true_positive_rate() + best_metrics.true_negative_rate()) / 2
        acc = best_metrics.accuracy()
        independence = np.abs(best_metrics.statistical_parity_difference())
        separation = np.abs(best_metrics.false_positive_rate_difference() + best_metrics.false_negative_rate_difference()) / 2
        sufficiency = np.abs(best_metrics.positive_predictive_value(True) - best_metrics.positive_predictive_value(False))
    elif(isinstance(best_metrics, dict)):
        bal_acc = best_metrics['bal_acc']
        acc = best_metrics['acc']
        independence = best_metrics['independence']
        separation = best_metrics['separation']
        sufficiency = best_metrics['sufficiency']
        print("Threshold corresponding to *best balanced accuracy*: {:6.4f}".format(best_metrics['best_threshold']))
        print("AUC: {:6.4f}".format(best_metrics['auc']))
    else:
        print("Object of type: {} not supported".format(type(best_metrics)))
    print("Balanced accuracy: {:6.4f}".format(bal_acc))
    print("Accuracy: {:6.4f}".format(acc))
    print("Independence ( |ΔP(Y_pred = 1)| ): {:6.4f}".format(independence))
    print("Separation ( |ΔFPR + ΔFNR|/2 ): {:6.4f}".format(separation))
    print("Sufficiency ( |ΔPPV| ) : {:6.4f}".format(sufficiency))


# PLOTTING -------------------------------------------------------------------------------------------------------------

def plot_fairness_and_accuracy(metrics_dict, accuracy_metric='bal_acc',
                               fairness_metrics=('independence', 'separation', 'sufficiency'), **kwargs):
    """
    Plots model's accuracy and selected fairness metrics (any from the output of the `metrics_threshold_sweep` function)
    Args:
        metrics_dict (dictionary): Output of `metrics_threshold_sweep` or `metrics_postprocessing_threshold_sweep`
        accuracy_metric (string): Metric used to find optimal threshold
        fairness_metrics (list): List of fairness metrics
    Return:
        A figure
    """

    # Check that fairness_metrics is a list
    if type(fairness_metrics) == 'str':
        fairness_metrics = [fairness_metrics]

    # Pivot longer the data
    y2 = []
    y2_labels = []
    for fairness_metric in fairness_metrics:
        try:
            if fairness_metric == 'disp_imp':  # this makes all fairness metrics to be better when close to 0
                y2.append(1 - np.array(metrics_dict[fairness_metric]))
                y2_labels.append('1 - {}'.format(fairness_metric))
            else:
                y2.append(np.array(metrics_dict[fairness_metric]))
                y2_labels.append(fairness_metric)
        except:
            AssertionError('invalid fairness metric. Try one of {}'.format(metrics_dict.keys()))

    # Find best values for the threshold sweep
    y1 = metrics_dict[accuracy_metric]
    thresh_arr = metrics_dict['thresh_arr']
    best_thresh = thresh_arr[np.argmax(y1)]

    # PLOTTING
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(thresh_arr, y1, color='gray', lw=4, **kwargs)
    ax1.axvline([best_thresh], color='black', ls='--')

    for i in range(len(fairness_metrics)):
        ax2.plot(thresh_arr, y2[i], label=y2_labels[i], lw=2)

    ax1.set_xlabel('Score threshold', size=15)
    ax1.set_ylabel(accuracy_metric, color='gray', size=14)
    ax2.set_ylabel('Fairness metrics', size=14)
    ax2.set_ylim(-0.01, 0.5)
    ax2.axhspan(0, 0.2, color='gray', alpha=0.2)  # span of acceptable fairness values

    ax2.legend()
    ax1.grid()
    
    plt.close()
    return fig, ax1


# PREPROCESSING DATA ---------------------------------------------------------------------------------------------------

def preprocess_homecredit(df):
    """
    Args:
        df: trainset from the homecredit data competition (https://www.kaggle.com/c/home-credit-default-risk)
    Returns:
        df: The dataframe after preprocessing for filling missing data and one-hot encoding
    """
    # Identify character and numeric variables
    char_vars = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
                 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
                 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
    numeric_vars = [i for i in df.columns if i not in char_vars]

    # NA values in OWN_CAR_AGE converted to 0: "no car"
    df.loc[df['FLAG_OWN_CAR'] == "N", "OWN_CAR_AGE"] = 0

    # NA values converted to median in numeric variables:
    df[numeric_vars] = df[numeric_vars].fillna(df[numeric_vars].median())

    # NA values converted to most frequent in categorical variables
    df[char_vars] = df[char_vars].fillna(df[char_vars].mode().iloc[0])

    # Conversion of age from days_from_birth to age (in years)
    df['AGE'] = -df['DAYS_BIRTH'].astype('float') / 365

    # Drop SK_ID_CURR: id column; DAYS_BIRTH: re-formated as AGE; CODE_GENDER: using AGE as sensitive atribute
    df = df.drop(columns=['SK_ID_CURR', 'DAYS_BIRTH', 'CODE_GENDER'])

    encoding_vars = ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
                     'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                     'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
                     'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']

    # OneHot encode the categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first')
    encoded_array = encoder.fit_transform(df.loc[:, encoding_vars])
    df_encoded = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())
    df_encoded = pd.concat([df, df_encoded], axis=1)
    df_encoded = df_encoded.drop(labels=encoding_vars, axis=1)

    return df_encoded

