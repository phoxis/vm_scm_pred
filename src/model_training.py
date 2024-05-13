# ###
# ### Copyright 2024 VistaMilk
# ###
# ### Licensed under the Apache License, Version 2.0 (the "License");
# ### you may not use this file except in compliance with the License.
# ### You may obtain a copy of the License at
# ###
# ###     http://www.apache.org/licenses/LICENSE-2.0
# ###
# ### Unless required by applicable law or agreed to in writing, software
# ### distributed under the License is distributed on an "AS IS" BASIS,
# ### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ### See the License for the specific language governing permissions and
# ### limitations under the License.
# ###

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMModel
from lightgbm import LGBMClassifier
import lightgbm
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import logging
import coloredlogs
from sklearn.metrics import precision_recall_curve
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import json
from imblearn.metrics import geometric_mean_score

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# TODO: Transfer this to config file
#this_day                = 0;

def load_cfg (cfg_file = "model_train_cfg.json"):
  with open (cfg_file, "r") as fp:
    cfg = json.load (fp);
    
  if "train_data_path" in cfg:
    cfg["train_data_path"] = Path (cfg["train_data_path"]);
    
  if "test_data_path" in cfg:
    cfg["test_data_path"] = Path (cfg["test_data_path"]);
    
  return (cfg);


def read_hyper_param (cfg_file = "hyper_param_cfg.cfg"):
  with open (cfg_file, "r") as fp:
    cfg = json.load (fp);

  return cfg;


def save_cfg (config, cfg_file = "model_train_cfg.json"):
  with open (cfg_file, "w") as fp:
    json.dump (config, fp);
  
  return None;

def brier (y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    grad = 2 * y_pred * (y_true - y_pred) * (y_pred - 1)
    hess = 2 * y_pred ** (1 - y_pred) * (2 * y_pred * (y_true + 1) - y_true - 3 * y_pred ** 2)
    return grad, hess


def cutoff_youdens_j (fpr,tpr,thresholds):
  
  j_scores = tpr-fpr
  j_ordered = sorted(zip(j_scores,thresholds))
  return j_ordered[-1][1]


## NOTE: NOT IN USE complete this and use this as the main function to work with, so that we don't manually compute anythings
#def confusion_metrics (conf_matrix):
  
    #TP = conf_matrix[1][1]
    #TN = conf_matrix[0][0]
    #FP = conf_matrix[0][1]
    #FN = conf_matrix[1][0]
    
    #print('True Positives:', TP)
    #print('True Negatives:', TN)
    #print('False Positives:', FP)
    #print('False Negatives:', FN)
    
    
    #scores = {"TP": TP, "TN": TN, "FP": FP, "FN": FN};
    
    ## calculate accuracy
    #scores["accuracy"] = (float (TP+TN) / float(TP + TN + FP + FN))
    
    #scores["balanced_accuracy"] = 1111;
    
    ## calculate the sensitivity
    #scores["sensitivity"] = (TP / float(TP + FN))
    
    ## calculate the specificity
    #scores["specificity"] = (TN / float(TN + FP))
    
    ## calculate precision
    #scores["precision"] = (TN / float(TN + FP))
    
    #scores["recall"] = 1111;
    
    ## calculate f_1 score
    #scores["f1"] = 2 * ((scores["precision"] * scores["sensitivity"]) / (scores["precision"] + scores["sensitivity"]));
    
    #return scores;


# Function to easily call the model training process using the training validation and test set and get scores.
# "training_set":    The training set, which should also include the ground truth.
# "valid_set":       The validation set, which should also include the ground truth.
# "test_set":        The test set, which shouls also include the ground truth.
# "model":           The algorithm object returned by sklearn call.
# "input_vars":      The input variables to include in the prediction task, Only these variables will be used to perform the prediction tasks (any other variables will be ignored).
# "output_var":      The target variable for this task. For example, for sub-clinical mastitis prediction, use "early_detect_sub_clinical" as the target string.
# "early_stopping_rounds": Early stopping rounds in case of XGBoost is used. Is None, then early stopping will not be performed.
# "eval_metric":     The evaluation metric based on which early stopping is done. If "early_stopping_rounds" is None, then this string is ignored.
# "scaling_flag":    If to min-max scale the attributes given in  "input_vars".
# "sample_flag":     If to perform sampling to help with imbalance
# "calibrate_model": If to calibrate the model explicitely.
#
# Returns:
def train_test_models (train_set, valid_set, test_set, model, input_vars, output_var, early_stopping_rounds = 100, eval_metric = "auc", scaling_flag = False, sample_flag = False, calibrate_model = False):
  
  model_trained, scaler, thresh = train_model (train_set, valid_set, input_vars, output_var, model, scaling_flag = scaling_flag, sample_flag = sample_flag, calibrate_model = calibrate_model, early_stopping_rounds = early_stopping_rounds, eval_metric = eval_metric);
  return (predict_model (model_trained, thresh, test_set, input_vars, output_var, scaler));



# Function to easily call the model training process using the training validation and test set and get scores.
# "training_set":    The training set, which should also include the ground truth.
# "valid_set":       The validation set, which should also include the ground truth.
# "test_set":        The test set, which shouls also include the ground truth.
# "model":           The algorithm object returned by sklearn call.
# "input_vars":      The input variables to include in the prediction task, Only these variables will be used to perform the prediction tasks (any other variables will be ignored).
# "output_var":      The target variable for this task. For example, for sub-clinical mastitis prediction, use "early_detect_sub_clinical" as the target string.
# "early_stopping_rounds": Early stopping rounds in case of XGBoost is used. Is None, then early stopping will not be performed.
# "eval_metric":     The evaluation metric based on which early stopping is done. If "early_stopping_rounds" is None, then this string is ignored.
# "scaling_flag":    If to min-max scale the attributes given in  "input_vars".
# "sample_flag":     If to perform sampling to help with imbalance
# "calibrate_model": If to calibrate the model explicitely.
#
# Returns:
# "model_calibrated": The final trained model, which uses both training and validation set data.
# "scaler":           A scaler object if "scaling_flag" was True, otherwise an empty string "".
# "youden_prob":      The threshold based on the Youden's method computed based on the validation set.
def train_model (train_set, valid_set, input_vars, output_var, model, scaling_flag = False, sample_flag = False, calibrate_model = True, early_stopping_rounds = 30, eval_metric = "auc"):
  # model: the model object. like model = SVC ();

  # storing the part of the output details to be saved
  train_input  = train_set.loc[:,input_vars];
  train_output = train_set.loc[:,output_var];
  print (train_input.columns);
  if scaling_flag == True:
      logger.info ("Scailing training and test data");
      scaler = MinMaxScaler();
      scaler.fit (train_input);
      train_input = scaler.transform (train_input);
  else:
    scaler = "";
  
  logger.info ("Training set datapoints: " + str (train_input.shape[0]));
  logger.info ("Train output class counts: " + str (list (train_output.value_counts ())));
  
  val_input  = valid_set.loc[:,input_vars];
  val_output = valid_set.loc[:,output_var];
  
  logger.info ("Training date range: " + str (min (train_set.date)) + " - " + str (max (train_set.date)));
  logger.info ("Validation date range: " + str (min (valid_set.date)) + " - " + str (max (valid_set.date)));
  
  logger.info ("Training datapoints: " + str (train_input.shape[0]));
  logger.info ("Validation datapoints: " + str (val_input.shape[0]));
  
   #Fitting the model


  print (train_output.value_counts ());
  if (sample_flag == True):
    logger.info ("ADASYN oversample, train");
    #samp = ADASYN (random_state = 0, n_jobs = 8, sampling_strategy = {0: train_class_values[0] * 1, 1: train_class_values[1] * 4});
    #samp = TomekLinks (n_jobs = 8, sampling_strategy = "auto");
    counts = train_output.value_counts ();
    samp = NearMiss (sampling_strategy = {0: int (counts[0] / 2), 1: counts[1]}, n_neighbors = 3, version = 1);
    train_input_resamp, train_output_resamp = samp.fit_resample (train_input, train_output)
    train_input = train_input_resamp.copy ();
    train_output = train_output_resamp.copy ();
    del train_input_resamp;
    del train_output_resamp;

    counts = train_output.value_counts ();
    samp = ADASYN (random_state = 0, n_jobs = 8, sampling_strategy = {0: counts[0], 1: counts[1] * 2});
    train_input_resamp, train_output_resamp = samp.fit_resample (train_input, train_output)
    train_input = train_input_resamp.copy ();
    train_output = train_output_resamp.copy ();
    del train_input_resamp;
    del train_output_resamp;


  train_class_values = train_output.value_counts ();
  imb_ratio_train = (train_class_values[0]/train_class_values[1]) * 1.0;

  val_class_values = val_output.value_counts ();
  imb_ratio_val = (val_class_values[0]/val_class_values[1]) * 1.0;

  #imb_ratio_to_set = 1.0;
  imb_ratio_to_set = (imb_ratio_train + imb_ratio_val)/2;
  #imb_ratio_to_set = imb_ratio_val;
  logger.info ("Setting classifier weight for imbalance ratio: " + str (imb_ratio_to_set));
  setattr (model, "scale_pos_weight", imb_ratio_to_set);
  
  print (train_input.columns)
  
  #model_calibrated = CalibratedClassifierCV  (model, method = 'isotonic', cv = 5)
  
  logger.info("Fitting Model");
  print (train_output.value_counts ());
  print (val_input.shape);
  if early_stopping_rounds is not None:
    model.fit (train_input, train_output.ravel (), early_stopping_rounds = early_stopping_rounds, eval_metric = eval_metric, eval_set = [(val_input, val_output)], verbose = True);
  else:
    model.fit (train_input, train_output.ravel (), eval_metric = eval_metric, eval_set = [(val_input, val_output)], verbose = True);

  #model.fit (train_input, train_output.ravel (), eval_metric = eval_metric, eval_set = [(val_input, val_output)], verbose = True)

  #model_calibrated.fit (train_input, train_output.ravel ());
  #model.fit (train_input, train_output.ravel (), eval_metric = eval_metric, eval_set = [(val_input, val_output)], verbose = True)

  if calibrate_model == True:
    logger.info ("Calibrating Model using valid set");
    model_calibrated = CalibratedClassifierCV (model, cv = "prefit", method = "isotonic", n_jobs = -1);
    model_calibrated.fit (val_input, val_output);
  else:
    logger.info ("Not calibrating model");
    model_calibrated = model;
  
  #Creating TPR and FPR values for val set
  model_probs = model_calibrated.predict_proba (val_input)
  model_probs = model_probs[:, 1]
  fpr, tpr, thresholds = metrics.roc_curve (val_output, model_probs);
  youden_prob = cutoff_youdens_j (fpr, tpr, thresholds);

  # NOTE: Here we refit the models using all the data. The validation set is used to compute the threshold only
  # Best is crossvalidation
  if (sample_flag == True):
    logger.info ("ADASYN oversample, valid");
    #samp = ADASYN (random_state = 0, n_jobs = 8, sampling_strategy = {0: val_class_values[0] * 1, 1: val_class_values[1] * 4});
    #samp = TomekLinks (n_jobs = 8, sampling_strategy = "auto");
    counts = val_output.value_counts ();
    samp = NearMiss (sampling_strategy = {0: int (counts[0] / 2), 1: counts[1]}, n_neighbors = 3, version = 1);
    val_input_resamp, val_output_resamp = samp.fit_resample (val_input, val_output);
    val_input = val_input_resamp.copy ();
    val_output = val_output_resamp.copy ();
    del val_input_resamp;
    del val_output_resamp;

    counts = val_output.value_counts ();
    samp = ADASYN (random_state = 0, n_jobs = 8, sampling_strategy = {0: counts[0], 1: counts[1] * 2});
    val_input_resamp, val_output_resamp = samp.fit_resample (val_input, val_output)
    val_input = val_input_resamp.copy ();
    val_output = val_output_resamp.copy ();
    del val_input_resamp;
    del val_output_resamp;



  logger.info (f"Refitting Model Using train + valid and optimal n_estimators = {model.best_iteration}");
  setattr (model, "n_estimators", model.best_iteration);
  setattr (model, "scale_pos_weight", imb_ratio_to_set);
  model.fit (train_input.append (val_input), np.append (train_output, val_output), eval_metric = eval_metric, verbose = True);
  
  ## WARNING:
  if calibrate_model == True:
    logger.info ("Calibrating Model using valid set");
    model_calibrated = CalibratedClassifierCV (model, cv = "prefit", method = "isotonic", n_jobs = -1);
    model_calibrated.fit (val_input, val_output);
  else:
    logger.info ("Not calibrating model");
    model_calibrated = model;

  return (model_calibrated, scaler, youden_prob);


# Function to easily call the model training process using the training validation and test set and get scores.
# "model":             The trained model returned by "train_model" function (first returned object).
# "youden_prob":       The youden value returned by "train_model" function (third returned object).
# "test_set":          The test set, including the target column.
# "input_vars":        The list of input variables. Must match with the list which was passed to "train_model" function during training.
# "output_var:         The target variable name, using which the performance metrics will be computed.
# "scaler":            The scaler object (or empty string) returned by "train_model" (second returned object).|
# "keep_vars_in_pred": The list of variables which is to be kept in the returned data frame. This is for convenience. Just returning (date, id) is enough. Saving all the attributes saves time joining data from the test set for further analysis after the predictions are stored.
#
#
# Returns: (separate values returned to make be compatable with older code, interoperability)
# "auc":            AUC of test set.
# "bal_acc":        Balanced accuracy of test set.
# "geo_mean":       Geometric mean of test set.
# "sensitivity":    Sensitivity of test set.
# "specificity":    Specificity of test set.
# "precision":      Precision of test set.
# "recall":         Recall of test set.
# "youden":         Not used, ignore this value.
# "youden_matrix":  Confusion matrix generated after using the youden value to the predicted probabilities.
# "model":          This is same as the "model" in the function argument.
# "scaler":         This is same as the "scaler" in the function argument.
# "youden_prob":    This is same as the "youden_prob" in the function argument.
# "pred_data":      This the prediciton data frame. The prediction score is in "pred_prob" and the thresholded binary predictions are in the columns, "pred_thresh"
#                   Along with this, the other variables in "keep_vars_in_pred" are also added to the dataframe. Therefore, use "pred_thresh" as the final predicted
#                   binary prediction.
#
def predict_model (model, youden_prob, test_set, input_vars, output_var, scaler, keep_vars_in_pred = ["date","id", "sub_clin", "clin", "days_since_calving", "time_till_inf_sub_clin", "time_till_inf_clin", "parity"                     ,
                 "PTA_mean"                   , "infections_per_parity_sub_clin",
                 "infections_per_cow_sub_clin", "month_of_milking","1_fat"     , "2_fat"    , "1_yield"  , "2_yield" , "1_max_flow",
                 "2_max_flow", "1_protein", "2_protein", "1_urea"  , "2_urea"    ,
                 "1_lactose" , "2_lactose", "1_scc"    , "2_weight", "2_BCS", "time_till_inf_sub_clin_0days", "1_scc_0days", "sub_clin_0days"]):
    # WARNING and TODO: When we have changed data processing script to store the data of 0days, also save time_till_inf_clin_0days as well
  
  keep_vars_in_pred.append (output_var);

  # For compatibility
  youden  = np.array([]);
  
  test_input  = test_set.loc[:,input_vars];
  test_output = test_set.loc[:,output_var];
  
  
  if scaler != "":
    logger.info ("Scailing training and test data");
    test_input = scaler.transform (test_input);
  else:
    logger.info ("Not scaling");
    
  
  # storing the part of the output details to be saved
  pred_data = test_set.loc[:,keep_vars_in_pred].copy ();

  #Creating TPR and FPR values for test set
  model_probs = model.predict_proba (test_input)
  model_probs = model_probs[:, 1]
  fpr, tpr, thresholds = metrics.roc_curve (test_output, model_probs)

  #Creating AUC
  auc = metrics.auc (fpr, tpr)

  #Calculating youden_index
  conditions_matrix = [(model_probs) <= youden_prob, 
                      (model_probs) > youden_prob]
  choices_matrix = [0,1]
  matrix_best_representation = np.select (conditions_matrix, choices_matrix)

  #Creating classification matrix at youden value to account for unequal cost values
  youden_matrix = metrics.confusion_matrix (test_output, matrix_best_representation)

  #Creating Sensitivity and Specificity
  sensitivity = (youden_matrix[1,1])/(youden_matrix[1,1] + youden_matrix[1,0])
  specificity = (youden_matrix[0,0])/(youden_matrix[0,0] + youden_matrix[0,1])

  #Creating Precision and recall
  precision = (youden_matrix[1,1]/ (youden_matrix[1,1] + youden_matrix[0,1]))
  recall = sensitivity
  geo_mean = np.sqrt(specificity*sensitivity)
  bal_acc = (sensitivity + specificity)/2
  
  print (test_input.shape);
  print (model_probs.shape)
  thresh_pred = pd.DataFrame ( (model_probs > youden_prob)).reset_index (drop = True);
  print (thresh_pred.shape);
  # output details
  #pred_data = pd.concat ([pred_data.reset_index (drop = True), pd.DataFrame (model_probs), thresh_pred, pd.DataFrame (test_output).reset_index (drop = True)], axis = 1)
  pred_data.reset_index (drop = True, inplace = True);
  pred_data = pd.concat ([pred_data, pd.DataFrame (model_probs), thresh_pred], axis = 1)
  print (pred_data.shape);
  keep_vars_in_pred.append ("pred_prob");
  keep_vars_in_pred.append ("pred_thresh");
  pred_data.columns = keep_vars_in_pred;
  
  return (auc, bal_acc, geo_mean, sensitivity, specificity, precision, recall, youden, youden_matrix, model, scaler, youden_prob, pred_data);


# NOTE: This function is not in use as it is an oldeer version
#def early_detect_by_days_subclin (model_probs, output_var, use_tuned_threshold = True):
  
  #infection_before = False;
  ##model_probs = pd.read_csv (f"{file_prefix}_{this_day}_pred_info.csv");
  #if use_tuned_threshold == True:
    #model_probs["pred_infected"] = model_probs["pred_thresh"];
  #else:
    #model_probs["pred_infected"] = model_probs["pred_prob"] >= 0.5;
  
  #if output_var == "early_detect_sub_clinical":
      #time_count_var = "time_till_inf_sub_clin_0days";
  #elif output_var == "early_detect_clinical":
      #time_count_var = "time_till_inf_clin_0days";

  ## NOTE: RECHECK, an infection is counted for the same cow if they are outside the days of calving range
  ##model_probs.reset_index (drop = True, inplace = True);
  #model_probs['Group_ID'] = model_probs.groupby('id').grouper.group_info[0];
  #value_inf = 0;
  #model_probs['amount_inf'] = np.nan;
  #inf_indices = [];
  #for row in tqdm (range (len (model_probs))):
      ##Check for infection
      ## NOTE: 'time_till_inf_sub_clin' or the time_count_var variable should be based on the "0" day freq
      #if model_probs.iloc[row][output_var] == 1: # Ground truth
          #if infection_before == False:
              #allowed = model_probs.iloc[row]['days_since_calving'] + model_probs.iloc[row][time_count_var];
              #infection_before = True;
              #value_inf += 1;
              #cow = model_probs.iloc[row]['Group_ID'];
          ##Check the same cow
          #elif model_probs.iloc[row]['Group_ID'] != cow:
              #value_inf += 1;
              #cow = model_probs.iloc[row]['Group_ID'];
              #allowed = model_probs.iloc[row]['days_since_calving'] + model_probs.iloc[row][time_count_var];
  ##             print('made_it')
          #elif model_probs.iloc[row]['Group_ID'] == cow:
              ##Check if its within the allowed distance
              #if model_probs.iloc[row]['days_since_calving'] > allowed + 1:
                  #allowed = model_probs.iloc[row]['days_since_calving'] + model_probs.iloc[row][time_count_var];
                  #value_inf += 1;
  ##             print('Allowed value = {}'.format(allowed))
  ##             print('Day since calving = {}'.format(model_probs.iloc[row]['days_since_calving']))
          #inf_indices.append(value_inf);
  ##             print(value_inf)
      #else:
          #inf_indices.append(0);
          
  #model_probs['amount_inf'] = inf_indices;

  #######
  
  #milked_so_far = [];
  ##Subsetting on observations 7 days or less before an infection
  ##subset_infected = model_probs[ model_probs["amount_inf"] != 0];
  #subset_infected = model_probs;
  ##subset_infected.reset_index (drop = True, inplace = True);

  #days = [7, 6, 5, 4, 3, 2, 1];

  #correct_inf = np.array ([]);
  #proportions = np.array ([]);

  #for day in range (len (days)):
      #tmpdf = subset_infected[subset_infected[time_count_var] == days[day]].copy ();
      
      #milked_so_far = np.append (milked_so_far, tmpdf.amount_inf.unique());
      
      #correctly = tmpdf[tmpdf['pred_infected'] == 1].copy ();
      #correct_inf = np.append(correct_inf, correctly.amount_inf.unique());

      #print('There is a total of {} infections milked atleast {} days early'.format(len(np.unique(milked_so_far)), days[day]));
      #print('There was a total of {} infections correctly classified {} days early'.format(len(np.unique(correct_inf)), days[day]));
      #print('There was a total of {} infections incorrectly classified {} days early'.format(len(np.unique(milked_so_far)) - len(np.unique(correct_inf)), days[day]));
      #print(f'{days[day]}');
      #print('Proportion of correct classifications = {}'.format(len(np.unique(correct_inf))/len(np.unique(milked_so_far))));
      #proportions = np.append(proportions, len(np.unique(correct_inf))/len(np.unique(milked_so_far)));
  
  #return (proportions);

def early_detect_by_days_subclin (model_probs, output_var, use_tuned_threshold = True, top_n = 1.0):

  if use_tuned_threshold == True:
    model_probs["pred_infected"] = model_probs["pred_thresh"];
  else:
    model_probs["pred_infected"] = model_probs["pred_prob"] >= 0.5;

  if output_var == "early_detect_sub_clinical":
      time_count_var = "time_till_inf_sub_clin_0days";
  elif output_var == "early_detect_clinical":
      time_count_var = "time_till_inf_clin_0days";


  # Sort by probability
  #model_probs[time_count_var] = model_probs[time_count_var].fillna (0);
  model_probs = model_probs.sort_values (by = "pred_prob", inplace = False, ascending = False);
  nrows = int (np.round (model_probs.shape[0] * top_n));
  if nrows == 0:
     return None;

  # Select top_n percent rows based on the probability
  model_probs = model_probs.iloc[range(nrows),:];

  # Make sure that the `date' field is in datetime format
  model_probs["date"] = pd.to_datetime (model_probs["date"]);

  # Sort back the top_n percent rows in id, date order (original order)
  model_probs = model_probs.sort_values (by = ["id", "date"], inplace =  False, ascending = True);

  infection_before = False;

  cow_early_list = {};
  all_cows = pd.DataFrame (np.zeros ((7, 2)));
  all_cows.index = [1, 2, 3, 4, 5, 6, 7];
  all_cows.columns = [True, False];

  inf_template = pd.DataFrame (np.zeros ((7, 2)));
  inf_template.index = all_cows.index;
  inf_template.columns = [time_count_var, "pred_infected"];
  inf_tot = inf_template.copy ();
  total_infections = 0;
  for key, row in tqdm (model_probs.groupby ("id")):
      #sick_rows = row[row[output_var]==1];
      sick_rows = row;
      if sick_rows.shape[0] == 0:
          continue;

      this_cow_all_inf = pd.DataFrame ();
      # Find a block of infection and then copy it over to the template
      inf_start = False;
      #print (key)
      last_date = sick_rows.iloc[0]["date"] - pd.to_timedelta(1);
      days_count_down = 0;
      for i in range (sick_rows.shape[0]):
          if inf_start == False and sick_rows.iloc[i][output_var] == True and 0 < sick_rows.iloc[i][time_count_var] <= 7: # WARNING: Now if the first record?
              inf_start = True;
              this_inf = inf_template.copy ();
              days_count_down = sick_rows.iloc[i][time_count_var];
              last_day_idx = i;
              inf_first_detect = False;
              total_infections += 1;
              #print ("Sick start", sick_rows.iloc[i]["date"]);

          if inf_start == True:

              #print (days_count_down)
              # If a milking was done in this day then we will "time_till_inf_sub_clin" value equal to our count down counter
              # So, make this row the one to propate to the next missing milking day
              if sick_rows.iloc[i][time_count_var] == days_count_down:
                  last_day_idx = i;

              inf_first_detect = inf_first_detect or bool (sick_rows.iloc[last_day_idx]["pred_infected"]);
              #inf_first_detect = bool (sick_rows.iloc[last_day_idx]["pred_infected"]);

              # Copy the infection prediction from the last milking day
              this_inf.loc[days_count_down,"pred_infected"] = inf_first_detect;
              this_inf.loc[days_count_down,time_count_var] = days_count_down;
              #print (days_count_down)
              days_count_down -= 1;

              # Reset infection
              if np.isnan (sick_rows.iloc[i][time_count_var]) or sick_rows.iloc[i][time_count_var] > days_count_down + 1 or (sick_rows.iloc[i]["date"] - last_date).days > 7 or i == sick_rows.shape[0] - 1:
                  #print ("Sick end", sick_rows.iloc[i]["date"]);
                  inf_start = False;
                  while days_count_down > 0:
                      this_inf.loc[days_count_down,"pred_infected"] = inf_first_detect;
                      this_inf.loc[days_count_down,time_count_var] = days_count_down;
                      days_count_down -= 1;

                  this_cow_all_inf = this_cow_all_inf.append (this_inf);
                  inf_tot =  inf_tot + this_inf;
                  #print (this_cow_all_inf)
                  #print (this_inf);
                  #print ("\n\n\n\n")

          last_date = sick_rows.iloc[i]["date"];


      if this_cow_all_inf.shape[0] == 0:
          continue;
      #print (this_cow_all_inf);
      #print (this_inf)
      #sick_early = pd.Categorical (sick_rows[time_count_var].astype (int), categories = [1, 2, 3, 4, 5, 6, 7]);
      #sick_predict = pd.Categorical (sick_rows["pred_infected"], categories = [True, False]);
      sick_early = pd.Categorical (this_cow_all_inf[time_count_var].astype (int), categories = [1, 2, 3, 4, 5, 6, 7]);
      sick_predict = pd.Categorical (this_cow_all_inf["pred_infected"], categories = [True, False]);
      this_cow = pd.crosstab (sick_early, sick_predict, dropna = False);
      cow_early_list[key] = this_cow;
      #print (all_cows);
      #print (this_cow);
      all_cows += this_cow;
      #print (all_cows);
  #proportions = np.flip (np.array (all_cows.apply (lambda x: x / np.sum (all_cows.iloc[0,:]), axis = 1).iloc[1:8,0]));
  proportions = np.array (all_cows.apply (lambda x: x / np.sum (all_cows.iloc[0,:]), axis = 1).iloc[:,0]);


  res = {};
  for key, frame in tqdm (pred_info.groupby (time_count_var)):
      tmp = frame[frame[output_var] == 1];
      cnt = tmp["pred_infected"].value_counts ();
      res[key] = cnt;

  pos_per = [0] * 8;
  i = 0;
  for this_key in res.keys ():
      #a = ((res[int(this_key)] / res[int(this_key)].sum ()) * 100);
      a = ((res[int(this_key)] / total_infections) * 100);
      print (res[int(this_key)].sum ());
      pos_per[i] = a[1];
      i = i + 1;
      if i >= 8:
          break;

  pos_per.pop (0);

  return ({"all_cows": all_cows, "proportions": proportions, "per_cow": cow_early_list, "total_infections": total_infections, "pos_per": pos_per, "inf_tot": inf_tot});


def xgboost_hyper_param (**kwargs):

    global train_input;
    global train_output;
    global val_input;
    global val_output;
    
    kwargs["max_depth"]    = int (round (kwargs["max_depth"]));
    kwargs["n_estimators"] = int (round (kwargs["n_estimators"]));
    
    train_class_values = train_output.value_counts ();
    
    args = kwargs.copy ();
    args["tree_method"] = "gpu_hist";
    args["objective"] = "binary:logistic";
    args["eval_metric"] = "aucpr";
    args["use_label_encoder"] = False;

    clf = XGBClassifier (**args);

    train_class_values = train_output.value_counts ();
    imb_ratio_train = (train_class_values[0]/train_class_values[1]) * 1.0;

    #val_class_values = val_output.value_counts ();
    #imb_ratio_val = (val_class_values[0]/val_class_values[1]) * 1.0;
    setattr (clf, "scale_pos_weight", kwargs["imb_ratio"]);
    
    clf.fit (train_input, train_output, eval_metric = "aucpr", verbose = True);
    
    ##if kwargs["model_calibrated"] == True:
    #model_calibrated = CalibratedClassifierCV (clf, cv = 3, n_jobs = -1);
    #model_calibrated.fit (train_input, train_output);
    #model_calibrated.fit (val_input, val_output); # This to be used with cv = "prefit"
    ##else:
      ##logger.info ("Not calibrating model");
      ##model_calibrated = model;
  
    pred = clf.predict_proba (val_input);
    pred = pred[:, 1]
    #fpr, tpr, thresholds = metrics.roc_curve (val_output, pred);
    fpr, tpr, thresholds = precision_recall_curve (val_output, pred);
    #auc = metrics.auc (fpr, tpr);

    thresh = cutoff_youdens_j (fpr, tpr, thresholds);

    bacc = balanced_accuracy_score (val_output, (pred > thresh) * 1);
    geom = geometric_mean_score (val_output, (pred > thresh) * 1);
    return geom;
    #return bacc;
    #return auc;

def xgboost_threshold_tune (**kwargs):
  
    global train_input;
    global train_output;
    global val_input;
    global val_output;
   
    args = {"colsample_bytree": 0.614496534265229, "gamma": 0.777427232118818, "learning_rate": 0.5210108354831031, "max_depth": 2, "n_estimators": 458, "subsample": 0.9989468685041512, "tree_method": "hist"};
    clf = XGBClassifier (**args);
    setattr (clf, "scale_pos_weight", kwargs["imb_ratio"]);
    
    clf.fit (train_input, train_output, eval_metric = "auc", verbose = True);
    model_calibrated = CalibratedClassifierCV (clf, cv = "prefit");
    model_calibrated.fit (val_input, val_output);

  
    pred = clf.predict_proba (val_input);
    pred = pred[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve (val_output, pred);
    #auc = metrics.auc (fpr, tpr);

    thresh = cutoff_youdens_j (fpr, tpr, thresholds);

    bacc = balanced_accuracy_score (val_output, (pred > thresh) * 1);
    geom = geometric_mean_score (val_output, (pred > thresh) * 1);
    return geom;
    #return bacc;
    #return auc;
  
def xgboost_var_select (**kwargs):
  
    global train_input;
    global train_output;
    global val_input;
    global val_output;
    global test_input;
    global test_output;
    
    kwargs["max_depth"]    = int (round (kwargs["max_depth"]));
    kwargs["n_estimators"] = int (round (kwargs["n_estimators"]));
    
    train_class_values = train_output.value_counts ();
    
    clf = XGBClassifier(**kwargs);
    setattr (clf, "scale_pos_weight", train_class_values[0]/train_class_values[1]);

    var_select_score = [];
    var_names = train_input.columns;
    for i in range (1, len (var_names) + 1):
      logger.info (f"Training iter {i} of {len (var_names)}");
      print (var_names[range (i)]);
      # clf.fit (train_input[var_names[range (i)]], train_output, eval_metric = "auc", verbose = True);
      clf.fit (train_input[var_names[range (i)]].append (val_input[var_names[range (i)]]), np.append (train_output, val_output), eval_metric = "auc", verbose = True);
      # pred = clf.predict_proba (val_input[var_names[range (i)]]);
      pred = clf.predict_proba (test_input[var_names[range (i)]]);
      pred = pred[:, 1]
      # fpr, tpr, thresholds = metrics.roc_curve (val_output, pred);
      fpr, tpr, thresholds = metrics.roc_curve (test_output, pred);
      auc = metrics.auc (fpr, tpr);
      var_select_score.append (auc);
      print (f"AUC = {round (auc, 4)}");
      np.savetxt (cfg["result_dir"] + "/subclin_0_var_select_auc.csv", var_select_score);
      
    return var_select_score;

# This function loads the dataset generated by "prepare_data.py"
def load_dataset (data_path, data_file, target_file, input_vars, output_var):

  data_path = Path (data_path);
  tmp_df = pd.read_csv (data_path / data_file);
  
  logger.info (f"Reset index of dataframe");
  tmp_df.reset_index (drop = True, inplace = True);

  logger.info (f"Convert date format");
  tmp_df.date = pd.to_datetime (tmp_df.date);
  tmp_df.calving_x = pd.to_datetime (tmp_df.calving_x);
  tmp_df.calving_y = pd.to_datetime (tmp_df.calving_y);

  logger.info (f"Adding month_of_milking column");
  tmp_df["month_of_milking"] = tmp_df.date.dt.month;

  logger.info ("Loading targets");
  targets = pd.read_csv (data_path / target_file);

  logger.info ("Adding target variable to data frame");
  tmp_df[output_var] = targets[output_var];
  
  ## NOTE: We remove this as we have removed in the past.
  logger.info ("Removing duplicate rows");
  print (tmp_df.shape);
  tmp_df.drop_duplicates (subset = ["id", "parity", "date"], inplace = True);
  print (tmp_df.shape);
  
  
  return tmp_df;



####
# remove_list  : Remove the variables from the remove_list from the set hardcoded in the function.
# remove_all   : remove all variables which has a substring in remove_list. This means if 
#                remove_list = ["scc", "yield"], then it will remove all variables which has any of these as substrings.
# var_list_file: A file with comma separated variable names to be read and returned. First this file will be read
#                and then the "remove_list" will be applied.
#
# return       : Returns a list of variables.
####
def get_input_variable_list (remove_list = [], remove_all = True, var_list_file = None):

  basic_vars = ["1_scc", "1_fat"     , "2_fat"    , "1_yield"  , "2_yield" , "1_max_flow",
                 "2_max_flow", "1_protein", "2_protein", "1_urea"  , "2_urea"    ,
                 "1_lactose" , "2_lactose", "2_weight", "2_BCS"         ];
  
  # If a "var_list_file" is provided then load the variable list.
  if var_list_file is not None:
    if os.file.isfile (var_list_file):
      var_list = np.genfromtxt(var_list_file, delimiter = ',');
    elif type (var_list_file) == list:
      var_list = var_list_file;
    else:
      logger.error (f"Input variable list file \"{var_list_file}\" not present, neither it is a list of variables");
      raise Exception ("File not found");
  else:
    # Otherwise return the hardcoded variable list
    var_list = [];
    # var_list.extend (basic_vars);
    for item in basic_vars:
        var_list.append (item);
        for feature in ["max", "min", "std", "skew", "mean", "med", "delta"]:
            for time in ["15", "30"]:
                    var_list.append(feature + "_" + item + "_last_" + time + "_days");
                    
    for item in ["parity"                     , "days_since_calving"            , 
                 "PTA_mean"                   , "infections_per_parity_sub_clin", 
                 "infections_per_cow_sub_clin", "month_of_milking"]:
        var_list.append(item);
  
  # Remove the variables in "remove_list" and remove even if a substring matches itss
  idx_remove = [];
  final_var_list = [];
  for i in range (len (var_list)):
    removed = False;
    if var_list[i] in remove_list:
        idx_remove.append (i);
        removed = True;
    elif remove_all == True:
        for x in remove_list:
            if x in var_list[i]:
                idx_remove.append (i);
                removed = True;
                break;
    if removed == False:
        final_var_list.append (var_list[i]);

  return final_var_list;


def get_output_variable (target_type = "sub-clinical"):
  
  if target_type == "sub-clinical":
    return ["early_detect_sub_clinical", "unsure_sub_clin", "time_till_inf_sub_clin_0days", "1_scc_0days", "sub_clin_0days"];
  
  if target_type == "clinical":
    return ["early_detect_clinical", "unsure_sub_clin"]; # TODO and WARNING time_till_inf_clin_0days in the frame when we make the patch

def split_train_valid_test (df, train_start, train_end, valid_start, valid_end, test_start, test_end):
  pass;
  
  
###########################
# If not set, take the default one.
try:
    cfg_file_name;
except NameError:
    cfg_file_name = "model_train_cfg.json";
    logger.info (f"Loading default configuration file: {cfg_file_name}");

# Load the configuration file
logger.info ("Loading configuration file");
cfg = load_cfg (cfg_file_name);
print (cfg);

# TODO: Make a random name in this directory so the results are saved automatically

# Determine saved file prefix
file_prefix = "";
if cfg["target_type"] == "clinical":
    file_prefix = "clin";
elif cfg["target_type"] == "sub-clinical":
    file_prefix = "subclin";


# Below block is when we just want to load the functions etc
if cfg["source_only"] == False:

  # NOTE: first the data set is split in train-test using "train_test_date_split"
  # next, the train porting is split in train-valid using "train_valid_date_split"
  
  if cfg["load_data_flag"] == True:# or cfg["model_train_flag"] == True:

    data_file    = "all_features.csv.gz" if this_day == 0 else f"all_features_{this_day}.csv.gz";
    target_file  = "targets.csv.gz";
    
    logger.info ("Getting input variable list");
    remove_list = [] if "remove_list" not in cfg else cfg["remove_list"];
    input_vars = get_input_variable_list (remove_list = remove_list);

    logger.info ("Getting output variable");
    output_var      = get_output_variable (cfg["target_type"]);
    output_var_name = output_var[0]; # Not elegant at all

    logger.info (f"Loading training datafile for {this_day} days");
    df = load_dataset (cfg["train_data_path"], data_file, target_file, input_vars, output_var);
    if os.path.isdir (cfg["test_data_path"]):
      logger.info (f"Loading testing datafile for {this_day} days");
      df_test = load_dataset (cfg["test_data_path"], data_file, target_file, input_vars, output_var);
    else:
      logger.info (f"No testing datafile provided for {this_day} days");
      
      
    # NOTE: Useful when making a full and final model, or when we want to filter out a specific
    # range of dates from the entire dataset
    if cfg["train_test_merge"] == True:
      logger.info (f"Merging \"training\" and \"test\" datasets");
      df = pd.concat ([df, df_test], axis = 0, ignore_index = True);
      del df_test;
    else:
      logger.info (f"Not merging \"training\" and \"test\" datasets");
      
      

    # NOTE: After this point we will extract the train and test ones.
    split_date = cfg["train_test_date_split"]
    logger.info (f"Partitioning train-test sets. Training: calving date < {split_date}, Testing: calving date >= {split_date}");
    train_idx = df.calving_x.dt.year <  pd.to_datetime (cfg["train_test_date_split"]).year;
    test_idx  = df.calving_x.dt.year >= pd.to_datetime (cfg["train_test_date_split"]).year;
    
    #logger.info (f"Partitioning train-test sets. Training: milking date < {split_date}, Testing: milking date >= {split_date}");
    #train_idx = df.date <  pd.to_datetime (cfg["train_test_date_split"]);
    #test_idx  = df.date >= pd.to_datetime (cfg["train_test_date_split"]);
    
    if (not os.path.isdir (cfg["test_data_path"])) or (os.path.isdir (cfg["test_data_path"]) and cfg["train_test_merge"] == True):
      logger.info (f"Partitioning train - test");
      df_test = df.loc[test_idx,:].copy ();
      df      = df.loc[train_idx,:].copy ();
    

    logger.info (f"Imputing NA values");
    #skew_vars = [x for x in columns if "skew" in x];
    #df = df[skew_vars].fillna (0);
    train_time_till_inf_sub_clin = df.time_till_inf_sub_clin;
    test_time_till_inf_sub_clin = df_test.time_till_inf_sub_clin;

    train_time_healthy_sub_clin = df.time_healthy_sub_clin;
    test_time_healthy_sub_clin = df_test.time_healthy_sub_clin;

    
    df = df.fillna (0);
    df_test = df_test.fillna (0); # Need to do for test as well
    
    df["time_till_inf_sub_clin"] = train_time_till_inf_sub_clin;
    df_test["time_till_inf_sub_clin"] = test_time_till_inf_sub_clin;

    df["time_healthy_sub_clin"] = train_time_healthy_sub_clin;
    df_test["time_healthy_sub_clin"] = test_time_healthy_sub_clin;


    #cond_after = [df["time_healthy_sub_clin"] <= 7, df["time_healthy_sub_clin"] > 7];
    #choices_after = [1,0];
    #df["unsure_sub_clin"] = np.select (cond_after, choices_after, default = 0);

    #cond_after = [df_test["time_healthy_sub_clin"] <= 7, df_test["time_healthy_sub_clin"] > 7];
    #choices_after = [1,0];
    #df_test["unsure_sub_clin"] = np.select (cond_after, choices_after, default = 0);

    # Load the above data block first, so we have the training partititon in df.
    #train_idx = df.date  < pd.to_datetime (cfg["train_valid_date_split"]);
    #valid_idx = df.date >= pd.to_datetime (cfg["train_valid_date_split"]);

    train_idx = df.calving_x.dt.year  < pd.to_datetime (cfg["train_valid_date_split"]).year;
    valid_idx = df.calving_x.dt.year >= pd.to_datetime (cfg["train_valid_date_split"]).year;
    
    logger.info (f"Partitioning train - valid");
    df_valid  = df.loc[valid_idx,:].copy ();
    df_tmp    = df.loc[train_idx,:].copy ();
    df        = df_tmp;
    del df_tmp;


    logger.info (f"Removing unsure subclinical from training instances"); # ONLY for train.
    df       = df[df["unsure_sub_clin"] == 0];
    df_valid = df_valid[df_valid["unsure_sub_clin"] == 0];
    # df_test  = df_test[df_test['unsure_sub_clin'] == 0];

    logger.info (f"Train rows before 10 days before calving removal {df.shape[0]}");
    logger.info (f"Valid rows before 10 days before calving removal {df_valid.shape[0]}");
    logger.info (f"Test rows before 10 days before calving removal {df_test.shape[0]}");


    logger.info (f"Removing data with days_since_calving <= 10 from training");
    df       = df[df["days_since_calving"] > 10];
    df_valid = df_valid[df_valid["days_since_calving"] > 10];
    df_test  = df_test[df_test['days_since_calving'] > 10];


    logger.info (f"Train rows after 10 days before calving removal {df.shape[0]}");
    logger.info (f"Valid rows after 10 days before calving removal {df_valid.shape[0]}");
    logger.info (f"Test rows after 10 days before calving removal {df_test.shape[0]}");



    # At this point we have the training set and test sets split
    if "remove_calving_months" in cfg:
      calving_months = cfg["remove_calving_months"];
      logger.info (f"Removing entries having months {calving_months} of calving cows from training set");
      to_remove_idx = df.calving_x.dt.month.isin (cfg["remove_calving_months"]);
      print (to_remove_idx.value_counts ());
      df = df[~to_remove_idx];
      
      to_remove_idx = df_valid.calving_x.dt.month.isin (cfg["remove_calving_months"]);
      print (to_remove_idx.value_counts ());
      df_valid = df_valid[~to_remove_idx];
        
      to_remove_idx = df_test.calving_x.dt.month.isin (cfg["remove_calving_months"]);
      print (to_remove_idx.value_counts ());
      df_test = df_test[~to_remove_idx];
      
    else:
      logger.info (f"Not removing cows based on calving dates from training set");


    logger.info (f"Train rows keeping only spring cows {df.shape[0]}");
    logger.info (f"Valid rows keeping only spring cows {df_valid.shape[0]}");
    logger.info (f"Test rows keeping only spring cows {df_test.shape[0]}");

      
    logger.info ("Training milking date range: " + str (min (df.date)) + " - " + str (max (df.date)));
    logger.info ("Validation milking date range: " + str (min (df_valid.date)) + " - " + str (max (df_valid.date)));
    logger.info ("Test milking date range: " + str (min (df_test.date)) + " - " + str (max (df_test.date)));
    
    
    logger.info ("Training calving date range: " + str (min (df.calving_x)) + " - " + str (max (df.calving_x)));
    logger.info ("Validation calving date range: " + str (min (df_valid.calving_x)) + " - " + str (max (df_valid.calving_x)));
    logger.info ("Test calving date range: " + str (min (df_test.calving_x)) + " - " + str (max (df_test.calving_x)));
    
      
  # TODO: Pack in a function, make general code to handle any date
  # port the time slicing code I wrote from R
  if cfg["hyper_param_tune_flag"] == True:
    
    if cfg["target_type"] == "sub-clinical":
      y_var = "early_detect_sub_clinical";
    elif cfg["target_type"] == "clinical":
      y_var = "early_detect_clinical";
    
    
    # NOTE: `val_input', `val_output', `train_input', `train_output' is used as global in `xgboost_hyper_param' function
    val_input  = df_valid.loc[:,input_vars];
    val_output = df_valid.loc[:,y_var];
    
    train_input  = df.loc[:,input_vars];
    train_output = df.loc[:,y_var];

    test_input  = df_test.loc[:,input_vars];
    test_output = df_test.loc[:,y_var];
    
    
    # TODO: Read from config file
    pbounds = {
      'learning_rate': (0.005, 1.0),
      'n_estimators': (200, 200),
      'max_depth': (1, 6),
      'subsample': (0.5, 1.0),  # Change for big datasets
      'colsample_bytree': (0.5, 1.0),  # Change for datasets with lots of features
      'gamma': (1, 5),
      'imb_ratio': (2, 50)}
    
    optimizer = BayesianOptimization (f = xgboost_hyper_param, pbounds = pbounds);

    
    target_name   = cfg["target_type"];
    log_file_path = f"./{target_name}_bayesianopt_log.json" if this_day == 0 else f"./{target_name}_bayesianopt_log_{this_day}.json";
    if cfg["hyper_param_load_log"] == True and os.path.isfile (log_file_path):
      load_logs (optimizer, logs = [log_file_path]);
    
    logger = JSONLogger (path = log_file_path);
    optimizer.subscribe (Events.OPTIMIZATION_STEP, logger)
    
    # TODO: Read from config file
    bayesian_hopt_hyperparams = {"init_points": 10, "n_iter": 40};
    optimizer.maximize (**bayesian_hopt_hyperparams);
    
    #del val_input;
    #del val_output;
    #del train_input;
    #del train_output;
    

  if cfg["model_train_flag"] == True:

    logger.info ("Configuring classifier");
    
    # DONE: hyper param 1
    # Found for sub-clinical
    #hyper_params = load_cfg (cfg["hyper_param_file"]);
    #hyper_params = {"colsample_bytree": 0.5, "gamma": 1.2532246033623349, "learning_rate": 0.005, "max_depth": 6, "n_estimators": 2000, "subsample": 1.0, "tree_method": "hist", "use_label_encoder": False, "objective": "binary:logistic", "eval_metric": "auc"}; # "gpu_hist" to run in GPU

    hyper_params = cfg["hyper_param_file"];
    logger.info (f"Loading hyperparameter configuration from {hyper_params}");
    hyper_params = read_hyper_param (hyper_params);
    print (hyper_params);

    # Found for clinical. OLD VALUE. Use, "hyper_param_file" from now on.
    #hyper_params = {"colsample_bytree": 0.6930703897317563, "gamma": 4.900021911549661, "learning_rate": 0.3461493165332064, "max_depth": 1, "n_estimators": 271, "subsample": 0.9872217000953398, "tree_method": "gpu_hist", "use_label_encoder": False}
    
    # Load the input variables again, in case just retrain and do not reload the dataset.
    input_vars = get_input_variable_list (remove_list = remove_list);
    print (input_vars);

    classifier =  XGBClassifier (**hyper_params);

    AUC_m, BAL_m, Geo_m, Sens_m, Spec_m, Prec_m, Recall_m, youden, matrix, model, scaler, thresh, pred_info = \
      train_test_models (train_set = df, valid_set = df_valid, test_set = df_test, model = classifier, input_vars = input_vars, calibrate_model = cfg["model_calibrate"], early_stopping_rounds = 100, eval_metric = "auc", scaling_flag = False, sample_flag = cfg["sample_data_flag"], output_var = output_var_name);
    
    save_dir = cfg["result_dir"];

    joblib.dump (model, f"{save_dir}/{file_prefix}_{str (this_day)}_days_model.joblib");
    joblib.dump (scaler, f"{save_dir}/{file_prefix}_{str (this_day)}_days_scaler.joblib");
    np.savetxt (f"{save_dir}/{file_prefix}_thresh_{str (this_day)}_days.csv", [thresh]);
    pred_info.to_csv (f"{save_dir}/{file_prefix}_{str (this_day)}_pred_info.csv", index = False);
    #save_cfg (hyper_params, f"{save_dir}/{file_prefix}_{str (this_day)}_hyperparams.json");
    #save_cfg (cfg, f"{save_dir}/{file_prefix}_{str (this_day)}_cfg.json"); #TODO: save, PosixPath not serialilsable


    print('AUC = {:.4f}'.format(AUC_m))
    print('Balanced Accuracy = {:.4f}'.format(BAL_m))
    print('Geometric Mean = {:.4f}'.format(Geo_m))
    print('Specificity = {:.4f}'.format(Spec_m))
    print('Sensitivity = {:.4f}'.format(Sens_m))
    print('Recall = {:.4f}'.format(Recall_m))
    print('Precision = {:.4f}'.format(Prec_m))
    print('Youden value = {:.4f}'.format(thresh))
    print('Matrix at youden value = \n {}'.format(matrix))

  # NOTE: Experimental
  if "automl" in cfg and cfg["automl"] == True:
      h2o.init ();

      df       = h2o.H2OFrame (df);
      df_valid = h2o.H2OFrame (df_valid);
      df_test  = h2o.H2OFrame (df_test);

      y = "early_detect_sub_clinical";
      x = get_input_variable_list();

      df[y] = df[y].asfactor ();
      df_valid[y] = df_valid[y].asfactor ();
      df_test[y] = df_test[y].asfactor ();

      #aml = H2OAutoML (max_models = 20, seed = 1, stopping_metric = "AUC", include_algos = ["GLM", "DeepLearning", "DRF", "XGBoost"], nfolds = 0);
      aml = H2OAutoML (max_models = 20, seed = 1, stopping_metric = "AUC", include_algos = ["DeepLearning"], nfolds = 0, balance_classes = True, max_after_balance_size = 0.2);

      aml.train (x = x, y = y, training_frame = df, validation_frame = df_valid);


  if cfg["early_pred_analysis_flag"] == True:
    logger.info ("Getting output variable");
    output_var      = get_output_variable (cfg["target_type"]);
    output_var_name = output_var[0]; # Not elegant at all

    #load_file_name = f'test_data_all_features.csv.gz' if this_day == 0 else f'all_features_{str(this_day)}.csv.gz'
    logger.info (f"Analysing predictions for {this_day}");
    #df_test = pd.read_csv (cfg["train_data_path"] / load_file_name);
    #time_till_inf_sub_clin = pd.read_csv (f"{file_prefix}_time_till_inf_sub_clin.csv");
    #df_test["time_till_inf_sub_clin"] = time_till_inf_sub_clin;
    result_dir = cfg["result_dir"];
    pred_info = pd.read_csv (f"{result_dir}/{file_prefix}_{str (this_day)}_pred_info.csv");
    #if "time_till_inf_sub_clin_0days" not in pred_info.columns:
        #pred_info["time_till_inf_sub_clin_0days"] = df_test["time_till_inf_sub_clin_0days"];
    result = early_detect_by_days_subclin (pred_info, output_var_name);

    proportions_file_path = f"{result_dir}/{file_prefix}_seven_day_proportiton.csv" if this_day == 0 else f"{result_dir}/{file_prefix}_seven_day_proportiton_{str (this_day)}.csv";
    overall_cows_file_path = f"{result_dir}/{file_prefix}_seven_day_cow_raw.joblib" if this_day == 0 else f"{result_dir}/{file_prefix}_seven_day_cow_raw_{str (this_day)}.joblib";

    np.savetxt (proportions_file_path, result["proportions"]);
    joblib.dump (result, overall_cows_file_path);
    

  if cfg["plot_borderline_hists"] == True:
    pred = pd.read_csv (f"vision_latest/{file_prefix}_{this_day}_pred_info.csv");
    scc_val = df_test["1_scc"];
    pred = pd.concat ([pred.reset_index (drop = True), scc_val.reset_index (drop = True)], axis = 1);

    tmp_real_true = pred.iloc[(pred["class"] == 1).tolist (),:];
    tmp_wrong = tmp_real_true.iloc[(tmp_real_true["pred_thresh"] != tmp_real_true["class"]).tolist (),:]
    tmp_right = tmp_real_true.iloc[(tmp_real_true["pred_thresh"] == tmp_real_true["class"]).tolist (),:]

    #tmp_wrong = pred.iloc[(pred["pred_thresh"] != pred["class"]).tolist (),:]
    #tmp_right = pred.iloc[(pred["pred_thresh"] == pred["class"]).tolist (),:]

    fig, ax = plt.subplots(1, 2)
    ax[0].hist (tmp_right["1_scc"], label = "Predict Right", density = False, stacked = True, bins = 20, histtype = "bar", alpha = 0.5);
    ax[0].hist (tmp_wrong["1_scc"], label = "Predict Wrong", density = False, stacked = True, bins = 20, histtype = "bar", alpha = 0.5);
    ax[0].set_xlabel ("1_scc");
    ax[0].legend ();


    tmp_real_true = pred.iloc[(pred["class"] == 0).tolist (),:];
    tmp_wrong = tmp_real_true.iloc[(tmp_real_true["pred_thresh"] != tmp_real_true["class"]).tolist (),:]
    tmp_right = tmp_real_true.iloc[(tmp_real_true["pred_thresh"] == tmp_real_true["class"]).tolist (),:]

    ax[1].hist (tmp_right["1_scc"], label = "Predict Right", density = False, stacked = True, bins = 20, histtype = "bar", alpha = 0.5);
    ax[1].hist (tmp_wrong["1_scc"], label = "Predict Wrong", density = False, stacked = True, bins = 20, histtype = "bar", alpha = 0.5);
    ax[1].set_xlabel ("1_scc");
    ax[1].legend ();

    ax[0].set_title ("True class Infected");
    ax[1].set_title ("True class Not Infected");

    fig.suptitle (f"Recording frequency {this_day} days");

    plt.show ();
else:
  # Just sourcing to update code
  pass;

