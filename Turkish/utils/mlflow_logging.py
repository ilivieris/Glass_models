import os
import mlflow
import numpy as np
import pandas as pd
import json
from utils.pretty_confusion_matrix import pp_matrix_from_data

def MLflow_log_performance(number, model, columns=None,
                           train_CV_results=None, test_CV_results=None, CM_cv=None, 
                           train_results=None, test_results=None, CM=None, 
                           signature=None,
                           params=None):
    # Get model name
    model_name = type(model).__name__

    # Start logging     
    mlflow.start_run(run_name=f'{model_name}-Trial:{number}')
    
    # MLflow tags
    tag = {"Simulation" : "Trial-" + str(number), "model": model_name}
    # Tags to help in tracking
    mlflow.set_tags(tag)

    # Log params/hyperparameters used in experiement
    for x in params.keys():
        mlflow.log_param(x, params[x])
    
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # CV results

    # Create artifact path
    artifact_path = f'Performance/Trial-{number}'
    if not os.path.isdir(artifact_path): os.mkdir(artifact_path)
    # Log metrics to MLflow (training CV performance)
    if (train_CV_results is not None):
        mlflow.log_metric("train_CV_Accuracy", np.mean(train_CV_results['Accuracy']))
        mlflow.log_metric("train_CV_AUC", np.mean(train_CV_results['AUC']))
        mlflow.log_metric("train_CV_Recall", np.mean(train_CV_results['Recall']))
        mlflow.log_metric("train_CV_Precision", np.mean(train_CV_results['Precision']))
        mlflow.log_metric("train_CV_GM", np.mean(train_CV_results['GM']))
        # Store performance
        pd.DataFrame.from_dict(train_CV_results).to_json(f'{artifact_path}/Training_CV_performance.json')
    # Log metrics to MLflow (testing CV performance)
    if (test_CV_results is not None):
        mlflow.log_metric("test_CV_Accuracy", np.mean(test_CV_results['Accuracy']))
        mlflow.log_metric("test_CV_AUC", np.mean(test_CV_results['AUC']))
        mlflow.log_metric("test_CV_Recall", np.mean(test_CV_results['Recall']))
        mlflow.log_metric("test_CV_Precision", np.mean(test_CV_results['Precision']))
        mlflow.log_metric("test_CV_GM", np.mean(test_CV_results['GM']))
        # Store performance
        pd.DataFrame.from_dict(test_CV_results).to_json(f'{artifact_path}/Testing_CV_performance.json')
    # Confusion matrix CV  
    if (CM_cv is not None):
        pp_matrix_from_data(CM=CM_cv, columns=columns, cmap="Oranges", figsize=(13,13), path=f'{artifact_path}/Confusion_Matrix_CV.png')
  
  
  

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Single-run results

    # Log metrics to MLflow (training CV performance)
    mlflow.log_metric("train_Accuracy", train_results['Accuracy'])
    mlflow.log_metric("train_AUC", train_results['AUC'])
    mlflow.log_metric("train_Recall", train_results['Recall'])
    mlflow.log_metric("train_Precision", train_results['Precision'])
    mlflow.log_metric("train_GM", train_results['GM'])
    # Log metrics to MLflow (testing CV performance)
    mlflow.log_metric("test_Accuracy", test_results['Accuracy'])
    mlflow.log_metric("test_AUC", test_results['AUC'])
    mlflow.log_metric("test_Recall", test_results['Recall'])
    mlflow.log_metric("test_Precision", test_results['Precision'])
    mlflow.log_metric("test_GM", test_results['GM'])

    # Store performance
    with open(f'{artifact_path}/Training_performance.json', "w") as outfile:
        outfile.write(json.dumps(train_results, indent=4))
    with open(f'{artifact_path}/Testing_performance.json', "w") as outfile:
        outfile.write(json.dumps(test_results, indent=4))        
    # Confusion matrix 
    pp_matrix_from_data(CM=CM, columns=columns, cmap="Blues", figsize=(13,13), path=f'{artifact_path}/Confusion_Matrix.png')
  

    # Store model's params
    mlflow.log_dict(model.get_params(), "parameters.json")

    # Log artifacts
    mlflow.log_artifacts(artifact_path, artifact_path="Performance")
    
    # Log model created
    mlflow.sklearn.log_model(model, artifact_path="models", signature = signature) 
    
    mlflow.end_run()