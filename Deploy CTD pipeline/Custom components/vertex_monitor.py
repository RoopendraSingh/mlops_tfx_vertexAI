#Required Packages for the component
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.annotations import OutputDict

from typing import Dict
from tfx import v1 as tfx
from absl import logging
import copy
import time
import json

from google.cloud import storage
from google.cloud import aiplatform

from google.protobuf.duration_pb2 import Duration
from google.cloud import aiplatform_v1beta1 as vertex_ai_beta


@component
def Monitor(pushed_model: tfx.dsl.components.InputArtifact[tfx.types.standard_artifacts.PushedModel],
            project: Parameter[str], location: Parameter[str], endpoint_display_name: Parameter[str], 
            bigquery_table: Parameter[str], monitoring_parameters_uri: Parameter[str], 
            model_display_name: Parameter[str]) -> tfx.dsl.components.OutputDict(result=str):
    '''
    Checks whether the current_model is pushed. If pushed, initiates model 
    monitoring.
    
    Args:
    pushed_model: A TFX component artifact used for orchestration. 
    project: The GCP project_id where the model and endpoints is deployed.
    location: The location where the model and endpoints is deployed.
    endpoint_display_name: The endpoint name where the model was deployed.
    bigquery_table: The BigQuery table on which the model was trained.
    skew_threshold: The skew thresholds for model.
    drift_threshold: The drift thresholds for model.
    monitor_params: Other monitoring parameters for monitoring alerts.
    
    Returns:
    A dictionary with the result of whether the model is being monitored.
    '''
    
    gcs_client = storage.Client()
    
    # Checks Model Push Status
    model_pushed = pushed_model.get_int_custom_property('pushed')
    if model_pushed==0:
        return {"result": "Model was not Pushed, hence no update required."}
    
    # Getting Monitoring Parameters
    bucket, file_name = monitoring_parameters_uri.split('gs://')[1].split('/',1)
    in_bucket = gcs_client.bucket(bucket)
    blob = in_bucket.get_blob(file_name)
    with open('monitoring_parameters.json', 'wb') as local_file:
        gcs_client.download_blob_to_file(blob, local_file)

    f = open('monitoring_parameters.json', 'r')
    monitoring_parameters = dict(json.load(f))
    f.close()
    
    monitor_params = monitoring_parameters['monitor_params']
    
    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.Endpoint.list(filter=f"display_name={endpoint_display_name}",
                                         order_by="update_time")[-1]
    
    PARENT = f"projects/{project}/locations/{location}"
    API_ENDPOINT = f"{location}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": API_ENDPOINT}
    
    # Removing previous model monitoring jobs on this endpoint.
    try:
        client = vertex_ai_beta.JobServiceClient(client_options=client_options)
        response = client.list_model_deployment_monitoring_jobs(parent=PARENT)
    
        for job_i in response:
            if endpoint.resource_name == job_i.endpoint:
                monit_job = job_i.name
                logging.info("Found {} as the Monitoring Job.".format(monit_job))    
                client.pause_model_deployment_monitoring_job(name=monit_job)
                time.sleep(30)
                client.delete_model_deployment_monitoring_job(name=monit_job)

        logging.info("Removed all model monitoring jobs for the endpoint.")
    except:
        logging.info("Error while removing model monitoring job.")
    
    time.sleep(300)
    
    model_version = pushed_model.get_string_custom_property('pushed_version')
    model_name = model_display_name+model_version
    not_deployed = 1
    
    while not_deployed > 0:
        if model_name in [model.display_name for model in endpoint.list_models()]:
            not_deployed = -1
        else:
            if not_deployed == 6:
                logging.info("Model took too long to deploy or Model not deployed.")
                break
            time.sleep(200)
            not_deployed += 1
            
    if not_deployed > 0:     
        return {"result": "No monitoring job for the pipeline"}
    
    # Creating model monitoring job on this endpoint.
    MONITORING_JOB_NAME = f"monitor-{endpoint.display_name}"
    NOTIFY_EMAILS = monitor_params['emails']
    LOG_SAMPLE_RATE = monitor_params['sample_rate']
    MONITOR_INTERVAL = monitor_params['interval']
    TARGET_FEATURE_NAME = monitor_params['target']
    
    SKEW_THRESHOLDS = monitoring_parameters['skew_threshold']
    DRIFT_THRESHOLDS = monitoring_parameters['drift_threshold']
    DESTINATION_TABLE_URI = bigquery_table
    
    job_client_beta = vertex_ai_beta.JobServiceClient(client_options=client_options)
    endpoint_uri = endpoint.gca_resource.name
    model_ids = [model.id for model in endpoint.list_models()]
    
    skew_thresholds = {
        feature: vertex_ai_beta.ThresholdConfig(value=float(value))
        for feature, value in SKEW_THRESHOLDS.items()
    }

    drift_thresholds = {
        feature: vertex_ai_beta.ThresholdConfig(value=float(value))
        for feature, value in DRIFT_THRESHOLDS.items()
    }

    skew_config = vertex_ai_beta.ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
        skew_thresholds=skew_thresholds
    )

    drift_config = vertex_ai_beta.ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
        drift_thresholds=drift_thresholds
    )

    sampling_config = vertex_ai_beta.SamplingStrategy(
        random_sample_config=vertex_ai_beta.SamplingStrategy.RandomSampleConfig(
            sample_rate=LOG_SAMPLE_RATE
        )
    )

    schedule_config = vertex_ai_beta.ModelDeploymentMonitoringScheduleConfig(
        monitor_interval=Duration(seconds=MONITOR_INTERVAL)
    )

    train_sampling_config = vertex_ai_beta.SamplingStrategy(
        random_sample_config=vertex_ai_beta.SamplingStrategy.RandomSampleConfig(
            sample_rate=LOG_SAMPLE_RATE
        )
    )
    training_dataset = vertex_ai_beta.ModelMonitoringObjectiveConfig.TrainingDataset(
        target_field=TARGET_FEATURE_NAME,
        logging_sampling_strategy=train_sampling_config,
        bigquery_source = vertex_ai_beta.types.io.BigQuerySource(
            input_uri=DESTINATION_TABLE_URI
        )
    )

    objective_template = vertex_ai_beta.ModelDeploymentMonitoringObjectiveConfig(
        objective_config=vertex_ai_beta.ModelMonitoringObjectiveConfig(
            training_dataset=training_dataset,
            training_prediction_skew_detection_config=skew_config,
            prediction_drift_detection_config=drift_config,
        )
    )

    deployment_objective_configs = []
    for model_id in model_ids:
        objective_config = copy.deepcopy(objective_template)
        objective_config.deployed_model_id = model_id
        deployment_objective_configs.append(objective_config)

    alerting_config = vertex_ai_beta.ModelMonitoringAlertConfig(
        email_alert_config=vertex_ai_beta.ModelMonitoringAlertConfig.EmailAlertConfig(
            user_emails=NOTIFY_EMAILS
        )
    )
    
    job = vertex_ai_beta.ModelDeploymentMonitoringJob(
        display_name=MONITORING_JOB_NAME,
        endpoint=endpoint_uri,
        model_deployment_monitoring_objective_configs=deployment_objective_configs,
        logging_sampling_strategy=sampling_config,
        model_deployment_monitoring_schedule_config=schedule_config,
        model_monitoring_alert_config=alerting_config,
        enable_monitoring_pipeline_logs=True
    )
    
    response = job_client_beta.create_model_deployment_monitoring_job(
        parent=PARENT, model_deployment_monitoring_job=job
    )
    
    logging.info("{} is the name of the Model Monitoring Job.".format(response.name))
    return {"result": "{} is the name of the Model Monitoring Job.".format(response.name)}
