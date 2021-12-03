#Required Packages for the component
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.annotations import OutputDict

from tfx import v1 as tfx
from absl import logging

from google.cloud import storage
from google.cloud import aiplatform

@component
def Deployer(pushed_model: tfx.dsl.components.InputArtifact[tfx.types.standard_artifacts.PushedModel],
             project: Parameter[str], location: Parameter[str], container_image_uri: Parameter[str], 
             endpoint_display_name: Parameter[str], model_display_name: Parameter[str],
            ) -> tfx.dsl.components.OutputDict(result=str):
    '''
    Checks whether the current_model is pushed. If pushed, uploads the model
    and deploys to the specified endpoint.
    
    Args:
    pushed_model: A TFX component artifact used for orchestration. 
    project: The GCP project_id where the model and endpoints will be uploaded
    and deployed.
    location: The location where the model and endpoints will be uploaded and 
    deployed.
    container_image_uri: The container_image_uri having all the packages installed
    for doing predictions using endpoint.
    endpoint_display_name: The endpoint name where the model will be deployed.
    model_display_name: The model name for the model to be uploaded.
    
    Returns:
    A dictionary with the result of whether the model is pushed. And if pushed, 
    details the model name and endpoint where it is deployed for online predicition.
    '''
    
    gcs_client = storage.Client()
    
    # Checks Model Push Status
    model_pushed = pushed_model.get_int_custom_property('pushed')
    if model_pushed==0:
        return {"result": "Model was not Blessed/Pushed"}
    
    # If Model was pushed, find the Serving URI and Model Version
    model_artifact_uri = pushed_model.get_string_custom_property('pushed_destination')
    model_version = pushed_model.get_string_custom_property('pushed_version')
    
    # Initialize the AI Platform API
    aiplatform.init(project=project, location=location)
    
    # Upload Model
    model = aiplatform.Model.upload(display_name=model_display_name+model_version,
        artifact_uri=model_artifact_uri,
        serving_container_image_uri=container_image_uri)
    
    #Find the most recent endpoint with the specific "endpoint_display_name" 
    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_display_name}",
                                         order_by="update_time")
    if len(endpoints) > 0:
        logging.info(f"Endpoint {endpoint_display_name} already exists.")
        endpoint = endpoints[-1]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    
    # Code to undeploy all models from endpoint
    endpoint.undeploy_all()
    
    # Deploy the model to the endpoint
    model.deploy(endpoint=endpoint, machine_type='n1-standard-4', traffic_split={"0": 100})
    
    # Log the Model and Endpoint information
    logging.info("Model {} was deployed to {} endpoint.".format(model.display_name, endpoint.display_name))
    return {"result": "Model {} was deployed to {} endpoint.".format(model.display_name, endpoint.display_name)}