import tensorflow as tf
from tfx import v1 as tfx
import kfp
import os
from typing import List, Optional
from tfx.components.base import executor_spec
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.executor import GenericExecutor
from tfx.proto import example_gen_pb2
from tfx_bsl.public import tfxio
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen
from custom_components.vertex_deployer import Deployer
from custom_components.vertex_monitor import Monitor
    
def create_pipeline(pipeline_name: str, pipeline_root: str,
                     query: str, NUM_RECORDS:int, beam_pipeline_args: Optional[List[str]],
                     module_file: str, project_id: str, use_gpu: int, serving_model_dir:str, 
                     region: str, endpoint_name: str, model_name: str, container_img: str, bq_table: str,
                    monitoring_parameters_uri:str) -> tfx.dsl.Pipeline:
    """Implements the loan pipeline with TFX."""

    # Brings data into the pipeline or otherwise joins/converts training data.
    output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
        example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
        example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)                                  
    ]))
    
    # Uses BigQueryExampleGen to fetch the data from a BQ table, splits it into
    # different splits and saves it to a tf.Example records format.
    example_gen = BigQueryExampleGen(query=query, output_config=output)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])
    
    
    schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'],
                                          infer_feature_shape=True)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

    # Transforms input data using preprocessing_fn in the 'module_file'.
    transform = tfx.components.Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      materialize=False,
      module_file=module_file)

    # vertex job spec to define the machine-specs for the container to be used by the training component.
    vertex_job_spec = {
      'project': project_id,
      'worker_pool_specs': [{
          'machine_spec': {
              'machine_type': 'n1-standard-4',
          },
          'replica_count': 1,
          'container_spec': {
              'image_uri': 'gcr.io/tfx-oss-public/tfx:1.3.2',
          }
      }],
    }

    if use_gpu!=0:
    # See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#acceleratortype
    # for available machine types.
        vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': use_gpu
        })

    # tfx trainer component uses the examples from example gen component and transform graph from the TFT
    # component to train the our model. 
    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=tfx.proto.TrainArgs(splits=['train'], num_steps=int(NUM_RECORDS*0.8/32)),
      eval_args=tfx.proto.EvalArgs(splits=['eval'], num_steps=int(NUM_RECORDS*0.1/32)),
      custom_config={
          tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
          tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
          tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
              vertex_job_spec,
          'use_gpu':
              use_gpu!=0}
    )
    # Fetches the previously trained best blessed model for the given pipeline name.
    model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

    #This eval configuration is used by the evaluator component to compare the 
    #latest trained modle with the previous best blessed model.
    eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and 
        # remove the label_key.
        tfma.ModelSpec(
            signature_name='serving_default',
            label_key='loan_status',
             preprocessing_function_names=["tft_layer"]
            )
        ],
    metrics_specs=[
        tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            # To add validation thresholds for metrics saved with the model,
            # add them keyed by metric name to the thresholds map.
            metrics=[
                tfma.MetricConfig(class_name='BinaryAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.5}),
                      # Change threshold will be ignored if there is no
                      # baseline model resolved from MLMD (first run).
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
                ]
            )
        ])

  # Evaluates the current model with previously best blessed model using the eval config and the test dataset
    evaluator = tfx.components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  # Pushes the model to a filesystem destination.
    pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))
    
   # Deploys the model to an endpoint, if pushed.
    deployer = Deployer(pushed_model=pusher.outputs["pushed_model"],
                      project=project_id, location=region, container_image_uri=container_img, 
                      endpoint_display_name = endpoint_name, model_display_name = model_name)   
    
    # Updates monitoring job, if pushed.
    monitor = Monitor(pushed_model=pusher.outputs["pushed_model"], project=project_id, location=region, 
                      endpoint_display_name = endpoint_name, bigquery_table=bq_table,
                      monitoring_parameters_uri = monitoring_parameters_uri, model_display_name = model_name)

    components = [
      example_gen,
      statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
        deployer,
        monitor
    ]

    return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      beam_pipeline_args=beam_pipeline_args,
      components=components)