import os, yaml
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from src.core.pipeline import build_eval_pipeline

SUB = os.environ["AZ_SUBSCRIPTION_ID"]
RG  = os.environ["AZ_RESOURCE_GROUP"]
WS  = os.environ["AZ_ML_WORKSPACE"]

ml = MLClient(DefaultAzureCredential(), SUB, RG, WS)

env_path     = "configs/env/conda.yaml"
datasets_dir = "azureml://subscriptions/3bcd6b3d-11a3-4a3e-9e9e-7beb46101ac4/resourcegroups/ws-multi-studio/workspaces/multistudioaml/datastores/workspaceblobstore/paths/LocalUpload/64c8c30cac2dd93c303d3bee5abdfa74d3e1cbf38e6e005a9dbf9c2e6371590c/"
config_path  = "configs/config.yaml"
compute      = "metaverse-compute" 

run = build_eval_pipeline(env_path, datasets_dir, config_path, compute)
run.display_name = yaml.safe_load(open(config_path))["run_name"]

submitted = ml.jobs.create_or_update(run)
print("Submitted:", submitted.name)
