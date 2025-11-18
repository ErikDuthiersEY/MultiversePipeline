import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import os, yaml
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from src.core.pipeline import build_eval_pipeline
from dotenv import load_dotenv
from azure.ai.ml.entities import Environment
load_dotenv()

SUB = os.environ["AZ_SUBSCRIPTION_ID"]
RG  = os.environ["AZ_RESOURCE_GROUP"]
WS  = os.environ["AZ_ML_WORKSPACE"]

ml = MLClient(DefaultAzureCredential(), SUB, RG, WS)

env_path     = "configs/env/conda.yaml"
datasets_dir = "datasets/raw"
#datasets_dir = "azureml://subscriptions/3bcd6b3d-11a3-4a3e-9e9e-7beb46101ac4/resourcegroups/ws-multi-studio/workspaces/multistudioaml/datastores/workspaceblobstore/paths/LocalUpload/08b21c0155fa8a18f99328761587c1addeda207ab8427f782709f247889aea51/processed/"
config_path  = "configs/config.yaml"
compute      = "metaverse-compute" 

env = Environment(
    name="multiverse-env",
    description="Env for evaluation pipeline",
    conda_file=env_path,
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest", 
)

run = build_eval_pipeline(env, datasets_dir, config_path, compute)
run.display_name = yaml.safe_load(open(config_path))["run_name"]

submitted = ml.jobs.create_or_update(run)
print("Submitted:", submitted.name)
