import os, yaml
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from src.core.pipeline import build_eval_pipeline

SUB = os.environ["AZ_SUBSCRIPTION_ID"]
RG  = os.environ["AZ_RESOURCE_GROUP"]
WS  = os.environ["AZ_ML_WORKSPACE"]

ml = MLClient(DefaultAzureCredential(), SUB, RG, WS)

env_path     = "configs/env/conda.yaml"
datasets_dir = "datasets"
config_path  = "configs/config.yaml"
compute      = "" #include compute name

run = build_eval_pipeline(env_path, datasets_dir, config_path, compute)
run.display_name = yaml.safe_load(open(config_path))["run_name"]

submitted = ml.jobs.create_or_update(run)
print("Submitted:", submitted.name)
