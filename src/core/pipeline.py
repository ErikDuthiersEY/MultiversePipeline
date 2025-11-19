from azure.ai.ml import Input, Output, command
from azure.ai.ml.dsl import pipeline
from pathlib import Path
from azure.ai.ml.entities import Environment

def build_eval_pipeline(env:Environment, datasets_path:str, config_path:str, compute:str):

    inference = command(
        code=str(Path(__file__).parents[1] / "components/inference"),
        environment=env, 
        command="python run_infer.py --datasets ${{inputs.datasets}} --output_dir ${{outputs.out_dir}} --config ${{inputs.config}}",
        inputs={"datasets": Input(type="uri_folder"), "config": Input(type="uri_file")},
        outputs={"out_dir": Output(type="uri_folder")},
    )

    rule_based_metrics = command(
        code=str(Path(__file__).parents[1] / "components/rule_based_metrics"),
        environment=env,
        command="python run_rule_based_metrics.py --raw_dir ${{inputs.raw_dir}} --out_dir ${{outputs.out_dir}} --config ${{inputs.config}} --datasets ${{inputs.datasets}}",
        inputs={"raw_dir": Input(type="uri_folder"), "datasets": Input(type="uri_folder"), "config": Input(type="uri_file")},
        outputs={"out_dir": Output(type="uri_folder")},
    )
    
    llm_based_metrics = command(
        code=str(Path(__file__).parents[1] / "components/llm_evaluated_metrics"),
        environment=env,
        command="python run_llm_evaluated_metrics.py --raw_dir ${{inputs.raw_dir}} --out_dir ${{outputs.out_dir}} --config ${{inputs.config}}",
        inputs={"raw_dir": Input(type="uri_folder"), "config": Input(type="uri_file")},
        outputs={"out_dir": Output(type="uri_folder")},
    )

    aggregate = command(
        code=str(Path(__file__).parents[1] / "components/aggregation"),
        environment=env,
        command="python run_aggregate.py --scores_dir ${{inputs.rule_scores}} ${{inputs.llm_scores}} --out_dir ${{outputs.out_dir}}",
        inputs={"rule_scores": Input(type="uri_folder"), "llm_scores": Input(type="uri_folder")},
        outputs={"out_dir": Output(type="uri_folder")},
    )

    @pipeline() 
    def _pipeline(datasets_folder: Input, config_file: Input) -> None:
        infer_job = inference(datasets=datasets_folder, config=config_file)
        rule_based_metrics_job = rule_based_metrics(raw_dir=infer_job.outputs.out_dir, datasets=datasets_folder, config=config_file)
        llm_based_metrics_job = llm_based_metrics(raw_dir=infer_job.outputs.out_dir, config=config_file)
        agg_job = aggregate(scores_dir=rule_based_metrics_job.outputs.out_dir, llm_scores=llm_based_metrics_job.outputs.out_dir)
        return {"summary": agg_job.outputs.out_dir}

    pl = _pipeline(
        datasets_folder=Input(type="uri_folder", path=datasets_path),
        config_file=Input(type="uri_file", path=config_path),
    )
    pl.settings.default_compute = compute
    return pl