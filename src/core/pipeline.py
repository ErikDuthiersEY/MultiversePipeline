from azure.ai.ml import Input, Output, command
from azure.ai.ml.dsl import pipeline
from pathlib import Path

def build_eval_pipeline(env_path:str, datasets_path:str, config_path:str, compute:str):
    inference = command(
        code=str(Path(__file__).parents[1] / "components/inference"),
        environment=env_path, 
        command="python infer.py --datasets ${{inputs.datasets}} --output_dir ${{outputs.out_dir}} --config ${{inputs.config}}",
        inputs={"datasets": Input(type="uri_folder"), "config": Input(type="uri_file")},
        outputs={"out_dir": Output(type="uri_folder")},
    )

    rule_based_metrics = command(
        code=str(Path(__file__).parents[1] / "components/rule_based_metrics"),
        environment=env_path,
        command="python metrics.py --raw_dir ${{inputs.raw_dir}} --out_dir ${{outputs.out_dir}}",
        inputs={"raw_dir": Input(type="uri_folder")},
        outputs={"out_dir": Output(type="uri_folder")},
    )

    aggregate = command(
        code=str(Path(__file__).parents[1] / "components/aggregate"),
        environment=env_path,
        command="python aggregate.py --scores ${{inputs.scores}} --out_dir ${{outputs.out_dir}}",
        inputs={"scores": Input(type="uri_folder")},
        outputs={"out_dir": Output(type="uri_folder")},
    )

    @pipeline() 
    def _pipeline(datasets_folder: Input, config_file: Input) -> None:
        infer_job = inference(datasets=datasets_folder, config=config_file)
        rule_based_metrics_job = rule_based_metrics(raw_dir=infer_job.outputs.out_dir),
        agg_job = aggregate(scores=rule_based_metrics_job[0].outputs.out_dir)
        return {"summary": agg_job.outputs.out_dir}

    pl = _pipeline(
        datasets_folder=Input(type="uri_folder", path=datasets_path),
        config_file=Input(type="uri_file", path=config_path),
    )
    pl.settings.default_compute = compute
    return pl