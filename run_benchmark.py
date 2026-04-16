from mot_pipeline.benchmark import MOT17Evaluator
from mot_pipeline.pipeline import load_config

config = load_config()
evaluator = MOT17Evaluator(config)
evaluator.run()
