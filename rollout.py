import tempfile
import timeout_decorator
from loguru import logger

from evaluator.evaluator import Learn2RaceEvaluator
from config import SubmissionConfig, EnvConfig, SimulatorConfig
import matplotlib.pyplot as plt

pre_evaluate_model_file, pre_evaluate_model_path = tempfile.mkstemp()


def training_routine(evaluator):
    try:
        # evaluator.load_agent_model('models/sac/checkpoints/my_sac.statedict')
        evaluator.train()
    except timeout_decorator.TimeoutError:
        logger.info("Stopping pre-evaluation run")

    evaluator.save_agent_model('models/sac/checkpoints/my_sac.statedict')


def evaluation_routine(evaluator):
    evaluator.load_agent_model('models/sac/checkpoints/my_sac.statedict')
    scores = evaluator.evaluate()
    logger.success(f"Average metrics: {scores}")


def run_evaluation():
    evaluator = Learn2RaceEvaluator(
        submission_config=SubmissionConfig,
        sim_config=SimulatorConfig,
        env_config=EnvConfig,
    )

    evaluator.create_env()
    evaluator.init_agent()

    ## Local development OR Stage 2 'practice' phase
    training_routine(evaluator)
    ## Evaluation
    evaluation_routine(evaluator)


if __name__ == "__main__":
    run_evaluation()
