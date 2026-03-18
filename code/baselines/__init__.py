from .bayesian import BayesianOptimizer
from .all_on import AllOnOptimizer
from .random_subset import RandomSubsetOptimizer
from .heuristic_subset import HeuristicCenterOptimizer, HeuristicIntensityOptimizer
from .phosopt_per_target import PhosOptPerTargetOptimizer

METHOD_REGISTRY: dict[str, type] = {
    "bayesian": BayesianOptimizer,
    "all_on": AllOnOptimizer,
    "random_subset": RandomSubsetOptimizer,
    "heuristic_center": HeuristicCenterOptimizer,
    "heuristic_intensity": HeuristicIntensityOptimizer,
    "phosopt_per_target": PhosOptPerTargetOptimizer,
}
