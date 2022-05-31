from graphgym.automl.algorithms.random import Random
from graphgym.automl.algorithms.hill_climb import HillClimb
from graphgym.automl.algorithms.simulated_anneal import SimulatedAnneal
from graphgym.automl.algorithms.bayesian_opt import BayesianOptimization

from graphgym.automl.algorithms.darts import Darts
from graphgym.automl.algorithms.enas import Enas
from graphgym.automl.algorithms.graphnas import GraphNASMacro
from graphgym.automl.algorithms.falcon import Falcon

algorithm_dict = {
    'random': Random, 
    'hill_climb': HillClimb, 
    'simulated_anneal': SimulatedAnneal,
    'bayesian_opt': BayesianOptimization,
    
    'graphnas': GraphNASMacro,
    'darts': Darts,
    'enas': Enas,
    'falcon': Falcon
}
