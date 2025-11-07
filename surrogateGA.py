from mealpy import IntegerVar, GA
from run_simulation import objective_function
import numpy as np
from xgboost import XGBRegressor
from mealpy.utils.target import Target
NUMBER_OF_WORKERS = 8
import concurrent.futures as parallel
from functools import partial
from typing import  Union, List, Tuple
from mealpy.utils.agent import Agent
import time
class SurrogateObjective:
    def __init__(self):
        self.model = XGBRegressor(n_jobs=1)
        self.X_train = []
        self.y_train = []
        self.is_trained = False

    def add_data(self, X, y):
        """Add training data for the surrogate model"""
        if isinstance(X, list):
            self.X_train.extend(X)
        else:
            self.X_train.append(X)

        if isinstance(y, list):
            self.y_train.extend(y)
        else:
            self.y_train.append(y)

    def train(self):
        """Train the surrogate model on collected data"""
        if len(self.X_train) > 10:
            self.model.fit(np.array(self.X_train), np.array(self.y_train))
            self.is_trained = True
            #print(f"  [Surrogate trained on {len(self.X_train)} samples]")

    def predict(self, X):
        """Predict fitness using surrogate model"""
        if not self.is_trained:
            return None
        X_arr = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)
        return self.model.predict(X_arr)[0] if len(X_arr) == 1 else self.model.predict(X_arr)






class GAWithSurrogate(GA.BaseGA):
    """Custom GA that uses surrogate model intelligently"""

    def __init__(self, surrogate, scenario, retrain_ratio=0.1, warmup_ratio=0.1, real_eval_ratio=0.1,  **kwargs):
        # Store surrogate and scenario before calling super().__init__
        self.surrogate = surrogate
        self.scenario = scenario
        self.retrain_ratio = retrain_ratio
        self.warmup_ratio = warmup_ratio
        self.real_eval_ratio = real_eval_ratio
        self.generation_count = 0
        self.eval_count = 0
        self.warmup_generations = None
        self.real_evaluations = {}
        self.is_real_evaluation = True
        super().__init__(**kwargs)

    def initialize_variables(self):
        """Override to set warmup threshold based on epochs"""
        super().initialize_variables()
        self.warmup_generations = int(self.epoch * self.warmup_ratio)

    def should_retrain(self):
        """Check if we should retrain surrogate """
        train_interval = max(1, int(self.epoch * self.retrain_ratio))
        return self.generation_count % train_interval == 0 and self.generation_count > 0

    def evaluate_agent_real(self, solution):
        """Evaluate a single solution with real objective function"""
        fitness = objective_function(self.scenario, solution)
        self.eval_count += 1
        self.real_evaluations[tuple(solution)] = fitness
        self.surrogate.add_data(solution, fitness)
        return fitness

    def evaluate_agent_surrogate(self, solution):
        """Evaluate using surrogate model"""
        if self.surrogate.is_trained:
            self.eval_count += 1
            return self.surrogate.predict(solution)
        else:
            # Fallback to real evaluation if surrogate not ready
            return self.evaluate_agent_real(solution)

    def get_target(self, solution: np.ndarray, counted: bool = True  ) -> Target:

        warmup_iter = int(self.epoch * self.warmup_ratio)
        if counted:
            self.nfe_counter += 1
        if(self.generation_count <= warmup_iter or self.generation_count==self.epoch or self.is_real_evaluation==True):
            objs = self.evaluate_agent_real(solution)
        else:
            if self.surrogate.is_trained:
                objs = self.evaluate_agent_surrogate(solution)
            else:
                objs = self.evaluate_agent_real(solution)
        t = Target(objectives=objs, weights=self.problem.obj_weights)
        t._fitness = objs
        return t



    def evolve(self, epoch):
        """
             The main operations (equations) of algorithm. Inherit from Optimizer class

             Args:
                 epoch (int): The current iteration
             """
        # Train surrogate periodically
        self.is_real_evaluation = False
        if self.should_retrain():
           # print(f"  Retraining surrogate at generation {epoch}...")
            self.surrogate.train()
        self.generation_count += 1
        list_fitness = np.array([agent.target.fitness for agent in self.pop])
        pop_new = []
        for i in range(0, int(self.pop_size / 2)):
            ### Selection
            child1, child2 = self.selection_process__(list_fitness)

            ### Crossover
            if self.generator.random() < self.pc:
                child1, child2 = self.crossover_process__(child1, child2)

            ### Mutation
            child1 = self.mutation_process__(child1)
            child2 = self.mutation_process__(child2)

            child1 = self.correct_solution(child1)
            child2 = self.correct_solution(child2)

            agent1 = self.generate_empty_agent(child1)
            agent2 = self.generate_empty_agent(child2)

            pop_new.append(agent1)
            pop_new.append(agent2)

            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-2].target = self.get_target(child1)
                pop_new[-1].target = self.get_target(child2)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        ### Survivor Selection
        self.pop = self.survivor_process__(self.pop, pop_new)
        # Step 2: Sort combined population and re-evaluate top "real_eval_ratio" % with REAL function
        self.pop = self.get_sorted_population(self.pop, self.problem.minmax)
        n_real_evals = max(1, int(len(self.pop) * self.real_eval_ratio))
        for idx in range(n_real_evals):
            self.is_real_evaluation = True
            self.pop[idx].target = self.get_target(self.pop[idx].solution)

        # Step 3: Re-sort based on real evaluations and update population
        self.pop = self.get_sorted_and_trimmed_population(self.pop, self.pop_size, self.problem.minmax)
        #print("Generation",epoch,"best solution:", self.g_best.target.fitness)

def runGA(scenario):
    """
    Run Genetic Algorithm with surrogate-assisted optimization using two-pass evaluation

    Args:
        scenario: Dictionary containing problem parameters including 'nDevices'

    Returns:
        tuple: (best_solution, best_fitness)
    """

    nDevices = scenario["nDevices"]

    # GA parameters
    epoch = 100
    pop_size = 50

    # Setup surrogate
    surrogate = SurrogateObjective()

    # Define problem
    problem_dict = {
        "obj_func": lambda sol: objective_function(scenario, sol),  # Placeholder, won't be used directly
        "bounds": IntegerVar(lb=(7,) * nDevices, ub=(12,) * nDevices),
        "minmax": "max",
        "log_to": None,
    }

    # Create custom optimizer with surrogate

    optimizer = GAWithSurrogate(
        surrogate=surrogate,
        scenario=scenario,
        retrain_ratio = 0.05,
        warmup_ratio=0.1,  # First 10% of generations use real function
        real_eval_ratio=0.1,  # Top 10% of population re-evaluated with real function
        epoch=epoch,
        pop_size=pop_size,
        pc=0.9,  # Crossover probability
        pm=0.05,  # Mutation probability
        verbose=False,
    )

    # Solve the problem
    start = time.time()
    best_solution = optimizer.solve(problem_dict)
    end = time.time()

    computational_time = end - start
    return best_solution.target.fitness, computational_time

# Example usage:
# if __name__ == "__main__":
#     test_scenario = {
#         "nDevices": 5,
#         # Add other scenario parameters here
#     }
#     best_sol, best_fit = runGA(test_scenario)