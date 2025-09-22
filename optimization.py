scenarioN = 0
from mealpy import IntegerVar, GA, SA
from run_simulation import objective_function
import numpy as np
import pandas as pd


df = pd.read_csv("scenarios.csv", sep=";")
nDevices = df.iloc[scenarioN]["nDevices"]
scenario = df.iloc[scenarioN]

def objective_func(solution):
    global scenarioN
    global scenario
    mapped_values = np.interp(solution, [0, 1], [7, 12]).astype(int)
    fitness = objective_function(scenario, mapped_values)
    print("mapped_values:", mapped_values)
    return fitness


problem_dict = {
    "obj_func": objective_func,
    "bounds": IntegerVar(lb=[0, ] * nDevices, ub=[1, ] * nDevices,),
    "minmax": "max",
}

NUMBER_OF_WORKERS = 8
model = SA.GaussianSA(epoch=10, pop_size=2, temp_init = 100, cooling_rate = 0.99, scale = 0.1, n_workers = NUMBER_OF_WORKERS)

g_best = model.solve(problem_dict)

print(f"Solution: {np.interp(g_best.solution, [0, 1], [7, 12]).astype(int)}, Fitness: {g_best.target.fitness}")

print(f"Solution: {np.interp(model.g_best.solution, [0, 1], [7, 12]).astype(int)}, Fitness: {model.g_best.target.fitness}")