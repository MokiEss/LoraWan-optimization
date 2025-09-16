scenarioN = 0
from mealpy import IntegerVar, GA
from run_simulation import objective_function
import pandas as pd


df = pd.read_csv("scenarios.csv", sep=";")
nDevices = df.iloc[scenarioN]["nDevices"]
scenario = df.iloc[scenarioN]

def objective_func(solution):
    global scenarioN
    global scenario
    return objective_function(scenario, solution)


problem_dict = {
    "obj_func": objective_func,
    "bounds": IntegerVar(lb=[7, ] * nDevices, ub=[12, ] * nDevices,),
    "minmax": "max",
}

optimizer = GA.BaseGA(epoch=10, pop_size=50, pc=0.85, pm=0.1, verbose=True)
optimizer.solve(problem_dict)

print(optimizer.g_best.solution)
print(optimizer.g_best.target.fitness)