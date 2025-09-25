
import pandas as pd
from mealpy import IntegerVar, GA, DE, CSA, PSO, SHADE
from run_simulation import objective_function
import numpy as np
import time
number_runs = 5
NUMBER_OF_WORKERS = 8

def choose_metaheuristic(index_scenario, scenario, all_results, curve_f):
    def objective_func(solution):
        return objective_function(scenario, solution)

    nDevices = scenario["nDevices"]
    problem_dict = {
        "obj_func": objective_func,
        "bounds": IntegerVar(lb=(7,) * nDevices, ub=(12,) * nDevices),
        "minmax": "max",
    }



    # Mapping algorithm name -> (class, kwargs)
    algo_mapping = {
        "GA": (GA.BaseGA, {"epoch": 100, "pop_size": 50, "pc": 0.9, "pm": 0.05, "n_workers": NUMBER_OF_WORKERS}),
        "DE": (DE.OriginalDE, {"epoch": 100, "pop_size": 50, "wf": 0.7, "cr": 0.9, "strategy": 0, "n_workers": NUMBER_OF_WORKERS}),
        "CS": (CSA.OriginalCSA, {"epoch": 100, "pop_size": 50, "p_a": 0.3, "n_workers": NUMBER_OF_WORKERS}),
        "PSO": (PSO.OriginalPSO, {"epoch": 100, "pop_size": 50, "c1": 2.05, "c2": 20.5, "w": 0.4, "n_workers": NUMBER_OF_WORKERS}),
        "SHADE": (SHADE.L_SHADE, {"epoch": 100, "pop_size": 50, "miu_f": 0.5, "miu_cr": 0.5, "n_workers": NUMBER_OF_WORKERS}),
    }

    if scenario["algorithm"] not in algo_mapping:
        raise ValueError(f"Algorithm {scenario['algorithm']} not recognized")

    AlgoClass, kwargs = algo_mapping[scenario["algorithm"]]

    all_curves, all_times, all_solutions = [], [], []

    for r in range(number_runs):
        model = AlgoClass(**kwargs)
        start = time.time()
        g_best = model.solve(problem_dict)
        end = time.time()

        all_curves.append(model.history.list_global_best_fit)
        all_times.append(end - start)
        all_solutions.append(g_best.target.fitness)

    # Convert to numpy arrays
    all_curves = np.array(all_curves)
    all_times = np.array(all_times)
    all_solutions = np.array(all_solutions)

    # Statistics
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)
    mean_time = np.mean(all_times)
    best_solution = np.min(all_solutions)
    mean_solution = np.mean(all_solutions)
    worst_solution = np.max(all_solutions)
    std_solution = np.std(all_solutions)

    # Save summary results
    all_results.append({
        "algorithm": scenario["algorithm"],
        "scenario_index": index_scenario,
        "nDevices": scenario["nDevices"],
        "nGateways": scenario["nGateways"],
        "radius": scenario["radius"],
        "simulationTime": scenario["simulationTime"],
        "appPeriod": scenario["appPeriod"],
        "payloadSize": scenario["payloadSize"],
        "best_solution": best_solution,
        "mean_solution": mean_solution,
        "worst_solution": worst_solution,
        "std_solution": std_solution,
        "mean_time": mean_time
    })

    # Save mean convergence curve to the common txt file
    curve_f.write(f"# Scenario {index_scenario}\n")
    curve_f.write("Mean: " + ",".join(map(str, mean_curve)) + "\n")
    curve_f.write("Std: " + ",".join(map(str, std_curve)) + "\n\n")

    return all_results


