import os
import subprocess
import sys
import pandas as pd
import concurrent.futures
# Path to ns-3 source and build directories
NS3_SRC = os.path.abspath("ns-3-dev")
NS3_BUILD = os.path.join(NS3_SRC, "build")
ns3_exe = os.path.join(NS3_SRC, "ns3")

def run_cmd(cmd, cwd=None):
    """Run a shell command and print its output."""
   # print(">>> Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result.stdout




def get_PDR_from_ADR(scenario):

    # 1. Ensure build directory exists
    if not os.path.exists(NS3_BUILD):
        os.makedirs(NS3_BUILD)

    # 2. Detect build system
    has_ninja = os.path.exists(os.path.join(NS3_BUILD, "build.ninja"))
    has_make = os.path.exists(os.path.join(NS3_BUILD, "Makefile"))

    # 3. Configure with CMake if no build system
    if not (has_ninja or has_make):
        print("No build system found, running cmake ..")
        run_cmd(["cmake", "..", "-DNS3_EXAMPLES=ON", "-DNS3_TESTS=ON"], cwd=NS3_BUILD)
        has_ninja = os.path.exists(os.path.join(NS3_BUILD, "build.ninja"))
        has_make = os.path.exists(os.path.join(NS3_BUILD, "Makefile"))

    # 4. Build only if needed
    if not os.path.exists(ns3_exe):
        print("ns3 executable not found, building ns-3...")
        if has_ninja:
            run_cmd(["ninja"], cwd=NS3_BUILD)
        elif has_make:
            run_cmd(["make", "-j4"], cwd=NS3_BUILD)
        else:
            print("ERROR: No Ninja or Makefile build system found in", NS3_BUILD)
            sys.exit(1)
    #else:
     #   print("ns3 executable already built, skipping build.")

    # 5. Run simulation

    nDevices = scenario["nDevices"]
    nGateways = scenario["nGateways"]
    radius = scenario["radius"]
    simulationTime = scenario["simulationTime"]
    appPeriod = scenario["appPeriod"]
    payloadSize = scenario["payloadSize"]



    sfs_arg = ""
    adrEnabled = "true"

    # build the ns-3 command string
    cmd_str = (
        "src/lorawan/examples/complete-network-example "
        f"--nDevices={nDevices} "
        f"--nGateways={nGateways} "
        f"--radius={radius} "
        f"--simulationTime={simulationTime} "
        f"--appPeriod={appPeriod} "
        f"--payloadSize={payloadSize} "
        f"{sfs_arg} "
        f"--adrEnabled={adrEnabled} "
    )

    output = run_cmd([ns3_exe, "run", cmd_str], cwd=NS3_BUILD)


    # Extraire TX et RX depuis la dernière ligne
    lines = output.strip().splitlines()

    if lines:
        try:
            line = lines[-1]
            parts = line.split()
            tx = float(parts[0])
            rx = float(parts[1])
            pdr = (rx / tx) if tx > 0 else 0.0
            #print(tx, rx, pdr)
            return pdr
        except ValueError:
            print("Impossible d’extraire les valeurs TX/RX.")


# get results of all scenarios using ADR
def pdr_thread(index, scenario):
    output = get_PDR_from_ADR(scenario)
    return (
        "ADR",
        index,
        scenario["nDevices"],
        scenario["nGateways"],
        scenario["radius"],
        scenario["simulationTime"],
        scenario["appPeriod"],
        scenario["payloadSize"],
        output
    )

def run_adr_results():
    # get the results of ADR for all scenarios in parallel
    df = pd.read_csv("adr_scenarios.csv")

    adr_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(pdr_thread, index, scenario)
            for index, scenario in df.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            adr_results.append(future.result())

    adr_results_df = pd.DataFrame(
        adr_results,
        columns=["algorithm","scenario_index","nDevices","nGateways","radius","simulationTime","appPeriod","payloadSize","pdr"]
    )

    # Sort by original scenario index
    adr_results_df = adr_results_df.sort_values(by=["scenario_index"]).reset_index(drop=True)
    adr_results_df.to_csv("adr_results.csv", index=False)

