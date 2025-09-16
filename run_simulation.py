import os
import subprocess
import sys


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




def objective_function(scenario, sf_solution):

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
    enableAdr = scenario["enableAdr"]



    sfs_arg = ""
    sf_solution = list(sf_solution)
    if sf_solution and len(sf_solution) == nDevices:
        sfs_str = " ".join(str(int(x)) for x in sf_solution)
        sfs_arg = f"--sfs={sfs_str}"
    elif sf_solution:
        raise ValueError(f"Length of sf_solution ({len(sf_solution)}) does not match nDevices ({nDevices})")


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
        f"--enableAdr={enableAdr}"
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



