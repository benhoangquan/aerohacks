import os
import sys
import json
import argparse
import subprocess
import csv
import re
import shutil

def main():
    parser = argparse.ArgumentParser(description="Generate, run, and log RTM Challenge scenarios.")
    parser.add_argument("--name", type=str, required=True, help="Base name of the scenario")
    parser.add_argument("--policy", type=str, default="./dstar_lite_algo", help="Path to the policy folder")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--traffic-count", type=int, default=5, help="Number of dynamic traffic traces")
    parser.add_argument("--notam-count", type=int, default=5, help="Number of shrinking NOTAMs")
    parser.add_argument("--static-count", type=int, default=3, help="Number of static obstacles")
    parser.add_argument("--map-size", type=float, default=1000.0, help="Size of the map boundaries (square)")
    parser.add_argument("--csv-file", type=str, default="results.csv", help="CSV file to append results to")
    
    args = parser.parse_args()

    # 1. Generate the scenario
    print(f"[*] Generating scenario '{args.name}'...")
    gen_cmd = [
        sys.executable, "generate_scenario.py",
        "--name", args.name,
        "--traffic-count", str(args.traffic_count),
        "--notam-count", str(args.notam_count),
        "--static-count", str(args.static_count),
        "--map-size", str(args.map_size)
    ]
    if args.seed is not None:
        gen_cmd.extend(["--seed", str(args.seed)])
        
    gen_res = subprocess.run(gen_cmd, capture_output=True, text=True)
    if gen_res.returncode != 0:
        print(f"[!] Error generating scenario:\n{gen_res.stderr}")
        return
        
    print(f"[*] Scenario '{args.name}' generated.")

    # 2. Run the simulator
    print(f"[*] Running simulator with policy '{args.policy}' on scenario '{args.name}'...")
    sim_cmd = [
        "./AeroHacksSim-mac",
        "--policy", args.policy,
        "--scenario", args.name
    ]
    
    # Passing the current environment (important if the user needs specific env vars)
    env = os.environ.copy()
    
    sim_res = subprocess.run(sim_cmd, env=env, capture_output=True, text=True)
    
    # 3. Parse the score from standard output
    score = None
    output = sim_res.stdout + "\n" + sim_res.stderr
    
    # Look for common score patterns in output
    score_match = re.search(r'(?:final\s+)?score[\s:=]+([-\d\.]+)', output, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
    else:
        print("[!] Could not parse score from console output. Simulator output was:")
        print(output[-1000:]) # print last 1000 chars just in case
        score = "ERROR"

    # 4. Process playback.json
    final_time = None
    final_energy = None
    playback_src = "playback.json"
    playback_dest = f"playback/playback_{args.name}.json"
    
    os.makedirs("playback", exist_ok=True)

    if os.path.exists(playback_src):
        try:
            with open(playback_src, "r") as f:
                playback_data = json.load(f)
                
            if isinstance(playback_data, list) and len(playback_data) > 0:
                last_frame = playback_data[-1]
                final_time = last_frame.get("time")
                final_energy = last_frame.get("energy")
                if final_energy is not None:
                    final_energy = round(final_energy, 2)
                    
            shutil.move(playback_src, playback_dest)
            print(f"[*] Playback saved to {playback_dest}")
        except Exception as e:
            print(f"[!] Error reading/moving {playback_src}: {e}")
    else:
        print(f"[!] {playback_src} was not found! The simulator may have crashed.")

    # 5. Write to CSV
    file_exists = os.path.isfile(args.csv_file)
    
    with open(args.csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Scenario Name", "Map Size", "Traffic Count", "NOTAM Count", 
                "Static Count", "Seed", "Score", "Final Energy", "Final Time"
            ])
        writer.writerow([
            args.name, args.map_size, args.traffic_count, args.notam_count,
            args.static_count, args.seed if args.seed else "None", 
            score, final_energy, final_time
        ])
        
    print(f"[*] Results appended to {args.csv_file}")
    print(f"    -> Score: {score}, Energy: {final_energy}, Time: {final_time}")

if __name__ == "__main__":
    main()
