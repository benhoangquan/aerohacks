import json
import os

def write_scenario(name, public_data, hidden_data):
    os.makedirs("scenarios/public", exist_ok=True)
    os.makedirs("scenarios/hidden", exist_ok=True)
    with open(f"scenarios/public/{name}.json", "w") as f:
        json.dump(public_data, f, indent=2)
    with open(f"scenarios/hidden/{name}.json", "w") as f:
        json.dump(hidden_data, f, indent=2)
    print(f"✅ Generated {name}")

def get_base_public(name, energy=450.0):
    return {
        "scenario_id": name,
        "map_boundaries": {
            "vertices": [{"x": 0, "y": 0}, {"x": 10000, "y": 0}, {"x": 10000, "y": 10000}, {"x": 0, "y": 10000}]
        },
        "start_state": {
            "position": {"x": 1000.0, "y": 5000.0},
            "alt_layer": 2,
            "energy": energy,
            "velocity": {"x": 0.0, "y": 0.0},
            "heading": 0.0
        },
        "mission_goal": {
            "region": {
                "vertices": [{"x": 8000, "y": 4000}, {"x": 9000, "y": 4000}, {"x": 9000, "y": 6000}, {"x": 8000, "y": 6000}]
            },
            "target_alt_layer": 2
        },
        "vehicle_limits": {
            "max_horizontal_speed": 15.0,
            "max_vertical_rate": 1,
            "energy_decay_rate": 0.1,
            "energy_reserve_threshold": 20.0
        },
        "scoring_config": {
            "base_success_score": 1000.0,
            "safe_emergency_landing_score": 300.0,
            "controlled_phase_penalty": -50.0,
            "restricted_phase_penalty": -200.0,
            "separation_loss_penalty_advisory": -10.0,
            "separation_loss_penalty_conflict": -150.0,
            "collision_penalty": -500.0,
            "battery_breach_penalty": -500.0,
            "time_bonus_multiplier": 10.0,
            "max_time": 1500
        },
        "permanent_constraints": [],
        "static_obstacles": [],
        "emergency_landing_sites": [
            {
                "id": "em_site_1",
                "region": {
                    "vertices": [{"x": 1500, "y": 4500}, {"x": 2500, "y": 4500}, {"x": 2500, "y": 5500}, {"x": 1500, "y": 5500}]
                }
            }
        ]
    }

def get_base_hidden():
    return {
        "shrinking_notams": [],
        "traffic_traces": []
    }

def generate_battery_edge_case():
    # Low energy. Can't reach the goal. Must detour to emergency site.
    public = get_base_public("edge_battery", energy=35.0) # 35 energy = 350 ticks = 5250m max distance. Goal is 8000m away.
    hidden = get_base_hidden()
    write_scenario("edge_battery", public, hidden)

def generate_impossible_edge_case():
    # A massive wall from y=0 to 10000 blocks the entire map across all altitudes 1-4.
    public = get_base_public("edge_impossible")
    public["permanent_constraints"].append({
        "id": "impossible_wall",
        "region": {
            "type": "PolygonRegion",
            "vertices": [{"x": 4000, "y": 0}, {"x": 6000, "y": 0}, {"x": 6000, "y": 10000}, {"x": 4000, "y": 10000}]
        },
        "alt_layers": [1, 2, 3, 4]
    })
    hidden = get_base_hidden()
    write_scenario("edge_impossible", public, hidden)

def generate_altitude_edge_case():
    # A wall that only blocks altitudes 2 and 3. The drone starts at 2, goal is at 2.
    # It MUST change altitude to 1 (under) or 4 (over) to bypass it.
    public = get_base_public("edge_altitude")
    public["permanent_constraints"].append({
        "id": "altitude_wall",
        "region": {
            "type": "PolygonRegion",
            "vertices": [{"x": 4000, "y": 0}, {"x": 6000, "y": 0}, {"x": 6000, "y": 10000}, {"x": 4000, "y": 10000}]
        },
        "alt_layers": [2, 3] # Leaves Alt 1 and Alt 4 open!
    })
    hidden = get_base_hidden()
    write_scenario("edge_altitude", public, hidden)
    
def generate_collision_avoidance_case():
    # Drone flies straight east. Traffic drone flies straight west, head on at Alt 2!
    public = get_base_public("edge_collision")
    hidden = get_base_hidden()
    hidden["traffic_traces"].append({
        "id": "head_on_traffic",
        "segments": [{
            "start_time": 0,
            "end_time": 1500,
            "start_pos": {"x": 9000, "y": 5000},
            "velocity": {"x": -10.0, "y": 0.0},
            "alt_layer": 2
        }]
    })
    write_scenario("edge_collision", public, hidden)

if __name__ == "__main__":
    generate_battery_edge_case()
    generate_impossible_edge_case()
    generate_altitude_edge_case()
    generate_collision_avoidance_case()
