import json
import random
import math
import argparse
import os

def create_circle(x, y, radius):
    return {
        "type": "CircleRegion",
        "center_pos": {"x": round(x, 1), "y": round(y, 1)},
        "radius": round(radius, 1)
    }

def create_polygon(x_center, y_center, radius, num_points=4):
    vertices = []
    # Generate random angles and sort them to form a convex polygon
    angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_points)])
    for angle in angles:
        vx = x_center + radius * math.cos(angle)
        vy = y_center + radius * math.sin(angle)
        vertices.append({"x": round(vx, 1), "y": round(vy, 1)})
        
    return {
        "type": "PolygonRegion",
        "vertices": vertices
    }

def generate_scenario(name, seed, traffic_count, notam_count, static_count, map_size):
    if seed is not None:
        random.seed(seed)
        
    map_max = float(map_size)
    scale = map_max / 40000.0
    
    # Randomize start and goal in opposite corners
    start_corner = random.choice([0, 1, 2, 3])
    margin = 2000 * scale
    corners = [
        (margin, margin),
        (map_max - margin, margin),
        (map_max - margin, map_max - margin),
        (margin, map_max - margin)
    ]
    goal_corner = (start_corner + 2) % 4
    
    start_x = corners[start_corner][0] + random.uniform(-1000 * scale, 1000 * scale)
    start_y = corners[start_corner][1] + random.uniform(-1000 * scale, 1000 * scale)
    
    goal_x = corners[goal_corner][0] + random.uniform(-1000 * scale, 1000 * scale)
    goal_y = corners[goal_corner][1] + random.uniform(-1000 * scale, 1000 * scale)
    
    goal_r = 1000 * scale
    max_time = max(1000, int(3800 * scale))

    # --- PUBLIC JSON ---
    public_data = {
        "scenario_id": name,
        "map_boundaries": {
            "vertices": [
                {"x": 0, "y": 0},
                {"x": map_max, "y": 0},
                {"x": map_max, "y": map_max},
                {"x": 0, "y": map_max}
            ]
        },
        "start_state": {
            "position": {"x": round(start_x, 1), "y": round(start_y, 1)},
            "alt_layer": random.choice([1, 2, 3]),
            "energy": 450.0,
            "velocity": {"x": 0.0, "y": 0.0},
            "heading": 0.0
        },
        "mission_goal": {
            "region": {
                "vertices": [
                    {"x": round(goal_x - goal_r, 1), "y": round(goal_y - goal_r, 1)},
                    {"x": round(goal_x + goal_r, 1), "y": round(goal_y - goal_r, 1)},
                    {"x": round(goal_x + goal_r, 1), "y": round(goal_y + goal_r, 1)},
                    {"x": round(goal_x - goal_r, 1), "y": round(goal_y + goal_r, 1)}
                ]
            },
            "target_alt_layer": random.choice([1, 2, 3])
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
            "max_time": max_time
        },
        "permanent_constraints": [],
        "static_obstacles": [],
        "emergency_landing_sites": [
            {
                "id": "em_site_center",
                "region": {
                    "vertices": [
                        {"x": round(map_max/2 - 500*scale, 1), "y": round(map_max/2 - 500*scale, 1)},
                        {"x": round(map_max/2 + 500*scale, 1), "y": round(map_max/2 - 500*scale, 1)},
                        {"x": round(map_max/2 + 500*scale, 1), "y": round(map_max/2 + 500*scale, 1)},
                        {"x": round(map_max/2 - 500*scale, 1), "y": round(map_max/2 + 500*scale, 1)}
                    ]
                }
            }
        ]
    }

    # Generate Static Obstacles
    for i in range(static_count):
        ox = random.uniform(5000 * scale, map_max - 5000 * scale)
        oy = random.uniform(5000 * scale, map_max - 5000 * scale)
        radius = random.uniform(500 * scale, 2500 * scale)
        
        if random.random() > 0.5:
            obs = create_circle(ox, oy, radius)
        else:
            obs = create_polygon(ox, oy, radius, random.randint(3, 6))
            
        public_data["static_obstacles"].append(obs)

    # --- HIDDEN JSON ---
    hidden_data = {
        "shrinking_notams": [],
        "traffic_traces": []
    }

    # Generate Shrinking NOTAMs
    for i in range(notam_count):
        nx = random.uniform(2000 * scale, map_max - 2000 * scale)
        ny = random.uniform(2000 * scale, map_max - 2000 * scale)
        radius = random.uniform(1000 * scale, 3000 * scale)
        
        if random.random() > 0.5:
            region = create_circle(nx, ny, radius)
        else:
            region = create_polygon(nx, ny, radius, random.randint(4, 8))
            
        adv_start = random.randint(0, int(1000 * scale))
        ctrl_start = adv_start + random.randint(int(100 * scale), int(500 * scale))
        rest_start = ctrl_start + random.randint(int(100 * scale), int(500 * scale))
        
        notam = {
            "id": f"notam_{i}",
            "region": region,
            "alt_layers": random.sample([1, 2, 3, 4], k=random.randint(1, 3)),
            "advisory_start_time": adv_start,
            "controlled_start_time": ctrl_start,
            "restricted_start_time": rest_start
        }
        hidden_data["shrinking_notams"].append(notam)

    # Generate Traffic Traces
    for i in range(traffic_count):
        # Start traffic randomly along the edges of the map
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            tx, ty = random.uniform(0, map_max), map_max - 100
            vx, vy = random.uniform(-10, 10), random.uniform(-10, -3)
        elif edge == "bottom":
            tx, ty = random.uniform(0, map_max), 100
            vx, vy = random.uniform(-10, 10), random.uniform(3, 10)
        elif edge == "left":
            tx, ty = 100, random.uniform(0, map_max)
            vx, vy = random.uniform(3, 10), random.uniform(-10, 10)
        else: # right
            tx, ty = map_max - 100, random.uniform(0, map_max)
            vx, vy = random.uniform(-10, -3), random.uniform(-10, 10)
            
        # Normalize velocity to reasonable drone speeds (5-15 m/s)
        speed = random.uniform(5.0, 14.0)
        mag = math.hypot(vx, vy)
        vx = round((vx / mag) * speed, 2)
        vy = round((vy / mag) * speed, 2)

        trace = {
            "id": f"traffic_{i}",
            "segments": [
                {
                    "start_time": 0,
                    "end_time": max_time,
                    "start_pos": {"x": round(tx, 1), "y": round(ty, 1)},
                    "velocity": {"x": vx, "y": vy},
                    "alt_layer": random.choice([1, 2, 3, 4])
                }
            ]
        }
        hidden_data["traffic_traces"].append(trace)

    # Ensure directories exist
    os.makedirs("scenarios/public", exist_ok=True)
    os.makedirs("scenarios/hidden", exist_ok=True)
    
    public_path = f"scenarios/public/{name}.json"
    hidden_path = f"scenarios/hidden/{name}.json"

    with open(public_path, "w") as f:
        json.dump(public_data, f, indent=2)
        
    with open(hidden_path, "w") as f:
        json.dump(hidden_data, f, indent=2)
        
    print(f"✅ Generated scenario '{name}' successfully with map size {map_max}x{map_max}.")
    print(f"  - {public_path}")
    print(f"  - {hidden_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RTM Challenge Scenarios")
    parser.add_argument("--name", type=str, required=True, help="Base name of the scenario")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--traffic-count", type=int, default=5, help="Number of dynamic traffic traces")
    parser.add_argument("--notam-count", type=int, default=5, help="Number of shrinking NOTAMs")
    parser.add_argument("--static-count", type=int, default=3, help="Number of static obstacles")
    parser.add_argument("--map-size", type=float, default=10000.0, help="Size of the map boundaries (square)")
    
    args = parser.parse_args()
    
    generate_scenario(
        name=args.name,
        seed=args.seed,
        traffic_count=args.traffic_count,
        notam_count=args.notam_count,
        static_count=args.static_count,
        map_size=args.map_size
    )
