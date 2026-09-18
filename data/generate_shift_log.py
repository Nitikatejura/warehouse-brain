"""
Generates 100 realistic warehouse shift events for the integrated control loop.
"""

import json
import random

def generate_shift_log(output_path: str = "data/shift_stream_100.json", num_events: int = 100):
    random.seed(42)
    events = []

    for i in range(1, num_events + 1):
        # Distribute event types:
        # 45% Box Inspections (Module B -> Module E)
        # 30% Telemetry Streams (Module D RNN + Hopfield)
        # 15% Pallet Stacking Orders (Module A Planning & Reactive Layer)
        # 10% Low Battery Dock Contention Races (Module C Game Playing)
        roll = random.random()

        if roll < 0.45:
            is_damaged = random.random() < 0.25
            event = {
                "event_id": f"EVT_{i:03d}",
                "timestamp": f"08:{i//2:02d}:{random.randint(0,59):02d}",
                "type": "BOX_INSPECTION",
                "box_id": f"BX-{random.randint(100, 999)}",
                "weight_anomaly": is_damaged and random.random() < 0.85,
                "weight_deviation_kg": round(random.uniform(5.5, 9.5) if is_damaged else random.uniform(0.1, 1.2), 2),
                "crack_detected": is_damaged and random.random() < 0.90,
                "crack_severity_score": round(random.uniform(6.0, 9.8) if is_damaged else random.uniform(0.0, 1.5), 1),
                "tilt_triggered": is_damaged and random.random() < 0.70,
                "sensor_noise": "High" if random.random() < 0.15 else "Low"
            }
        elif roll < 0.75:
            # Telemetry stream for Module D
            has_anomaly = random.random() < 0.20
            # 10 steps of [vib, temp, curr, drift]
            seq = []
            for step in range(10):
                if has_anomaly and step >= 6:
                    vib = round(random.uniform(0.8, 1.8), 3)
                    temp = round(random.uniform(60.0, 85.0), 2)
                    curr = round(random.uniform(5.5, 8.0), 2)
                    drift = round(random.uniform(0.3, 0.7), 3)
                else:
                    vib = round(random.uniform(0.1, 0.3), 3)
                    temp = round(random.uniform(42.0, 48.0), 2)
                    curr = round(random.uniform(2.8, 3.4), 2)
                    drift = round(random.uniform(-0.05, 0.05), 3)
                seq.append([vib, temp, curr, drift])

            event = {
                "event_id": f"EVT_{i:03d}",
                "timestamp": f"08:{i//2:02d}:{random.randint(0,59):02d}",
                "type": "TELEMETRY_STREAM",
                "robot_id": "AMR-01",
                "sensor_sequence": seq,
                "barcode_scan_noisy": random.random() < 0.35,
                "label_type": random.choice(["HAZARD_CROSS", "PALLET_BORDER", "FRAGILE_PLUS", "ARROW_UP"])
            }
        elif roll < 0.90:
            # Pallet Stacking for Module A
            has_disturbance = random.random() < 0.40
            event = {
                "event_id": f"EVT_{i:03d}",
                "timestamp": f"08:{i//2:02d}:{random.randint(0,59):02d}",
                "type": "PALLET_STACK_TASK",
                "order_id": f"ORD-{random.randint(1000, 9999)}",
                "pallet_id": f"PLT-{random.randint(1, 5)}",
                "has_disturbance": has_disturbance,
                "disturbance_step": random.randint(2, 4) if has_disturbance else None
            }
        else:
            # Battery Low Dock Contention for Module C
            event = {
                "event_id": f"EVT_{i:03d}",
                "timestamp": f"08:{i//2:02d}:{random.randint(0,59):02d}",
                "type": "BATTERY_LOW_DOCK_RACE",
                "robot_battery_pct": round(random.uniform(5.0, 14.0), 1),
                "competitor_robot_id": "RIVAL_BOT_9"
            }

        events.append(event)

    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Generated {len(events)} shift events in '{output_path}'.")

if __name__ == "__main__":
    generate_shift_log()
