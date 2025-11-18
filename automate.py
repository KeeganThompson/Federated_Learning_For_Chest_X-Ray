import argparse
import subprocess
from pathlib import Path
from train_hospital import run_local
from eval_local import eval_local
from eval_global_new import eval_global_all

def automate_training(algorithm="fedavg", num_rounds=3):
    """
    Automate federated learning training with specified aggregation algorithm.
    
    Args:
        algorithm: Either "fedavg" or "fedprox"
        num_rounds: Number of federated learning rounds to run
    """
    num_hospitals = 3
    
    for round_id in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_id}/{num_rounds} - Training local models")
        print(f"{'='*60}\n")
        
        # Train local models for each hospital
        for hospital_id in range(1, num_hospitals + 1):
            print(f"Training hospital {hospital_id}...")
            run_local(hospital_id, round_id)
            
            # Evaluate local model after training
            print(f"\nEvaluating local model for hospital {hospital_id}...")
            try:
                eval_local(hospital_id, round_id)
            except Exception as e:
                print(f"⚠️ Warning: Local evaluation failed for hospital {hospital_id}: {e}")
        
        # Aggregate models after each round
        print(f"\n{'='*60}")
        print(f"Round {round_id}/{num_rounds} - Aggregating models using {algorithm.upper()}")
        print(f"{'='*60}\n")
        
        # Collect checkpoint paths
        checkpoint_paths = [
            f"hospital{hospital_id}_densenet_torchvision{round_id}.pt"
            for hospital_id in range(1, num_hospitals + 1)
        ]
        
        # Verify checkpoints exist
        missing = [cp for cp in checkpoint_paths if not Path(cp).exists()]
        if missing:
            print(f"⚠️ Warning: Missing checkpoints: {missing}")
            continue
        
        # Determine output path for global model
        global_output = f"global/round{round_id + 1}_global.pt"
        Path("global").mkdir(exist_ok=True)
        
        # Call the appropriate aggregator
        aggregator_script = f"{algorithm}_aggregator.py"
        cmd = [
            "python", aggregator_script,
            "--ckpts"] + checkpoint_paths + [
            "--out", global_output
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running aggregator: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            continue
        
        print(f"✅ Round {round_id} completed. Global model saved to {global_output}\n")
        
        # Evaluate global model on all hospitals after aggregation
        print(f"\n{'='*60}")
        print(f"Round {round_id}/{num_rounds} - Evaluating global model")
        print(f"{'='*60}\n")
        
        try:
            hospital_ids = list(range(1, num_hospitals + 1))
            # Evaluate global model for the next round (since we just created round{round_id+1}_global.pt)
            eval_global_all(round_id + 1, hospital_ids)
        except Exception as e:
            print(f"⚠️ Warning: Global evaluation failed: {e}")
        
        print(f"\n✅ Round {round_id} evaluation completed.\n")

def main():
    parser = argparse.ArgumentParser(
        description="Automate federated learning training with FedAvg or FedProx aggregation"
    )
    parser.add_argument(
        "algorithm",
        choices=["fedavg", "fedprox"],
        help="Aggregation algorithm to use: 'fedavg' or 'fedprox'"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="Number of federated learning rounds to run (default: 3)"
    )
    args = parser.parse_args()
    
    if args.rounds < 1:
        parser.error("Number of rounds must be at least 1")
    
    automate_training(algorithm=args.algorithm.lower(), num_rounds=args.rounds)

if __name__ == "__main__":
    main()