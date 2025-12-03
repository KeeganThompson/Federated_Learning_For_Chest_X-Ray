"""
Function to compare local and global model predictions and count cases where
local model is incorrect but global model is correct.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ===== Dataset =====
class CheXpertDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.transform = transform
        self.label_cols = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        # NaN, -1 (uncertain) -> 0
        self.df[self.label_cols] = self.df[self.label_cols].fillna(0).replace(-1, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_path = row["Path"]                
        img_path = Path(raw_path)

        parts = img_path.parts
        if "train" in parts:
            train_idx = parts.index("train")
            img_path = Path(*parts[train_idx:])
        full_path = self.img_root / img_path

        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_cols].values.astype("float32"))
        return image, labels, idx  # Return index for tracking


def load_model(checkpoint_path, device=None):
    """Load a DenseNet121 model from checkpoint."""
    if device is None:
        device = DEVICE
    
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
        nn.Sigmoid()
    )
    
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None


def compare_local_vs_global(hospital_id, round_id=3, threshold=0.5, hospital_dir="hospitals_split", img_root=None, batch_size=32):
    """
    Compare local and global model predictions and count cases where:
    - Local model is incorrect
    - Global model is correct
    
    Args:
        hospital_id: Hospital ID (integer, e.g., 1, 2, 3)
        round_id: Round number (default: 3)
        threshold: Threshold for binary classification (default: 0.5)
        hospital_dir: Directory containing hospital CSV files (default: "hospitals_split")
        img_root: Root directory for images (default: Path.cwd())
        batch_size: Batch size for evaluation (default: 32)
    
    Returns:
        dict: Dictionary containing comparison metrics
    """
    if img_root is None:
        img_root = Path.cwd()
    else:
        img_root = Path(img_root)
    
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Comparing local vs global models for hospital {hospital_id}, round {round_id}")
    
    # Load models
    local_ckpt = f"hospital{hospital_id}_densenet_torchvision{round_id}.pt"
    global_ckpt = f"global/round{round_id}_global.pt"
    
    print(f"[INFO] Loading local model from: {local_ckpt}")
    local_model = load_model(local_ckpt, DEVICE)
    if local_model is None:
        return None
    
    print(f"[INFO] Loading global model from: {global_ckpt}")
    global_model = load_model(global_ckpt, DEVICE)
    if global_model is None:
        return None
    
    # Load dataset
    csv_path = f"{hospital_dir}/{hospital_id}.csv"
    print(f"[INFO] Loading data from: {csv_path}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = CheXpertDataset(csv_path, img_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"[INFO] Dataset size: {len(dataset)} samples")
    
    # Collect predictions and ground truth
    all_local_preds = []
    all_global_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for imgs, labels, indices in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Get predictions from both models
            local_outputs = local_model(imgs)
            global_outputs = global_model(imgs)
            
            # Convert to binary predictions using threshold
            local_preds = (local_outputs > threshold).float()
            global_preds = (global_outputs > threshold).float()
            
            all_local_preds.append(local_preds.cpu().numpy())
            all_global_preds.append(global_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_indices.append(indices.numpy())
    
    # Concatenate all results
    all_local_preds = np.concatenate(all_local_preds, axis=0)
    all_global_preds = np.concatenate(all_global_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    
    # Compare predictions
    # For each sample, check if local is wrong and global is right
    # We'll count at the sample level (if any label is wrong in local but correct in global)
    
    local_correct = (all_local_preds == all_labels)
    global_correct = (all_global_preds == all_labels)
    local_wrong = ~local_correct
    global_wrong = ~global_correct
    
    # Cases where local is wrong but global is correct (for each label)
    local_wrong_global_correct = local_wrong & global_correct
    
    # Count samples where at least one label fits the criteria
    samples_with_improvement = np.any(local_wrong_global_correct, axis=1)
    num_samples_improved = np.sum(samples_with_improvement)
    
    # Count per-label improvements
    per_label_improvements = np.sum(local_wrong_global_correct, axis=0)
    
    # Also count the reverse: global wrong but local correct
    global_wrong_local_correct = global_wrong & local_correct
    samples_with_regression = np.any(global_wrong_local_correct, axis=1)
    num_samples_regressed = np.sum(samples_with_regression)
    per_label_regressions = np.sum(global_wrong_local_correct, axis=0)
    
    # Calculate accuracy for both models
    local_accuracy = np.mean(local_correct)
    global_accuracy = np.mean(global_correct)
    
    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS: Local vs Global Model")
    print("="*60)
    print(f"\nHospital {hospital_id} - Round {round_id}")
    print(f"Total samples: {len(all_labels)}")
    print(f"\nModel Accuracies:")
    print(f"  Local model accuracy:  {local_accuracy:.4f}")
    print(f"  Global model accuracy: {global_accuracy:.4f}")
    print(f"  Improvement: {global_accuracy - local_accuracy:.4f}")
    
    print(f"\nSamples where Global is better than Local:")
    print(f"  Number of samples improved: {num_samples_improved} ({num_samples_improved/len(all_labels)*100:.2f}%)")
    print(f"  Number of samples regressed: {num_samples_regressed} ({num_samples_regressed/len(all_labels)*100:.2f}%)")
    
    print(f"\nPer-label improvements (Local wrong → Global correct):")
    label_cols = dataset.label_cols
    for i, label in enumerate(label_cols):
        print(f"  {label:<30}: {per_label_improvements[i]:4d} labels")
    
    print(f"\nPer-label regressions (Global wrong → Local correct):")
    for i, label in enumerate(label_cols):
        print(f"  {label:<30}: {per_label_regressions[i]:4d} labels")
    
    print("="*60)
    
    # Prepare data for CSV
    label_cols = dataset.label_cols
    
    # Create summary row
    summary_data = {
        'Hospital_ID': hospital_id,
        'Round_ID': round_id,
        'Total_Samples': len(all_labels),
        'Local_Accuracy': local_accuracy,
        'Global_Accuracy': global_accuracy,
        'Accuracy_Improvement': global_accuracy - local_accuracy,
        'Num_Samples_Improved': num_samples_improved,
        'Num_Samples_Regressed': num_samples_regressed,
        'Pct_Samples_Improved': num_samples_improved/len(all_labels)*100,
        'Pct_Samples_Regressed': num_samples_regressed/len(all_labels)*100
    }
    
    # Create per-label improvement rows
    per_label_data = []
    for i, label in enumerate(label_cols):
        per_label_data.append({
            'Hospital_ID': hospital_id,
            'Round_ID': round_id,
            'Label': label,
            'Local_Wrong_Global_Correct': per_label_improvements[i],
            'Global_Wrong_Local_Correct': per_label_regressions[i],
            'Net_Improvement': per_label_improvements[i] - per_label_regressions[i]
        })
    
    # Save summary to CSV
    summary_df = pd.DataFrame([summary_data])
    summary_filename = f"sample_evaluation_summary_hospital{hospital_id}_round{round_id}.csv"
    try:
        summary_df.to_csv(summary_filename, index=False)
        print(f"\n[INFO] Summary results saved to: {summary_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save summary CSV: {e}")
    
    # Save per-label details to CSV
    per_label_df = pd.DataFrame(per_label_data)
    per_label_filename = f"sample_evaluation_per_label_hospital{hospital_id}_round{round_id}.csv"
    try:
        per_label_df.to_csv(per_label_filename, index=False)
        print(f"[INFO] Per-label results saved to: {per_label_filename}")
        
        # Create per-label plot
        plot_per_label_improvements(hospital_id, round_id, per_label_data)
    except Exception as e:
        print(f"[ERROR] Failed to save per-label CSV: {e}")
    
    # Return summary
    result = {
        'hospital_id': hospital_id,
        'round_id': round_id,
        'total_samples': len(all_labels),
        'local_accuracy': local_accuracy,
        'global_accuracy': global_accuracy,
        'accuracy_improvement': global_accuracy - local_accuracy,
        'num_samples_improved': num_samples_improved,
        'num_samples_regressed': num_samples_regressed,
        'per_label_improvements': dict(zip(label_cols, per_label_improvements.tolist())),
        'per_label_regressions': dict(zip(label_cols, per_label_regressions.tolist())),
        'summary_file': summary_filename,
        'per_label_file': per_label_filename
    }
    
    return result


def compare_across_rounds(hospital_id, rounds=[1, 2, 3], hospital_dir="hospitals_split", img_root=None, batch_size=32):
    """
    Compare local vs global model improvements across multiple rounds for a hospital.
    Tracks how the global model improves upon incorrectly classified samples per round.
    
    Args:
        hospital_id: Hospital ID (integer)
        rounds: List of round IDs to compare (default: [1, 2, 3])
        hospital_dir: Directory containing hospital CSV files
        img_root: Root directory for images
        batch_size: Batch size for evaluation
    
    Returns:
        pd.DataFrame: DataFrame with per-round comparison results
    """
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"COMPARING ACROSS ROUNDS FOR HOSPITAL {hospital_id}")
    print(f"{'='*60}\n")
    
    for round_id in rounds:
        print(f"\nProcessing Round {round_id}...")
        result = compare_local_vs_global(hospital_id, round_id, hospital_dir=hospital_dir, 
                                        img_root=img_root, batch_size=batch_size)
        if result:
            all_results.append({
                'Hospital_ID': hospital_id,
                'Round_ID': round_id,
                'Total_Samples': result['total_samples'],
                'Local_Accuracy': result['local_accuracy'],
                'Global_Accuracy': result['global_accuracy'],
                'Accuracy_Improvement': result['accuracy_improvement'],
                'Num_Samples_Improved': result['num_samples_improved'],
                'Num_Samples_Regressed': result['num_samples_regressed'],
                'Pct_Samples_Improved': result['num_samples_improved']/result['total_samples']*100,
                'Pct_Samples_Regressed': result['num_samples_regressed']/result['total_samples']*100
            })
    
    if all_results:
        # Create DataFrame with per-round results
        rounds_df = pd.DataFrame(all_results)
        
        # Save cross-round comparison
        cross_round_filename = f"sample_evaluation_cross_rounds_hospital{hospital_id}.csv"
        try:
            rounds_df.to_csv(cross_round_filename, index=False)
            print(f"\n[INFO] Cross-round comparison saved to: {cross_round_filename}")
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"SUMMARY: Hospital {hospital_id} - Improvements Across Rounds")
            print(f"{'='*60}")
            print(rounds_df.to_string(index=False))
            print(f"{'='*60}\n")
            
            # Create plots
            plot_improvements(rounds_df, hospital_id)
            
        except Exception as e:
            print(f"[ERROR] Failed to save cross-round CSV: {e}")
        
        return rounds_df
    
    return None


def plot_improvements(rounds_df, hospital_id):
    """
    Create plots showing improvements and regressions across rounds.
    
    Args:
        rounds_df: DataFrame with per-round comparison results
        hospital_id: Hospital ID for labeling
    """
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Hospital {hospital_id}: Local vs Global Model Comparison Across Rounds', 
                     fontsize=16, fontweight='bold')
        
        rounds = rounds_df['Round_ID'].values
        
        # Plot 1: Improvements vs Regressions (Bar chart)
        ax1 = axes[0, 0]
        x = np.arange(len(rounds))
        width = 0.35
        ax1.bar(x - width/2, rounds_df['Num_Samples_Improved'], width, 
                label='Improved', color='green', alpha=0.7)
        ax1.bar(x + width/2, rounds_df['Num_Samples_Regressed'], width, 
                label='Regressed', color='red', alpha=0.7)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Samples Improved vs Regressed')
        ax1.set_xticks(x)
        ax1.set_xticklabels(rounds)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Accuracy Comparison (Line plot)
        ax2 = axes[0, 1]
        ax2.plot(rounds, rounds_df['Local_Accuracy'], marker='o', label='Local Model', 
                linewidth=2, markersize=8)
        ax2.plot(rounds, rounds_df['Global_Accuracy'], marker='s', label='Global Model', 
                linewidth=2, markersize=8)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy Across Rounds')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Plot 3: Accuracy Improvement (Bar chart)
        ax3 = axes[1, 0]
        colors = ['green' if x > 0 else 'red' for x in rounds_df['Accuracy_Improvement']]
        ax3.bar(rounds, rounds_df['Accuracy_Improvement'], color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Accuracy Improvement')
        ax3.set_title('Global Model Accuracy Improvement')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Percentage Improved vs Regressed (Stacked area or bar)
        ax4 = axes[1, 1]
        ax4.bar(rounds, rounds_df['Pct_Samples_Improved'], label='% Improved', 
               color='green', alpha=0.7)
        ax4.bar(rounds, -rounds_df['Pct_Samples_Regressed'], label='% Regressed', 
               color='red', alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Percentage of Samples')
        ax4.set_title('Percentage Improved vs Regressed')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"sample_evaluation_plots_hospital{hospital_id}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plots saved to: {plot_filename}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Failed to create plots: {e}")
        import traceback
        traceback.print_exc()


def plot_per_label_improvements(hospital_id, round_id, per_label_data):
    """
    Create a plot showing per-label improvements and regressions.
    
    Args:
        hospital_id: Hospital ID
        round_id: Round ID
        per_label_data: List of dictionaries with per-label data
    """
    try:
        df = pd.DataFrame(per_label_data)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['Local_Wrong_Global_Correct'], width, 
               label='Local Wrong → Global Correct', color='green', alpha=0.7)
        ax.bar(x + width/2, df['Global_Wrong_Local_Correct'], width, 
               label='Global Wrong → Local Correct', color='red', alpha=0.7)
        
        ax.set_xlabel('Label')
        ax.set_ylabel('Number of Cases')
        ax.set_title(f'Hospital {hospital_id} - Round {round_id}: Per-Label Improvements')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Label'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plot_filename = f"sample_evaluation_per_label_plots_hospital{hospital_id}_round{round_id}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"[INFO] Per-label plot saved to: {plot_filename}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Failed to create per-label plot: {e}")


def run_all_hospitals(hospital_ids=[1, 2, 3, 4], rounds=[1, 2, 3], hospital_dir="hospitals_split", img_root=None, batch_size=32):
    """
    Run comparison for all hospitals and save results incrementally.
    
    Args:
        hospital_ids: List of hospital IDs to process (default: [1, 2, 3, 4])
        rounds: List of round IDs to compare (default: [1, 2, 3])
        hospital_dir: Directory containing hospital CSV files
        img_root: Root directory for images
        batch_size: Batch size for evaluation
    
    Returns:
        pd.DataFrame: Combined results from all hospitals
    """
    print(f"\n{'='*80}")
    print(f"RUNNING COMPARISON FOR ALL HOSPITALS: {hospital_ids}")
    print(f"ROUNDS: {rounds}")
    print(f"{'='*80}\n")
    
    all_hospital_results = []
    all_summary_results = []
    
    for hospital_id in hospital_ids:
        print(f"\n{'='*80}")
        print(f"PROCESSING HOSPITAL {hospital_id}")
        print(f"{'='*80}\n")
        
        try:
            # Run comparison across rounds for this hospital
            rounds_df = compare_across_rounds(hospital_id, rounds, hospital_dir=hospital_dir, 
                                             img_root=img_root, batch_size=batch_size)
            
            if rounds_df is not None and not rounds_df.empty:
                # Save individual hospital results immediately
                hospital_summary_file = f"sample_evaluation_hospital{hospital_id}_summary.csv"
                rounds_df.to_csv(hospital_summary_file, index=False)
                print(f"[INFO] Hospital {hospital_id} summary saved to: {hospital_summary_file}")
                
                # Collect for combined results
                all_hospital_results.append(rounds_df)
                all_summary_results.append({
                    'Hospital_ID': hospital_id,
                    'Rounds_Processed': len(rounds_df),
                    'Avg_Local_Accuracy': rounds_df['Local_Accuracy'].mean(),
                    'Avg_Global_Accuracy': rounds_df['Global_Accuracy'].mean(),
                    'Avg_Accuracy_Improvement': rounds_df['Accuracy_Improvement'].mean(),
                    'Total_Samples_Improved': rounds_df['Num_Samples_Improved'].sum(),
                    'Total_Samples_Regressed': rounds_df['Num_Samples_Regressed'].sum(),
                    'Best_Round_Improvement': rounds_df['Accuracy_Improvement'].max(),
                    'Best_Round': rounds_df.loc[rounds_df['Accuracy_Improvement'].idxmax(), 'Round_ID']
                })
                
                print(f"✅ Hospital {hospital_id} completed successfully!")
            else:
                print(f"⚠️ Warning: No results for hospital {hospital_id}")
                
        except Exception as e:
            print(f"❌ Error processing hospital {hospital_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create combined summary
    if all_hospital_results:
        print(f"\n{'='*80}")
        print("CREATING COMBINED SUMMARY")
        print(f"{'='*80}\n")
        
        # Combine all hospital results
        combined_df = pd.concat(all_hospital_results, ignore_index=True)
        combined_filename = "sample_evaluation_all_hospitals_combined.csv"
        try:
            combined_df.to_csv(combined_filename, index=False)
            print(f"[INFO] Combined results saved to: {combined_filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save combined CSV: {e}")
        
        # Create hospital summary
        if all_summary_results:
            summary_df = pd.DataFrame(all_summary_results)
            summary_filename = "sample_evaluation_all_hospitals_summary.csv"
            try:
                summary_df.to_csv(summary_filename, index=False)
                print(f"[INFO] Hospital summary saved to: {summary_filename}")
                
                # Print summary
                print(f"\n{'='*80}")
                print("ALL HOSPITALS SUMMARY")
                print(f"{'='*80}")
                print(summary_df.to_string(index=False))
                print(f"{'='*80}\n")
                
            except Exception as e:
                print(f"[ERROR] Failed to save summary CSV: {e}")
        
        # Create combined plots
        try:
            plot_all_hospitals_comparison(combined_df, hospital_ids)
        except Exception as e:
            print(f"[ERROR] Failed to create combined plots: {e}")
        
        return combined_df
    
    return None


def plot_all_hospitals_comparison(combined_df, hospital_ids):
    """
    Create plots comparing all hospitals.
    
    Args:
        combined_df: DataFrame with results from all hospitals
        hospital_ids: List of hospital IDs
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('All Hospitals: Local vs Global Model Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy comparison across hospitals and rounds
        ax1 = axes[0, 0]
        for hospital_id in hospital_ids:
            hospital_data = combined_df[combined_df['Hospital_ID'] == hospital_id]
            if not hospital_data.empty:
                ax1.plot(hospital_data['Round_ID'], hospital_data['Local_Accuracy'], 
                        marker='o', label=f'Hospital {hospital_id} Local', linewidth=2, markersize=6)
                ax1.plot(hospital_data['Round_ID'], hospital_data['Global_Accuracy'], 
                        marker='s', linestyle='--', label=f'Hospital {hospital_id} Global', 
                        linewidth=2, markersize=6)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Across Hospitals and Rounds')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Plot 2: Accuracy improvement by hospital
        ax2 = axes[0, 1]
        for hospital_id in hospital_ids:
            hospital_data = combined_df[combined_df['Hospital_ID'] == hospital_id]
            if not hospital_data.empty:
                ax2.plot(hospital_data['Round_ID'], hospital_data['Accuracy_Improvement'], 
                        marker='o', label=f'Hospital {hospital_id}', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy Improvement')
        ax2.set_title('Accuracy Improvement by Hospital')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Samples improved vs regressed (grouped bar)
        ax3 = axes[1, 0]
        x = np.arange(len(hospital_ids))
        width = 0.35
        improved = []
        regressed = []
        for hospital_id in hospital_ids:
            hospital_data = combined_df[combined_df['Hospital_ID'] == hospital_id]
            if not hospital_data.empty:
                improved.append(hospital_data['Num_Samples_Improved'].sum())
                regressed.append(hospital_data['Num_Samples_Regressed'].sum())
            else:
                improved.append(0)
                regressed.append(0)
        
        ax3.bar(x - width/2, improved, width, label='Improved', color='green', alpha=0.7)
        ax3.bar(x + width/2, regressed, width, label='Regressed', color='red', alpha=0.7)
        ax3.set_xlabel('Hospital')
        ax3.set_ylabel('Total Samples')
        ax3.set_title('Total Samples Improved vs Regressed')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Hospital {h}' for h in hospital_ids])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Average accuracy improvement by hospital
        ax4 = axes[1, 1]
        avg_improvements = []
        for hospital_id in hospital_ids:
            hospital_data = combined_df[combined_df['Hospital_ID'] == hospital_id]
            if not hospital_data.empty:
                avg_improvements.append(hospital_data['Accuracy_Improvement'].mean())
            else:
                avg_improvements.append(0)
        
        colors = ['green' if x > 0 else 'red' for x in avg_improvements]
        ax4.bar(range(len(hospital_ids)), avg_improvements, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_xlabel('Hospital')
        ax4.set_ylabel('Average Accuracy Improvement')
        ax4.set_title('Average Accuracy Improvement by Hospital')
        ax4.set_xticks(range(len(hospital_ids)))
        ax4.set_xticklabels([f'Hospital {h}' for h in hospital_ids])
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plot_filename = "sample_evaluation_all_hospitals_comparison.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"[INFO] Combined comparison plot saved to: {plot_filename}")
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Failed to create combined plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) >= 1:
        # Check if "all" is specified
        if len(sys.argv) >= 2 and sys.argv[1].lower() == 'all':
            # Run for all hospitals
            hospital_ids = [1, 2, 3, 4]
            if len(sys.argv) >= 3:
                rounds_arg = sys.argv[2]
                if ',' in rounds_arg:
                    rounds = [int(r.strip()) for r in rounds_arg.split(',')]
                else:
                    rounds = [int(rounds_arg)]
            else:
                rounds = [1, 2, 3]  # Default rounds
            
            result = run_all_hospitals(hospital_ids=hospital_ids, rounds=rounds)
            if result is not None:
                print(f"\n✅ All hospitals comparison completed successfully!")
        elif len(sys.argv) >= 2:
            hospital_id = int(sys.argv[1])
            
            if len(sys.argv) >= 3:
                # Single round or multiple rounds
                rounds_arg = sys.argv[2]
                if ',' in rounds_arg:
                    # Multiple rounds: "1,2,3"
                    rounds = [int(r.strip()) for r in rounds_arg.split(',')]
                    result = compare_across_rounds(hospital_id, rounds)
                else:
                    # Single round
                    round_id = int(rounds_arg)
                    result = compare_local_vs_global(hospital_id, round_id)
                    if result:
                        print(f"\n✅ Comparison completed successfully!")
            else:
                # Default: compare across rounds 1, 2, 3
                result = compare_across_rounds(hospital_id, rounds=[1, 2, 3])
                if result is not None:
                    print(f"\n✅ Cross-round comparison completed successfully!")
        else:
            print("Usage: python sample_evaluation.py <hospital_id|all> [round_id(s)]")
            print("Examples:")
            print("  python sample_evaluation.py 1 3          # Single round for hospital 1")
            print("  python sample_evaluation.py 1 1,2,3     # Multiple rounds for hospital 1")
            print("  python sample_evaluation.py 1            # All rounds 1,2,3 for hospital 1")
            print("  python sample_evaluation.py all         # All hospitals, rounds 1,2,3")
            print("  python sample_evaluation.py all 1,2,3   # All hospitals, specific rounds")

