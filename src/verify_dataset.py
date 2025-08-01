#!/usr/bin/env python3

import os
import librosa
import numpy as np
from pathlib import Path
from collections import defaultdict
import soundfile as sf

def analyze_dataset(base_path):
    """Analyze the balanced dataset for quality and statistics"""
    base_path = Path(base_path)
    
    stats = {
        'categories': {},
        'total_samples': 0,
        'audio_stats': defaultdict(list)
    }
    
    print("=== DATASET ANALYSIS ===\n")
    
    # Analyze each category
    for category_dir in base_path.iterdir():
        if not category_dir.is_dir():
            continue
            
        wav_files = list(category_dir.glob("*.wav"))
        category_count = len(wav_files)
        stats['categories'][category_dir.name] = category_count
        stats['total_samples'] += category_count
        
        print(f"{category_dir.name}: {category_count} samples")
        
        # Sample a few files for quality analysis
        sample_files = wav_files[:5] if len(wav_files) >= 5 else wav_files
        
        for sample_file in sample_files:
            try:
                audio, sr = librosa.load(sample_file, sr=None)
                duration = len(audio) / sr
                
                stats['audio_stats']['duration'].append(duration)
                stats['audio_stats']['sample_rate'].append(sr)
                stats['audio_stats']['length'].append(len(audio))
                
                # Basic audio quality checks
                if np.max(np.abs(audio)) > 0:
                    rms = np.sqrt(np.mean(audio**2))
                    stats['audio_stats']['rms'].append(rms)
                    
            except Exception as e:
                print(f"  Error analyzing {sample_file}: {e}")
    
    # Print summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Categories: {len(stats['categories'])}")
    print(f"Balance: {'✓ Balanced' if len(set(stats['categories'].values())) == 1 else '✗ Unbalanced'}")
    
    if stats['audio_stats']['duration']:
        print(f"\nAudio Quality Metrics (from samples):")
        print(f"Duration range: {min(stats['audio_stats']['duration']):.3f} - {max(stats['audio_stats']['duration']):.3f} seconds")
        print(f"Sample rates: {set(stats['audio_stats']['sample_rate'])}")
        print(f"Average RMS: {np.mean(stats['audio_stats']['rms']):.4f}")
        
    # Check for augmented files
    print(f"\n=== AUGMENTATION VERIFICATION ===")
    for category, count in stats['categories'].items():
        category_path = base_path / category
        original_files = [f for f in category_path.glob("*.wav") if "_aug_" not in f.name]
        augmented_files = [f for f in category_path.glob("*.wav") if "_aug_" in f.name]
        
        print(f"{category}:")
        print(f"  Original: {len(original_files)} samples")
        print(f"  Augmented: {len(augmented_files)} samples")
        print(f"  Total: {len(original_files) + len(augmented_files)} samples")
    
    return stats

if __name__ == "__main__":
    dataset_path = "/Users/rashid.urbina/Documents/audio_classifier/audio_set"
    stats = analyze_dataset(dataset_path)