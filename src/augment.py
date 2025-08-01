import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import random
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift, 
    Shift, PolarityInversion, BandStopFilter,
    LowPassFilter, HighPassFilter, Reverse
)

class AudioAugmenter:
    def __init__(self, base_path, target_samples=750):
        self.base_path = Path(base_path)
        self.target_samples = target_samples
        
        # Define augmentation pipeline
        self.augmentations = [
            # Pitch variations
            PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
            PitchShift(min_semitones=-2, max_semitones=2, p=1.0),
            
            # Time stretching
            TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0),
            
            # Noise addition
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.025, p=1.0),
            
            # Time shifting
            Shift(min_shift=-0.5, max_shift=0.5, p=1.0),
            
            # Filtering
            LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=8000, p=1.0),
            HighPassFilter(min_cutoff_freq=200, max_cutoff_freq=1000, p=1.0),
            BandStopFilter(min_center_freq=1000, max_center_freq=4000, p=1.0),
            
            # Other effects
            PolarityInversion(p=1.0),
            Reverse(p=1.0),
            
            # Combined effects
            Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.8),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.7),
            ]),
            Compose([
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.8),
                LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=8000, p=0.6),
            ]),
        ]
        
    def get_category_stats(self):
        """Get current sample counts for each category"""
        stats = {}
        for category_dir in self.base_path.iterdir():
            if category_dir.is_dir():
                wav_files = list(category_dir.glob("*.wav"))
                stats[category_dir.name] = len(wav_files)
        return stats
    
    def load_audio(self, file_path):
        """Load audio file with librosa"""
        try:
            audio, sr = librosa.load(file_path, sr=None)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def save_audio(self, audio, sr, output_path):
        """Save audio file"""
        try:
            sf.write(output_path, audio, sr)
            return True
        except Exception as e:
            print(f"Error saving {output_path}: {e}")
            return False
    
    def augment_file(self, input_path, output_path, augmentation):
        """Apply augmentation to a single file"""
        audio, sr = self.load_audio(input_path)
        if audio is None:
            return False
        
        try:
            # Convert to format expected by audiomentations (float32, normalized)
            audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Apply augmentation
            augmented_audio = augmentation(samples=audio, sample_rate=sr)
            
            # Save augmented audio
            return self.save_audio(augmented_audio, sr, output_path)
        except Exception as e:
            print(f"Error augmenting {input_path}: {e}")
            return False
    
    def augment_category(self, category_name):
        """Augment all files in a category to reach target sample count"""
        category_path = self.base_path / category_name
        if not category_path.exists():
            print(f"Category {category_name} not found")
            return
        
        # Get original files
        original_files = list(category_path.glob("*.wav"))
        current_count = len(original_files)
        needed_samples = self.target_samples - current_count
        
        if needed_samples <= 0:
            print(f"Category {category_name} already has {current_count} samples (target: {self.target_samples})")
            return
        
        print(f"Augmenting {category_name}: {current_count} -> {self.target_samples} samples (+{needed_samples})")
        
        # Create augmented samples
        augmented_count = 0
        max_attempts = needed_samples * 3  # Prevent infinite loops
        attempts = 0
        
        while augmented_count < needed_samples and attempts < max_attempts:
            # Randomly select source file and augmentation
            source_file = random.choice(original_files)
            augmentation = random.choice(self.augmentations)
            
            # Generate unique output filename
            base_name = source_file.stem
            aug_name = f"{base_name}_aug_{augmented_count:04d}.wav"
            output_path = category_path / aug_name
            
            # Apply augmentation
            if self.augment_file(source_file, output_path, augmentation):
                augmented_count += 1
                if augmented_count % 50 == 0:
                    print(f"  Generated {augmented_count}/{needed_samples} samples...")
            
            attempts += 1
        
        print(f"Completed {category_name}: generated {augmented_count} new samples")
    
    def augment_all_categories(self):
        """Augment all categories to reach target sample count"""
        stats = self.get_category_stats()
        print("Current dataset statistics:")
        for category, count in stats.items():
            needed = max(0, self.target_samples - count)
            print(f"  {category}: {count} samples (need +{needed})")
        
        print(f"\nStarting augmentation to {self.target_samples} samples per category...")
        
        for category in stats.keys():
            self.augment_category(category)
        
        # Final stats
        print("\nFinal dataset statistics:")
        final_stats = self.get_category_stats()
        for category, count in final_stats.items():
            print(f"  {category}: {count} samples")

def main():
    # Configuration
    audio_dataset_path = "/Users/rashid.urbina/Documents/audio_classifier/audio_set"
    target_samples = 750
    
    # Initialize augmenter
    augmenter = AudioAugmenter(audio_dataset_path, target_samples)
    
    # Run augmentation
    augmenter.augment_all_categories()

if __name__ == "__main__":
    main()