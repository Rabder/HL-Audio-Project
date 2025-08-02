"""
FIR Filter Coefficient Optimization for Half-Life Audio Classification
Finds optimal 8-tap FIR coefficients for distinguishing between audio classes
"""

import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from hardware_filter import HardwareFIRFilter
import itertools

class FIRCoefficientOptimizer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.categories = ['ambience', 'hgrunt', 'scientist', 'weapons', 'zombie']
        self.audio_data = []
        self.labels = []
        
    def load_sample_data(self, samples_per_class=50):
        """Load sample audio data from each category"""
        print("Loading audio samples...")
        
        for label, category in enumerate(self.categories):
            category_path = self.dataset_path / category
            wav_files = list(category_path.glob("*.wav"))[:samples_per_class]
            
            for wav_file in wav_files:
                try:
                    audio, sr = librosa.load(wav_file, sr=22050, duration=2.0)
                    if len(audio) > 1024:  # Ensure minimum length
                        self.audio_data.append(audio[:1024])  # Use first 1024 samples
                        self.labels.append(label)
                except Exception as e:
                    print(f"Error loading {wav_file}: {e}")
        
        print(f"Loaded {len(self.audio_data)} audio samples")
    
    def extract_filter_features(self, audio_samples, coefficients):
        """Extract features using FIR filter with given coefficients"""
        fir_filter = HardwareFIRFilter(coefficients=coefficients)
        
        features = []
        for audio in audio_samples:
            # Apply hardware FIR filter
            filtered = fir_filter.filter_audio(audio, scale_factor=0.1)
            
            # Extract meaningful features from filtered signal
            feature_vector = [
                np.mean(np.abs(filtered)),           # Average magnitude
                np.std(filtered),                    # Standard deviation
                np.max(np.abs(filtered)),            # Peak amplitude
                np.sum(filtered**2),                 # Energy
                np.mean(np.diff(filtered)**2),       # Smoothness measure
                len(np.where(np.diff(np.sign(filtered)))[0]),  # Zero crossings
            ]
            features.append(feature_vector)
            
            fir_filter.reset()  # Reset for next audio sample
        
        return np.array(features)
    
    def evaluate_coefficients(self, coefficients):
        """Evaluate classification performance with given FIR coefficients"""
        # Extract features using the FIR filter
        features = self.extract_filter_features(self.audio_data, coefficients)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, self.labels, test_size=0.3, random_state=42, stratify=self.labels
        )
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def optimize_coefficients_grid_search(self):
        """Grid search optimization for FIR coefficients"""
        print("Optimizing FIR coefficients...")
        
        # Define search space (8-bit signed integers, limited range for efficiency)
        coeff_values = [-8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32]
        
        best_accuracy = 0
        best_coefficients = None
        results = []
        
        # Try some predefined coefficient patterns
        coefficient_patterns = [
            # Original uniform coefficients
            [16, 16, 16, 16, 16, 16, 16, 16],
            
            # Low-pass patterns
            [1, 2, 4, 8, 8, 4, 2, 1],
            [2, 4, 6, 8, 8, 6, 4, 2],
            
            # High-pass patterns  
            [-1, -2, -4, 16, 16, -4, -2, -1],
            [1, -2, 4, -8, -8, 4, -2, 1],
            
            # Bandpass patterns
            [1, 0, -2, 4, 4, -2, 0, 1],
            [2, -1, -4, 8, 8, -4, -1, 2],
            
            # Edge detection patterns
            [-1, -1, -1, 8, 8, -1, -1, -1],
            [1, 1, -2, -4, -4, -2, 1, 1],
            
            # Voice-optimized patterns (emphasis on mid frequencies)
            [1, 3, 5, 8, 8, 5, 3, 1],
            [2, 2, 4, 8, 8, 4, 2, 2],
        ]
        
        for i, coeffs in enumerate(coefficient_patterns):
            try:
                accuracy = self.evaluate_coefficients(coeffs)
                results.append({'coefficients': coeffs, 'accuracy': accuracy})
                
                print(f"Pattern {i+1:2d}: {coeffs} -> Accuracy: {accuracy:.3f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_coefficients = coeffs.copy()
                    
            except Exception as e:
                print(f"Error evaluating coefficients {coeffs}: {e}")
        
        return best_coefficients, best_accuracy, results
    
    def generate_optimized_systemverilog(self, coefficients):
        """Generate optimized SystemVerilog code with best coefficients"""
        sv_code = f'''`timescale 1ns / 1ps

// Optimized FIR Filter for Half-Life Audio Classification
// Coefficients optimized for distinguishing between:
// ambience, hgrunt, scientist, weapons, zombie

module fir_optimized(input_data, CLK, RST, EN, output_data, sampleT);
parameter N1 = 8;
parameter N2 = 16; 
parameter N3 = 32;

// Optimized coefficients
wire signed [N1-1:0] b[0:7];
genvar i;
generate 
    assign b[0] = 8'sd{coefficients[0]};
    assign b[1] = 8'sd{coefficients[1]};
    assign b[2] = 8'sd{coefficients[2]};
    assign b[3] = 8'sd{coefficients[3]};
    assign b[4] = 8'sd{coefficients[4]};
    assign b[5] = 8'sd{coefficients[5]};
    assign b[6] = 8'sd{coefficients[6]};
    assign b[7] = 8'sd{coefficients[7]};
endgenerate 

input signed [N2-1:0] input_data;
output signed [N2-1:0] sampleT;
input CLK, RST, EN;
output signed [N3-1:0] output_data;

reg signed [N3-1:0] output_data_reg;
reg signed [N2-1:0] samples[0:7];

// Filter implementation
always @(posedge CLK) begin
    if (RST) begin
        for (int i = 0; i < 8; i++) 
            samples[i] <= 0;
        output_data_reg <= 0;
    end
    else if (EN) begin
        output_data_reg <= b[0] * input_data +
                          b[1] * samples[0] +
                          b[2] * samples[1] +
                          b[3] * samples[2] +
                          b[4] * samples[3] +
                          b[5] * samples[4] +
                          b[6] * samples[5] +
                          b[7] * samples[6];
        
        samples[0] <= input_data;
        for (int k = 1; k < 8; k++) 
            samples[k] <= samples[k-1];
    end
end

assign output_data = output_data_reg;
assign sampleT = samples[0];

endmodule'''
        
        return sv_code

def main():
    # Initialize optimizer
    optimizer = FIRCoefficientOptimizer("/Users/rashid.urbina/Documents/audio_classifier/audio_set")
    
    # Load sample data (reduced for efficiency)
    optimizer.load_sample_data(samples_per_class=30)
    
    # Optimize coefficients
    best_coeffs, best_accuracy, all_results = optimizer.optimize_coefficients_grid_search()
    
    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"Best coefficients: {best_coeffs}")
    print(f"Best accuracy: {best_accuracy:.3f}")
    
    # Generate optimized SystemVerilog
    if best_coeffs:
        sv_code = optimizer.generate_optimized_systemverilog(best_coeffs)
        
        # Save to file
        output_file = "/Users/rashid.urbina/Documents/audio_classifier/fir_optimized.sv"
        with open(output_file, 'w') as f:
            f.write(sv_code)
        
        print(f"\nOptimized SystemVerilog saved to: {output_file}")
        
        # Save results summary
        with open("/Users/rashid.urbina/Documents/audio_classifier/fir_optimization_results.txt", 'w') as f:
            f.write("FIR Coefficient Optimization Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Best Coefficients: {best_coeffs}\n")
            f.write(f"Best Accuracy: {best_accuracy:.3f}\n\n")
            f.write("All Results:\n")
            for i, result in enumerate(all_results):
                f.write(f"{i+1:2d}. {result['coefficients']} -> {result['accuracy']:.3f}\n")

if __name__ == "__main__":
    main()