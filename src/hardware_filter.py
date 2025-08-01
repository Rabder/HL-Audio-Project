"""
Hardware FIR Filter Integration for Audio Classification
Simulates the SystemVerilog FIR filter behavior for preprocessing audio data
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import soundfile as sf

class HardwareFIRFilter:
    """
    Python implementation that mimics the SystemVerilog FIR filter behavior
    This can be used to simulate hardware processing or generate test vectors
    """
    
    def __init__(self, coefficients=None):
        # Default coefficients matching the SystemVerilog implementation
        # b[i] = 8'b00010000 = 16 in decimal (fixed-point representation)
        if coefficients is None:
            self.coefficients = np.array([16, 16, 16, 16, 16, 16, 16, 16], dtype=np.int8)
        else:
            self.coefficients = np.array(coefficients, dtype=np.int8)
        
        # Sample buffer (matches samples[0:7] in SystemVerilog)
        self.samples = np.zeros(8, dtype=np.int16)
        self.reset()
    
    def reset(self):
        """Reset filter state (RST signal)"""
        self.samples.fill(0)
        self.output_data_reg = 0
    
    def process_sample(self, input_data):
        """
        Process single sample (mimics one clock cycle with EN=1)
        
        Args:
            input_data: 16-bit signed input sample
            
        Returns:
            output_data: 32-bit signed filtered output
            sampleT: Current input sample (delay line tap)
        """
        # Convert input to 16-bit signed integer (matching N2 parameter)
        input_sample = np.clip(input_data, -32768, 32767).astype(np.int16)
        
        # FIR computation (matches the SystemVerilog multiply-accumulate)
        output_data_reg = (
            self.coefficients[0] * input_sample +
            self.coefficients[1] * self.samples[0] +
            self.coefficients[2] * self.samples[1] +
            self.coefficients[3] * self.samples[2] +
            self.coefficients[4] * self.samples[3] +
            self.coefficients[5] * self.samples[4] +
            self.coefficients[6] * self.samples[5] +
            self.coefficients[7] * self.samples[6]
        )
        
        # Update delay line (shift register behavior)
        self.samples[1:] = self.samples[:-1]  # Shift samples
        self.samples[0] = input_sample         # Insert new sample
        
        # Return outputs (matching SystemVerilog assignments)
        return np.int32(output_data_reg), self.samples[0]
    
    def filter_audio(self, audio_data, scale_factor=1.0):
        """
        Filter entire audio signal
        
        Args:
            audio_data: Input audio array (float, -1 to 1)
            scale_factor: Scaling for fixed-point conversion
            
        Returns:
            filtered_audio: Processed audio (scaled back to float)
        """
        # Convert float audio to 16-bit integer
        audio_int = (audio_data * 32767 * scale_factor).astype(np.int16)
        
        # Process each sample
        filtered_samples = []
        self.reset()
        
        for sample in audio_int:
            output_data, _ = self.process_sample(sample)
            filtered_samples.append(output_data)
        
        # Convert back to float and normalize
        filtered_audio = np.array(filtered_samples, dtype=np.float32) / (32767 * 16 * 8)  # Account for coefficient sum
        return filtered_audio

def design_audio_specific_filters():
    """
    Design FIR coefficients for audio classification tasks
    """
    filters = {
        # Low-pass filter for removing high-frequency noise
        'lowpass': np.array([1, 2, 3, 4, 4, 3, 2, 1]) * 2,
        
        # High-pass filter for emphasizing transients
        'highpass': np.array([-1, -2, -3, 4, 4, -3, -2, -1]) * 2,
        
        # Bandpass for voice frequency range (important for hgrunt/scientist)
        'voice_bandpass': np.array([1, 0, -2, 4, 4, -2, 0, 1]) * 2,
        
        # Moving average (smoothing filter)
        'smooth': np.array([2, 2, 2, 2, 2, 2, 2, 2]),
        
        # Edge detection (good for weapon sounds)
        'edge_detect': np.array([-1, -1, -1, 8, 8, -1, -1, -1]),
    }
    return filters

def generate_filter_test_vectors(audio_file_path, filter_type='lowpass'):
    """
    Generate test vectors for SystemVerilog simulation
    """
    # Load audio file
    audio, sr = librosa.load(audio_file_path, sr=22050)
    
    # Create filter
    filters = design_audio_specific_filters()
    fir_filter = HardwareFIRFilter(coefficients=filters[filter_type])
    
    # Process audio
    filtered_audio = fir_filter.filter_audio(audio[:1000])  # Process first 1000 samples
    
    # Generate test vectors
    test_vectors = []
    fir_filter.reset()
    
    for i, sample in enumerate((audio[:100] * 32767).astype(np.int16)):
        output, sampleT = fir_filter.process_sample(sample)
        test_vectors.append({
            'clock_cycle': i,
            'input_data': int(sample),
            'expected_output': int(output),
            'sampleT': int(sampleT)
        })
    
    return test_vectors, filtered_audio

def visualize_filter_response(audio_file_path):
    """
    Visualize different filter responses on audio data
    """
    # Load sample audio
    audio, sr = librosa.load(audio_file_path, sr=22050)
    audio_segment = audio[:2048]  # Use first 2048 samples
    
    filters = design_audio_specific_filters()
    
    plt.figure(figsize=(15, 10))
    
    # Plot original audio
    plt.subplot(3, 2, 1)
    plt.plot(audio_segment)
    plt.title('Original Audio')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Plot filtered versions
    for i, (filter_name, coeffs) in enumerate(filters.items(), 2):
        fir_filter = HardwareFIRFilter(coefficients=coeffs)
        filtered = fir_filter.filter_audio(audio_segment)
        
        plt.subplot(3, 2, i)
        plt.plot(filtered[:len(audio_segment)])
        plt.title(f'{filter_name.title()} Filtered')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        if i >= 6:  # Only show first 5 filters
            break
    
    plt.tight_layout()
    plt.savefig('/Users/rashid.urbina/Documents/audio_classifier/filter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    sample_audio = "/Users/rashid.urbina/Documents/audio_classifier/audio_set/weapons/357_shot1.wav"
    
    # Generate test vectors for SystemVerilog simulation
    test_vectors, filtered_audio = generate_filter_test_vectors(sample_audio, 'voice_bandpass')
    
    print("Sample test vectors for SystemVerilog:")
    for tv in test_vectors[:10]:
        print(f"Cycle {tv['clock_cycle']:2d}: input={tv['input_data']:6d}, expected_output={tv['expected_output']:8d}")
    
    # Visualize filter responses
    visualize_filter_response(sample_audio)