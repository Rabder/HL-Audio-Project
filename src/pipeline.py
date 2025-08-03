import numpy as np
import scipy.io.wavfile as wav
import os
import subprocess
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

def generate_filter_coefficients(filter_type, sample_rate=11025, num_taps=20):
    """Generate different filter coefficients for audio processing"""
    nyquist = sample_rate / 2
    
    if filter_type == 'lowpass':
        cutoff = 2000 / nyquist  # 2kHz cutoff - removes high freq noise
        coeffs = signal.firwin(num_taps, cutoff, window='hamming')
    
    # set num_taps + 1 for highpass to ensure odd number of taps
    elif filter_type == 'highpass':
        cutoff = 1000 / nyquist  # 1kHz cutoff - removes low freq rumble
        coeffs = signal.firwin(num_taps + 1, cutoff, window='hamming', pass_zero=False)
    
    # set num_taps + 1 for bandpass to ensure odd number of taps
    elif filter_type == 'bandpass':
        low = 800 / nyquist   # For speech/vocals (800-3000 Hz)
        high = 3000 / nyquist
        coeffs = signal.firwin(num_taps + 1, [low, high], window='hamming', pass_zero=False)
    
    elif filter_type == 'bandstop':  # Notch filter
        low = 1500 / nyquist   # Remove specific frequency band
        high = 2500 / nyquist
        coeffs = signal.firwin(num_taps + 1, [low, high], window='hamming', pass_zero=True)
    
    elif filter_type == 'allpass':  # No filtering, just delay
        coeffs = np.zeros(num_taps)
        coeffs[0] = 1.0  # Unit impulse
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Normalize and scale to 8-bit signed integers
    # Ensure coefficients sum to reasonable value to prevent overflow
    coeffs_sum = np.sum(np.abs(coeffs))
    if coeffs_sum > 0:
        coeffs = coeffs / coeffs_sum * 2.0  # Scale to prevent massive amplification
    
    # Convert to 8-bit signed integers (range: -128 to 127)
    coeffs_scaled = np.round(coeffs * 64).astype(np.int8)  # Scale by 64 instead of 127
    
    # Ensure we don't have all zeros
    if np.all(coeffs_scaled == 0):
        coeffs_scaled[0] = 1
    
    return coeffs_scaled

class SystemVerilogFIRInterface:
    """
    Enhanced interface with dynamic coefficient loading
    """
    
    def __init__(self, fir_module_path=None, simulator="iverilog"):
        # Set default path if none provided
        if fir_module_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fir_module_path = os.path.join(script_dir, "fir_project.srcs", "sources_1", "new", "fir.sv")
        
        self.fir_module_path = os.path.abspath(fir_module_path)
        self.simulator = simulator
        self.current_coefficients = None
        self.filter_type = "lowpass"  # Default
        
        if not os.path.exists(self.fir_module_path):
            raise FileNotFoundError(f"FIR module not found at: {self.fir_module_path}")

    def set_filter_coefficients(self, filter_type, sample_rate=11025):
        """Set filter coefficients for the specified filter type"""
        self.filter_type = filter_type
        self.current_coefficients = generate_filter_coefficients(filter_type, sample_rate)
        print(f"Set {filter_type} filter coefficients: {self.current_coefficients}")

    def generate_testbench(self, audio_data, output_dir="processed/sim_temp"):
        """Generate SystemVerilog testbench with dynamic coefficients"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Use current coefficients or default lowpass
        if self.current_coefficients is None:
            self.set_filter_coefficients('lowpass')
        
        # Generate coefficient assignments
        coeff_assignments = ""
        for i in range(8):
            if i < len(self.current_coefficients):
                coeff_val = int(self.current_coefficients[i])
                # Handle negative coefficients properly
                if coeff_val < 0:
                    coeff_assignments += f"        assign b[{i}] = 8'sd{abs(coeff_val)} * (-1);\n"
                else:
                    coeff_assignments += f"        assign b[{i}] = 8'sd{coeff_val};\n"
            else:
                coeff_assignments += f"        assign b[{i}] = 8'sd0;\n"
        
        testbench_content = f'''`timescale 1ns / 1ps

module fir_tb();

    // Parameters matching your FIR module
    parameter N1 = 8;
    parameter N2 = 16;
    parameter N3 = 32;
    parameter AUDIO_LENGTH = {len(audio_data)};
    
    // Testbench signals
    reg signed [N2-1:0] input_data;
    reg CLK, RST, EN;
    wire signed [N3-1:0] output_data;
    wire signed [N2-1:0] sampleT;
    
    // Audio data storage
    reg signed [N2-1:0] audio_samples [0:AUDIO_LENGTH-1];
    reg signed [N3-1:0] filtered_results [0:AUDIO_LENGTH-1];
    
    // Test control
    integer i;
    integer output_file;
    
    // Dynamic coefficient assignment for {self.filter_type} filter
    wire signed [N1-1:0] b[0:7];
{coeff_assignments}
    
    // Modified FIR filter with external coefficients
    reg signed [N3-1:0] output_data_reg;
    reg signed [N2-1:0] samples[0:7];

    // FIR filter logic (embedded instead of module instantiation)
    always @(posedge CLK) begin
        if (RST) begin
            for (int j = 0; j < 8; j++) 
                samples[j] <= 0;
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
    
    // Clock generation
    always begin
        #1 CLK = ~CLK;  // 100MHz clock for simulation
    end
    
    // Load audio data
    initial begin
        $display("Loading audio samples for {self.filter_type} filter...");
        $readmemh("audio_input.hex", audio_samples);
        if (audio_samples[0] === 'bx) begin
            $display("ERROR: Failed to load audio samples");
            $finish;
        end
        $display("Audio samples loaded successfully");
        $display("Filter coefficients: %d %d %d %d %d %d %d %d", 
                 b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
    end
    
    // Test sequence
    initial begin
        // Initialize
        CLK = 0;
        RST = 1;
        EN = 0;
        input_data = 0;
        
        // Open output file
        output_file = $fopen("filtered_output.txt", "w");
        if (output_file == 0) begin
            $display("ERROR: Could not open output file");
            $finish;
        end
        
        // Reset sequence
        #4 RST = 0;
        #2 EN = 1;
        
        // Process audio samples
        for (i = 0; i < AUDIO_LENGTH; i = i + 1) begin
            @(posedge CLK);
            input_data = audio_samples[i];
            
            @(posedge CLK);  // Wait for output to settle
            filtered_results[i] = output_data;
            
            // Write to file
            $fwrite(output_file, "%d\\n", $signed(output_data));
            
            // Print progress
            if (i % 1000 == 0) begin
                $display("Processed sample %d/%d", i, AUDIO_LENGTH);
            end
        end
        
        $fclose(output_file);
        $display("{self.filter_type.upper()} filtering complete! Results saved to filtered_output.txt");
        $finish;
    end
    
    // VCD dump for waveform viewing
    initial begin
        $dumpfile("fir_filter.vcd");
        $dumpvars(0, fir_tb);
    end

endmodule'''

        # Write testbench file
        tb_path = os.path.join(output_dir, "fir_tb.sv")
        with open(tb_path, 'w') as f:
            f.write(testbench_content)
            
        return tb_path

    def write_audio_hex(self, audio_data, output_dir="processed/sim_temp"):
        """Convert audio data to hex format for SystemVerilog $readmemh"""
        hex_path = os.path.join(output_dir, "audio_input.hex")
        
        with open(hex_path, 'w') as f:
            for sample in audio_data:
                # Convert to 16-bit 2's complement
                sample = int(sample) & 0xFFFF
                # Write as 4-digit hex without any extra characters
                f.write(f"{sample:04X}\n")
        
        print(f"Audio data written to {hex_path} ({len(audio_data)} samples)")
        return hex_path

    def _run_iverilog(self, output_dir):
        """Run simulation using Icarus Verilog"""
        try:
            # Get absolute paths
            output_dir = os.path.abspath(output_dir)
            tb_path = os.path.join(output_dir, "fir_tb.sv")
            
            # Compile command - only need testbench since FIR logic is embedded
            compile_cmd = [
                "iverilog",
                "-g2012", # use SystemVerilog 2012 standard to avoid legacy verilog bullshit
                "-o", os.path.join(output_dir, "fir_sim"),
                tb_path
            ]
            
            print(f"Compiling {self.filter_type} filter with iverilog...")
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Compilation failed:")
                print(result.stderr)
                return False
            
            # Run simulation
            run_cmd = ["vvp", os.path.join(output_dir, "fir_sim")]
            print("Running simulation...")
            result = subprocess.run(run_cmd, capture_output=True, text=True, cwd=output_dir)
            
            if result.returncode != 0:
                print("Simulation failed:")
                print(result.stderr)
                return False
            
            print("Simulation output:")
            print(result.stdout)
            return True
            
        except FileNotFoundError:
            print("Error: iverilog not found. Please install Icarus Verilog.")
            return False

    def run_simulation(self, output_dir="processed/sim_temp"):
        """Run the SystemVerilog simulation"""
        if self.simulator == "iverilog":
            return self._run_iverilog(output_dir)
        else:
            raise ValueError(f"Unsupported simulator: {self.simulator}")

    def read_simulation_results(self, output_dir="processed/sim_temp"):
        """Read the filtered audio results from simulation output"""
        results_path = os.path.join(output_dir, "filtered_output.txt")
        
        try:
            with open(results_path, 'r') as f:
                results = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        value = int(line)
                        # Handle 32-bit signed conversion
                        if value > 0x7FFFFFFF:
                            value -= 0x100000000
                        results.append(value)
                    except ValueError:
                        print(f"Warning: Skipping invalid line: {line}")
                        continue
                
            print(f"Read {len(results)} filtered samples from simulation")
            return np.array(results, dtype=np.int32)
            
        except FileNotFoundError:
            print(f"Results file not found: {results_path}")
            return None

    def process_audio(self, audio_data, output_dir="processed/sim_temp"):
        """Complete pipeline with current filter settings"""
        print(f"Processing {len(audio_data)} audio samples through {self.filter_type} FIR filter...")
        
        # Ensure output directory exists
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate testbench with current coefficients
        print("1. Generating testbench with filter coefficients...")
        self.generate_testbench(audio_data, output_dir)
        
        # Write audio data
        print("2. Writing audio data to hex file...")
        self.write_audio_hex(audio_data, output_dir)
        
        # Run simulation
        print("3. Running SystemVerilog simulation...")
        success = self.run_simulation(output_dir)
        
        if not success:
            print("Simulation failed!")
            return None
        
        # Read results
        print("4. Reading simulation results...")
        filtered_audio = self.read_simulation_results(output_dir)
        
        return filtered_audio

def load_wav_file(filepath):
    """Load WAV file and convert to appropriate format"""
    try:
        sample_rate, audio_data = wav.read(filepath)
        
        print(f"Loaded: {filepath}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Shape: {audio_data.shape}")
        print(f"  Data type: {audio_data.dtype}")
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            print(f"  Converted to mono")
        
        # Convert to 16-bit signed integers
        if audio_data.dtype != np.int16:
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
            print(f"  Converted to int16")
        
        print(f"  Final range: [{np.min(audio_data)}, {np.max(audio_data)}]")
        return sample_rate, audio_data
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def process_audio_folder_with_multiple_filters(folder_path, output_base="processed"):
    """Process audio folder with multiple filter types"""
    folder_path = Path(folder_path)
    category_name = folder_path.name
    
    # Different filter types to apply
    filter_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']
    
    wav_files = list(folder_path.glob("*.wav"))[:]  # Limit to first 10 files for testing
    
    for filter_type in filter_types:
        print(f"\n{'='*80}")
        print(f"Processing {category_name} with {filter_type.upper()} filter")
        print(f"{'='*80}")
        
        # Create filter-specific output directory
        filter_output_dir = Path(output_base) / f"{category_name}_{filter_type}"
        filter_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize interface with this filter type
        fir_interface = SystemVerilogFIRInterface(simulator="iverilog")
        
        results = []
        
        for i, wav_file in enumerate(wav_files):
            print(f"\nProcessing file {i+1}/{len(wav_files)}: {wav_file.name}")
            
            sample_rate, audio_data = load_wav_file(wav_file)
            
            if audio_data is not None:
                # Set filter coefficients based on sample rate
                fir_interface.set_filter_coefficients(filter_type, sample_rate)
                
                # Create simulation directory
                sim_dir = filter_output_dir / "sim" / f"sim_{wav_file.stem}"
                sim_dir.mkdir(parents=True, exist_ok=True)
                
                # Process audio
                filtered_audio = fir_interface.process_audio(audio_data, str(sim_dir))
                
                if filtered_audio is not None:
                    print(f"  ✓ {filter_type} filter applied successfully")
                    print(f"  Filtered range: [{np.min(filtered_audio)}, {np.max(filtered_audio)}]")
                    
                    # Determine appropriate scaling based on filter type and output range
                    max_output = np.max(np.abs(filtered_audio))
                    if max_output > 32767:
                        scale_factor = max_output // 32767 + 1
                    else:
                        scale_factor = 1
                    
                    scaled_filtered = (filtered_audio // scale_factor).astype(np.int16)
                    
                    # Save filtered audio
                    output_file = filter_output_dir / f"{filter_type}_{wav_file.name}"
                    wav.write(output_file, sample_rate, scaled_filtered)
                    print(f"  Saved to: {output_file}")
                    
                    # Store result
                    results.append({
                        'filename': wav_file.name,
                        'filter_type': filter_type,
                        'category': category_name,
                        'sample_rate': sample_rate,
                        'scale_factor': scale_factor,
                        'output_range': [np.min(filtered_audio), np.max(filtered_audio)]
                    })
                else:
                    print(f"  ✗ Failed to apply {filter_type} filter to {wav_file.name}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Test single filter type
    print("Testing single filter type...")
    fir_interface = SystemVerilogFIRInterface(simulator="iverilog")
    fir_interface.set_filter_coefficients('lowpass', sample_rate=11025)
    
    # Test multiple filter types on dataset
    print("\nTesting multiple filter types on dataset...")
    categories = ["ambience"]  # Start with one category for testing
    
    base_path = "audio_set"
    output_base = "processed_multifilter"
    
    for category in categories:
        folder_path = f"{base_path}/{category}"
        if os.path.exists(folder_path):
            results = process_audio_folder_with_multiple_filters(
                folder_path, 
                output_base=output_base
            )
            print(f"\nProcessed {len(results)} files with multiple filters")
        else:
            print(f"Folder not found: {folder_path}")