`timescale 1ns / 1ps

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
    assign b[0] = 8'sd2;
    assign b[1] = 8'sd-1;
    assign b[2] = 8'sd-4;
    assign b[3] = 8'sd8;
    assign b[4] = 8'sd8;
    assign b[5] = 8'sd-4;
    assign b[6] = 8'sd-1;
    assign b[7] = 8'sd2;
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

endmodule