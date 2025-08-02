`timescale 1ns / 1ps

module fir(input_data, CLK, RST, EN, output_data, sampleT);
parameter N1 = 16;
parameter N2 = 32;
parameter N3 = 64;

wire signed [N1-1:0] b[0:7];
genvar i;
generate 
    for (i = 0; i < 8; i++) begin: create
        assign b[i] = 8'b00010000;
    end 
endgenerate 

input signed [N2-1:0] input_data;
output signed [N2-1:0] sampleT;
input CLK, RST, EN;
output signed [N3-1:0] output_data;

reg signed [N3-1:0] output_data_reg;
reg signed [N2-1:0] samples[0:7];

// Fixed always block structure
always @(posedge CLK) begin
    if (RST) begin
        for (int i = 0; i < 7; i++) 
            samples[i] <= 0;
        output_data_reg <= 0;
    end
    else if (EN) begin  // Changed from "if (!RST && EN)" to "else if (EN)"
        output_data_reg <= b[0] * input_data +
                          b[1] * samples[0] +
                          b[2] * samples[1] +
                          b[3] * samples[2] +
                          b[4] * samples[3] +
                          b[5] * samples[4] +
                          b[6] * samples[5] +
                          b[7] * samples[6];
        
        samples[0] <= input_data;
        for (int k = 1; k < 7; k++) 
            samples[k] <= samples[k-1];
    end
end

assign output_data = output_data_reg;
assign sampleT = samples[0]; 

endmodule