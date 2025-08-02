`timescale 1ns / 1ps

module testbench;
parameter N1 = 8;
parameter N2 = 16;
parameter N3 = 32;
reg CLK, RST, EN;
reg [N2-1:0] input_data;
// 1000 time points
reg [N2-1:0] data[999:0];
wire [N3-1:0] output_data;
wire [N2-1:0] sampleT;

fir UUT(.input_data(input_data), .CLK(CLK), .RST(RST), .EN(EN), .output_data(output_data), .sampleT(sampleT));

integer FILE1;
integer k;

always #10 CLK=~CLK;

initial begin
    k = 0;
    $readmemb("input.data", data);
    FILE1 = $fopen("save.data", "w");
    
    CLK = 0;
    #20
    RST = 1'b1;
    #40
    RST = 0;
    EN = 1'b1;
    input_data <= data[0];
    #10
    
    for (k = 1'h1; k < 10'h3E8; k++) begin
        @(posedge CLK);
        $fdisplay(FILE1, "%b", output_data);
        input_data <= data[k];
        $display("k = %d", k); // Debug output to see progress
    end
    
    // Write final output
    @(posedge CLK);
    $fdisplay(FILE1, "%b", output_data);
    
    $fclose(FILE1);
    $display("Simulation completed successfully at k = %d", k);
    $finish; // Explicitly end simulation
end

endmodule