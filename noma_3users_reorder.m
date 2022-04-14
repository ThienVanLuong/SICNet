%% Downlink NOMA with 3 users
% Modulation BPSK, real signal

%%
% clc;
clear all;

M = 2; % Modulation
num_sample = 5e6;
snr_db = 0:3:15;
global s_nLevels;
global s_fDynRange;

s_nLevels = 8;
s_fDynRange = 8;

h = [1 1 1];
iter = 1;

ber0 = zeros(1,length(snr_db));
for c=1:iter
    ber3 = zeros(1,length(snr_db));
    for k=1:length(snr_db)
        snr = 10.^(snr_db(k)/10);

        bit1 = randi([0 1],num_sample,1);
        bit2 = randi([0 1],num_sample,1);
        bit3 = randi([0 1],num_sample,1);
        x1 = 2*bit1 - 1;
        x2 = 2*bit2 - 1;
        x3 = 2*bit3 - 1;
        
    %     x = x1 + x2 + x3;
        p2 = sqrt(4);
        p1 = sqrt(1/9);
        x = x3 + p2*x2 + p1*x1;

        noise_std3 = (1/snr);
        noise3 = sqrt(noise_std3)*randn(num_sample,1);
        y3 = h(3)*x + noise3;
        
        h3_est = h(3)*ones(num_sample,1) + randn(num_sample,1)/sqrt(100); 
        y3_est = h3_est.*x + noise3;

        %quantized Gaussian signal
        y3_gau_quant = m_fQuant(y3);
        y3_est_gau_quant = m_fQuant(y3_est);  

%% detection SIC
        x3_hat = zeros(num_sample,1);
        for n=1:num_sample
            r3 = y3(n);
            h3 = h3_est(n);
%             h3 = h(3);

            re_x2 = sign(r3/h3);
            r3 = r3 - p2*h3*re_x2;
    %         r3 = r3 - p2*h3*x2(n);

            re_x3 = sign(r3/h3);
            x3_hat(n) = re_x3;
        end
        err_3 = sum(x3_hat~=x3);
        ber3(k) = err_3/num_sample;  
    end
% ber3
ber0 = ber0 + ber3/iter;
% ber0
end
ber_0 = ber0
