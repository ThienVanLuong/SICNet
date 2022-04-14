%% Downlink NOMA with 3 users
% Modulation BPSK, real signal

%%
% clc;
clear all;

M = 2; % Modulation
num_sample = 5e3;
snr_db = 0:3:45;
global s_nLevels;
global s_fDynRange;

s_nLevels = 4;
s_fDynRange = 4;

h = [1 1];
iter = 1;

ber0 = zeros(1,length(snr_db));
for c=1:iter
    ber2 = zeros(1,length(snr_db));
    for k=1:length(snr_db)
        snr = 10.^(snr_db(k)/10);

        bit1 = randi([0 1],num_sample,1);
        bit2 = randi([0 1],num_sample,1);

        x1 = 2*bit1 - 1;
        x2 = 2*bit2 - 1;
        
        p1 = sqrt(16);
        x = x2 + p1*x1;

        noise_std2 = (1/snr);
        noise_std1 = noise_std2/1;

        noise1 = sqrt(noise_std1)*randn(num_sample,1);
        noise2 = sqrt(noise_std2)*randn(num_sample,1);

        y1 = h(1)*x + noise1;
        y2 = h(2)*x + noise2;

        h2_est = h(2)*ones(num_sample,1) + randn(num_sample,1)/sqrt(20); 
        y2_est = h2_est.*x + noise2;

        %quantized Gaussian signal
        y2_gau_quant = m_fQuant(y2);
        y2_est_gau_quant = m_fQuant(y2_est);      
    
        %%poisson channel
        z1 = bit1;
        z2 = bit2;
        
%         p1 = 4;       
        z = z2 + p1*z1;    
        y2_poi_quant = poissrnd(sqrt(snr/1)*h(2)*z  + 1);
        y2_est_poi_quant = poissrnd(sqrt(snr/1)*h2_est.*z  + 1); %17.5755

%         save(strcat('./data/train_data_2users_',num2str(snr_db(k)),'dB.mat'),'x1','x2', 'bit1', 'bit2', 'y2', 'y2_est', 'y2_gau_quant', 'y2_est_gau_quant', 'y2_poi_quant', 'y2_est_poi_quant')

%% detection SIC
        x2_hat = zeros(num_sample,1);
        for n=1:num_sample
            r2 = y2(n);
%             h2 = h2_est(n);
            h2 = h(2);

            re_x1 = sign(r2/h2);
            r2 = r2 - p1*h2*re_x1;
            
            re_x2 = sign(r2/h2);
            x2_hat(n) = re_x2;
        end
        err_2 = sum(x2_hat~=x2);
        ber2(k) = err_2/num_sample;  
    end
% ber3
ber0 = ber0 + ber2/iter;
% ber0
end
ber_0 = ber0
