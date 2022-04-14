%% Downlink NOMA with 3 users
% Modulation BPSK, real signal

%%
% clc;
clear all;

M = 2; % Modulation
num_sample = 2e6;
snr_db = 0:3:12;
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
        p1 = sqrt(16);
        x = x3 + p2*x2 + p1*x1;

        noise_std3 = (1/snr);
        noise_std2 = noise_std3/1;
        noise_std1 = noise_std2/1;

        noise1 = sqrt(noise_std1)*randn(num_sample,1);
        noise2 = sqrt(noise_std2)*randn(num_sample,1);
        noise3 = sqrt(noise_std3)*randn(num_sample,1);

        y1 = h(1)*x + noise1;
        y2 = h(2)*x + noise2;
        y3 = h(3)*x + noise3;
        
%         h3_est = mean(1 + randn(1000,num_sample)/sqrt(db2pow(6))); 
%         h3_est = h3_est';
        
        h3_est = h(3)*ones(num_sample,1) + randn(num_sample,1)/sqrt(100); 
        y3_est = h3_est.*x + noise3;

        %quantized Gaussian signal
        y3_gau_quant = m_fQuant(y3);
        y3_est_gau_quant = m_fQuant(y3_est);
        
        %clipping to reduce PAPR
    %     A = 7;
    %     y3_clipped = y3;
    %     y3_clipped(find(y3_clipped > A)) = A;
    %     y3_clipped(find(y3_clipped < -A)) = -A;
    %     
    %     y3_est_clipped = y3_est;
    %     y3_est_clipped(find(y3_est_clipped > A)) = A;
    %     y3_est_clipped(find(y3_est_clipped < -A)) = -A;    
    
        %%poisson channel
        z1 = bit1;
        z2 = bit2;
        z3 = bit3;
        
%         p1 = 8;
%         p2 = 2;
        
        z = z3 + p2*z2 + p1*z1;    
        y3_poi_quant = poissrnd(sqrt(snr/1)*h(3)*z  + 1);
        y3_est_poi_quant = poissrnd(sqrt(snr/1)*h3_est.*z  + 1); %17.5755
        

%% detection SIC
        x3_hat = zeros(num_sample,1);
        for n=1:num_sample
            r3 = y3(n);
%             h3 = h3_est(n);
            h3 = h3_est(3);

            re_x1 = sign(r3/h3);
            r3 = r3 - p1*h3*re_x1;
    %         r3 = r3 - p1*h3*x1(n);

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

figure(1)
semilogy(snr_db,ber0,'k >-','LineWidth',1.5,'MarkerSize',8)
hold on
axis([0 snr_db(end) 10^-5 10^0])
grid on
hold on
title('')
xlabel('SNR (dB)')
ylabel('SER')

