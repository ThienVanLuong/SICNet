%% Downlink NOMA with 3 users
% Modulation BPSK, real signal

%%
% clc;
clear all;

M = 4; % Modulation
m = log2(M); 
num_sample = 1e6;
snr_db = 0:3:0;
global s_nLevels;
global s_fDynRange;

s_nLevels = 8;
s_fDynRange = 8;

h0 = (1+1i)/sqrt(2);
h = [h0 h0 h0];
iter = 1;

sym_ref = qammod([0,1,2,3],M,'Gray')/sqrt(2);

ser0 = zeros(1,length(snr_db));
for c=1:iter
    ser3 = zeros(1,length(snr_db));
    for k=1:length(snr_db)
        snr = 10.^(snr_db(k)/10);

        bit1 = randi([0 1],num_sample,m);
        bit2 = randi([0 1],num_sample,m);
        bit3 = randi([0 1],num_sample,m);
        
        dec1 = BitoDe(bit1);
        dec2 = BitoDe(bit2);
        dec3 = BitoDe(bit3);
        
        x1 = qammod(dec1,M,'Gray')/sqrt(2);
        x2 = qammod(dec2,M,'Gray')/sqrt(2);
        x3 = qammod(dec3,M,'Gray')/sqrt(2);
        
    %     x = x1 + x2 + x3;
        p2 = sqrt(4);
        p1 = sqrt(16);
        x = x3 + p2*x2 + p1*x1;

        noise_std3 = (1/snr);
        noise_std2 = noise_std3/1;
        noise_std1 = noise_std2/1;

        noise3 = sqrt(noise_std3/2)*(randn(1,num_sample)+1i*randn(1,num_sample));
        y3 = h(3)*x + noise3;
        
%         h3_est = mean(1 + randn(1000,num_sample)/sqrt(db2pow(6))); 
%         h3_est = h3_est';
        
        h3_est = h(3)*ones(1,num_sample) + randn(1,num_sample)/sqrt(2*100)+1i*randn(1,num_sample)/sqrt(2*100); 
        y3_est = h3_est.*x + noise3;

        %quantized Gaussian signal
%         y3_gau_quant = m_fQuant(y3);
%         y3_est_gau_quant = m_fQuant(y3_est);
      

%% detection SIC
tic
        r3 = y3;
        h3 = h(3);
        re_x1 = qammod(qamdemod(sqrt(2)*r3/h3,M,'Gray'),M,'Gray')/sqrt(2);
        r3 = r3 - p1*h3*re_x1;
        re_x2 = qammod(qamdemod(sqrt(2)*r3/h3,M,'Gray'),M,'Gray')/sqrt(2);
        r3 = r3 - p2*h3*re_x2;
        dec3_hat = qamdemod(sqrt(2)*r3/h3,M,'Gray');
toc
%         r3 = y3;
%         h3 = h3_est;
%         re_x1 = qammod(qamdemod(sqrt(2)*r3./h3,M,'Gray'),M,'Gray')/sqrt(2);
%         r3 = r3 - p1*h3.*re_x1;
%         re_x2 = qammod(qamdemod(sqrt(2)*r3./h3,M,'Gray'),M,'Gray')/sqrt(2);
%         r3 = r3 - p2*h3.*re_x2;
%         dec3_hat = qamdemod(sqrt(2)*r3./h3,M,'Gray');
%         
        err_3 = sum(dec3_hat~=dec3);
        ser3(k) = err_3/num_sample;  
    end
% ber3
ser0 = ser0 + ser3/iter;
% ber0
end
ser_conSIC = ser0

[0.475030000000000,0.255650000000000,0.0790150000000000,0.00850000000000000,7.00000000000000e-05,0];
% ser_SICNet = [0.477875, 0.265535, 0.08637, 0.01127, 0.000325, 2e-6];
ser_SICNet = [0.52437 0.30156 0.11264 0.02295 0.00351 0.0006 ];

figure(1)
semilogy(snr_db,ser_conSIC,'k >-','LineWidth',1.5,'MarkerSize',8)
hold on
semilogy(snr_db,ser_SICNet,'b o-','LineWidth',1.5,'MarkerSize',8)
hold on
axis([0 snr_db(end) 10^-5 10^0])
grid on
hold on
title('')
xticks([0,3,6,9,12,15])
legend('Conv. SIC', 'Our SICNet', 'FontSize', 10)
xlabel('SNR (dB)')
ylabel('SER')
