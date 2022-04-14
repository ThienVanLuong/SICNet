'''SICNet with single SNR training - coded system with convolutional codes'''

import numpy as np
# from py_hamming_code import py_hamming_code as phc
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from scipy.stats import levy_stable

# coding package
import sk_dsp_comm.fec_conv as fec
import sk_dsp_comm.digitalcom as dc # for calc BER with coding

# channel coding parameters
decoder_type = 'hard'
# cc1 = fec.fec_conv(('11101','10011'),25) # constraint length 5 and decision depth 25
# d_cc = 24
cc1 = fec.fec_conv(('101', '111'), 10)
d_cc = 9
code_rate = 1/2
state = '0000' # Encode with shift register starting state of '0000'

EbNodB = list(np.linspace(0, 18, 10))
num_blocks = 100
block_size = 2000    # number of uncoded bits per block
channel_type = 'time-varying' # time-varying AWGN
noise_type = 'awgn' #t-dist , stable-dist, awgn, radar

vv = 5  # t-distribution scale
radar_prob = 0.05
radar_power_factor = 4
alpha, beta = 1.8, -0.5
# alpha, beta = 0.5, 0.75

if channel_type == 'AWGN':
    h = np.ones(num_blocks)  # AWGN channel
else:
    h = 0.8 + 0.2 * np.cos((2 * np.pi / 17) * np.arange(num_blocks))
    # h_I = 0.8 + 0.2 * np.cos((2 * np.pi / 17 + np.pi/4) * np.arange(num_blocks))
    # h = 0.8 + 0.01*np.random.normal(0,1,(num_blocks,))
print('channel type: ', channel_type, '; --- noise type: ', noise_type)
print('decoder type: ', decoder_type)

# hyperparameter
snr_train_db = 6
learning_rate = 0.001
batch_size = 200
epochs = 300
act_func = tf.nn.relu

c1 = np.sqrt(16) # power allocation factor
c2 = np.sqrt(4) # power allocation factor

# build model
y = tf.placeholder("float", [None, 1])  # received signal y = hx+n
s1 = tf.placeholder("float", [None, 1])  # bit sequence s, known as label
s2 = tf.placeholder("float", [None, 1])
s3 = tf.placeholder("float", [None, 1])


def decoder1(y):
    s = tf.layers.dense(y, units=16, activation=act_func)
    s = tf.layers.dense(s, units=8, activation=act_func)
    s = tf.layers.dense(s, units=1, activation=tf.nn.sigmoid)
    return s

def decoder2(y):
    s = tf.layers.dense(y, units=24, activation=act_func)
    s = tf.layers.dense(s, units=12, activation=act_func)
    s = tf.layers.dense(s, units=1, activation=tf.nn.sigmoid)
    return s

def decoder3(y):
    s = tf.layers.dense(y, units=32, activation=act_func)
    s = tf.layers.dense(s, units=16, activation=act_func)
    s = tf.layers.dense(s, units=1, activation=tf.nn.sigmoid)
    return s

p1 = decoder1(y)
z2 = tf.concat([y, p1], axis=-1)
p2 = decoder2(z2)
z3 = tf.concat([z2,p2], axis=-1)
p3 = decoder3(z3)
loss = tf.reduce_mean(tf.pow(p1 - s1, 2) + tf.pow(p2 - s2, 2) + tf.pow(p3 - s3, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()


def fetch_batch(epoch, batch_index, batch_size, y_, s1_, s2_, s3_):
    train_size = len(y_)
    n_batches = train_size // batch_size
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(train_size, size=batch_size)
    _y = y_[indices]
    _s1 = s1_[indices]
    _s2 = s2_[indices]
    _s3 = s3_[indices]
    return _y, _s1, _s2, _s3


def generate_data(block_size, noise_std, h):
    uncoded_bits1 = np.random.binomial(n=1, p=0.5, size=(int(block_size*code_rate)))
    coded_bits1, _ = cc1.conv_encoder(uncoded_bits1, state)
    
    uncoded_bits2 = np.random.binomial(n=1, p=0.5, size=(int(block_size*code_rate)))
    coded_bits2, _ = cc1.conv_encoder(uncoded_bits2, state)   

    uncoded_bits3 = np.random.binomial(n=1, p=0.5, size=(int(block_size*code_rate)))
    coded_bits3, _ = cc1.conv_encoder(uncoded_bits3, state)  
    
    x1 = 2 * coded_bits1 - 1
    x2 = 2 * coded_bits2 - 1
    x3 = 2 * coded_bits3 - 1
    x = x3 + c2*x2 + c1*x1

    if noise_type == 'awgn':
        noise = noise_std * np.random.normal(0, 1, x.shape)
    elif noise_type == 't-dist':
        noise = noise_std * np.sqrt((vv - 2) / vv) * np.random.standard_t(vv, size=x.shape)
    elif noise_type == 'radar':
        add_pos = np.random.choice([0.0, 1.0], x.shape, p=[1 - radar_prob, radar_prob])
        corrupted_signal = radar_power_factor * np.random.standard_normal(size=x.shape) * add_pos
        noise = noise_std * (np.random.normal(0, 1, x.shape) + corrupted_signal)
        
    elif noise_type == 'stable-dist':
        noise = noise_std * levy_stable.rvs(alpha, beta, size=x.shape)

    y = h * x + noise
    # quantized, poisson, Laplacian 

    y_ = np.reshape(y, (len(y), 1))
    s1_ = np.reshape(coded_bits1, (len(y), 1))
    s2_ = np.reshape(coded_bits2, (len(y), 1))
    s3_ = np.reshape(coded_bits3, (len(y), 1))
    return y_, s1_, s2_, s3_, uncoded_bits3, coded_bits3


ber_coded = [None] * len(EbNodB)  # ber of coded bits
ber_uncoded = [None] * len(EbNodB)  # ber of uncoded bits


with tf.Session() as sess:
    init.run()
    snr_train = 10 ** (snr_train_db / 10.0)
    noise_std_train = np.sqrt(1 / (snr_train))
    y_, s1_, s2_, s3_, _, _ = generate_data(5000, noise_std_train, h[0])
    
    n_batches = len(y_) // batch_size
    for epoch in range(epochs):
        for batch_index in range(n_batches):
            sys.stdout.flush()
            _y, _s1, _s2, _s3 = fetch_batch(epoch, batch_index, batch_size, y_, s1_, s2_, s3_)
            sess.run(optimizer, feed_dict={y: _y, s1: _s1, s2: _s2, s3: _s3})
        loss_train = loss.eval(feed_dict={y: _y, s1: _s1, s2: _s2, s3: _s3})  # not shown
        print("\r{}".format(epoch), 'Train loss:', loss_train)


    def recover_bits(y_, p, decoder_type):
        s_est = sess.run(p, feed_dict={y: y_})
        s_est = np.reshape(s_est, (len(s_est),))
        s_re = np.sign(s_est - 0.5)
        coded_bits_re = ((s_re + 1) / 2)
        coded_bits_re = coded_bits_re.astype(int)
        if decoder_type == 'hard':
            uncoded_bits_re = (cc1.viterbi_decoder(coded_bits_re, decoder_type)).astype(int)
        else:
            s_refined = np.zeros(s_est.shape)
            for k in range(len(s_est)):
                if s_est[k] == 0.0:
                    s_refined[k] = 1e-16
                elif s_est[k] == 1.0:
                    s_refined[k] = 1.0 - 1e-16
                else:
                    s_refined[k] = s_est[k]
            bits_llr = 10*np.log10(s_refined) - 10*np.log10(1-s_refined) 
            uncoded_bits_re = (cc1.viterbi_decoder(bits_llr, decoder_type)).astype(int)          
        return uncoded_bits_re, coded_bits_re

  
    for ii in range(len(EbNodB)):
        EbNo = 10 ** (EbNodB[ii] / 10.0)
        noise_std = np.sqrt(1 / (EbNo))
        num_errors_uncoded = 0
        num_errors_coded = 0
        uncoded_bit_count = 0
        coded_bit_count = 0

        for jj in range(num_blocks):
            y_, _, _, _, uncoded_bits3, coded_bits3 = generate_data(block_size, noise_std, h[jj])
            uncoded_bits_re3, coded_bits_re3 = recover_bits(y_, p3, decoder_type)   
            uncoded_bit_count, errors_uncoded = dc.bit_errors(coded_bits3, coded_bits_re3)
            coded_bit_count, errors_coded = dc.bit_errors(uncoded_bits3, uncoded_bits_re3)

            num_errors_uncoded = num_errors_uncoded + errors_uncoded
            num_errors_coded = num_errors_coded + errors_coded
    
        ber_uncoded[ii] = num_errors_uncoded / num_blocks / uncoded_bit_count
        ber_coded[ii] = num_errors_coded / num_blocks / coded_bit_count

print('Coded BER: ', np.round(ber_coded, 6))
print('Uncoded BER: ', np.round(ber_uncoded, 6))

np.reshape(ber_coded, (len(ber_uncoded),))
np.reshape(ber_uncoded, (len(ber_uncoded),))

EbNodB_coded = list(np.asarray(EbNodB)-10*np.log10(code_rate))
plt.plot(EbNodB_coded, ber_coded,'-r+',label='coded bits')
plt.plot(EbNodB, ber_uncoded,'-bo',label='Uncoded bits')
plt.yscale('log')
plt.xlabel('EbNo (dB)')
plt.ylabel('BER')
plt.title('NOMA 2 users under '+channel_type+' channel, noise type: '+noise_type)
plt.grid()
fig = plt.gcf()
# fig.set_size_inches(16,12)
# fig.savefig('graph/0501/rayleighBLER2.png',dpi=100)
plt.show()
