'''SICNet with single SNR training - add user 4 to 3-user system'''

import numpy as np
# from py_hamming_code import py_hamming_code as phc
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from scipy.stats import levy_stable

epsilon = 0.01 # CSI error variance
EbNodB = list(np.linspace(0, 15, 6))
num_bits_train = 5000
num_bits_test = 200000
noise_type = 'awgn' #t-dist , stable-dist, awgn, radar

vv = 5  # t-distribution scale
radar_prob = 0.05
radar_power_factor = 4
alpha, beta = 1.8, -0.5
# alpha, beta = 0.5, 0.75

# hyperparameter
snr_train_db = 6
learning_rate = 0.0002
batch_size = 200
epochs = 500
act_func = tf.nn.relu

c1 = np.sqrt(16) # power allocation factor
c2 = np.sqrt(4) # power allocation factor
c0 = np.sqrt(1/9)
h = 1

# quantizaition parameters
q_levels = 8
q_range = 8

def quantized(y, q_levels, q_range):
    delta = 2*q_range/q_levels
    value_max = q_range - delta/2
    y_quantized = delta*np.floor((y + q_range) / delta) - value_max
    y_quantized[np.where(y_quantized > value_max)] = value_max
    y_quantized[np.where(y_quantized < -value_max)] = -value_max
    
    return y_quantized

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
loss = tf.reduce_mean(tf.pow(p3 - s3, 2))
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


def generate_data(num_bits, noise_std, h):
    bits1 = np.random.binomial(n=1, p=0.5, size=num_bits)
    bits2 = np.random.binomial(n=1, p=0.5, size=num_bits)
    bits3 = np.random.binomial(n=1, p=0.5, size=num_bits)

    x1 = 2 * bits1 - 1
    x2 = 2 * bits2 - 1
    x3 = 2 * bits3 - 1
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
    # y = quantized(y, q_levels, q_range)
    # quantized, poisson, Laplacian 

    y_ = np.reshape(y, (len(y), 1))
    s1_ = np.reshape(bits1, (len(y), 1))
    s2_ = np.reshape(bits2, (len(y), 1))
    s3_ = np.reshape(bits3, (len(y), 1))
    return y_, s1_, s2_, s3_, bits3


def generate_data_change_order(num_bits, noise_std, h):
    bits1 = np.random.binomial(n=1, p=0.5, size=num_bits)
    bits2 = np.random.binomial(n=1, p=0.5, size=num_bits)
    bits3 = np.random.binomial(n=1, p=0.5, size=num_bits)

    x1 = 2 * bits1 - 1
    x2 = 2 * bits2 - 1
    x3 = 2 * bits3 - 1
    x = x3 + c2*x2 + x1/3

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
    # y = quantized(y, q_levels, q_range)
    # quantized, poisson, Laplacian 

    y_ = np.reshape(y, (len(y), 1))
    s1_ = np.reshape(bits1, (len(y), 1))
    s2_ = np.reshape(bits2, (len(y), 1))
    s3_ = np.reshape(bits3, (len(y), 1))
    return y_, s1_, s2_, s3_, bits3

def generate_data_user4(num_bits, noise_std, h):
    bits1 = np.random.binomial(n=1, p=0.5, size=num_bits)
    bits2 = np.random.binomial(n=1, p=0.5, size=num_bits)
    bits3 = np.random.binomial(n=1, p=0.5, size=num_bits)
    bits4 = np.random.binomial(n=1, p=0.5, size=num_bits)

    x1 = 2 * bits1 - 1
    x2 = 2 * bits2 - 1
    x3 = 2 * bits3 - 1
    x4 = 2 * bits4 - 1
    x = x3 + c2*x2 + c1*x1 +c0*x4

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
    # y = quantized(y, q_levels, q_range)
    # quantized, poisson, Laplacian 

    y_ = np.reshape(y, (len(y), 1))
    s1_ = np.reshape(bits1, (len(y), 1))
    s2_ = np.reshape(bits2, (len(y), 1))
    s3_ = np.reshape(bits3, (len(y), 1))
    return y_, s1_, s2_, s3_, bits3


ber_coded = [None] * len(EbNodB)  # ber of coded bits
ber_uncoded = [None] * len(EbNodB)  # ber of uncoded bits


with tf.Session() as sess:
    init.run()
    snr_train = 10 ** (snr_train_db / 10.0)
    noise_std_train = np.sqrt(1 / (snr_train))
    h_train = h + np.sqrt(epsilon)*np.random.normal(0, 1, num_bits_train)
    y_, s1_, s2_, s3_, _ = generate_data_change_order(num_bits_train, noise_std_train, h_train)

    n_batches = len(y_) // batch_size
    for epoch in range(epochs):
        for batch_index in range(n_batches):
            sys.stdout.flush()
            _y, _s1, _s2, _s3 = fetch_batch(epoch, batch_index, batch_size, y_, s1_, s2_, s3_)
            sess.run(optimizer, feed_dict={y: _y, s1: _s1, s2: _s2, s3: _s3})
        loss_train = loss.eval(feed_dict={y: _y, s1: _s1, s2: _s2, s3: _s3})  # not shown
        if epoch % 10 == 0: 
            print("\r{}".format(epoch), 'Train loss:', loss_train)


    def recover_bits(y_, p):
        s_est = sess.run(p, feed_dict={y: y_})
        s_est = np.reshape(s_est, (len(s_est),))
        s_re = np.sign(s_est - 0.5)
        bits_re = ((s_re + 1) / 2)
        bits_re = bits_re.astype(int)       
        return bits_re

  
    for ii in range(len(EbNodB)):
        EbNo = 10 ** (EbNodB[ii] / 10.0)
        noise_std = np.sqrt(1 / (EbNo))
        num_errors_uncoded = 0
        num_errors_coded = 0
        uncoded_bit_count = 0
        coded_bit_count = 0
        
        h_test = h + np.sqrt(0)*np.random.normal(0, 1, num_bits_test)
        # h_test = 0.6
        y_, _, _, _, bits3 = generate_data_change_order(num_bits_test, noise_std, h_test)
        bits_re3 = recover_bits(y_, p3)   
        num_errors_uncoded = sum(bits_re3!=bits3)
        ber_uncoded[ii] = num_errors_uncoded / num_bits_test


print('Uncoded BER: ', np.round(ber_uncoded, 6))
np.reshape(ber_coded, (len(ber_uncoded),))

plt.plot(EbNodB, ber_uncoded,'-bo',label='Uncoded bits')
plt.yscale('log')
plt.xlabel('EbNo (dB)')
plt.ylabel('BER')
plt.title('NOMA 2 users under channel, noise type: '+noise_type)
plt.grid()
fig = plt.gcf()
# fig.set_size_inches(16,12)
# fig.savefig('graph/0501/rayleighBLER2.png',dpi=100)
plt.show()
