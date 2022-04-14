'''SICNet with single SNR training - Complex symbols QPSK'''

# from utils import generate_data_qam_symbols
import numpy as np
# from py_hamming_code import py_hamming_code as phc
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from scipy.stats import levy_stable
cce = tf.keras.losses.CategoricalCrossentropy()

M = 4 # M-ary modulation size
K = 3 # number of users
m = int(np.log2(M))

epsilon = 0.01 # CSI error variance
EbNodB = list(np.linspace(0, 15, 6))
num_symbols_train = 5000
num_symbols_test = 400000
noise_type = 'awgn' #t-dist , stable-dist, awgn, radar

vv = 5  # t-distribution scale
radar_prob = 0.05
radar_power_factor = 4
alpha, beta = 1.8, -0.5
# alpha, beta = 0.5, 0.75

# hyperparameter
snr_train_db = 8
learning_rate = 0.001
batch_size = 200
epochs = 250
act_func = tf.nn.relu

c1 = np.sqrt(16) # power allocation factor
c2 = np.sqrt(4) # power allocation factor
h = (1 + 1j*1)/np.sqrt(2)

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
y = tf.placeholder("float", [None, 2])  # received signal y = hx+n
s1 = tf.placeholder("float", [None, M])  # bit sequence s, known as label
s2 = tf.placeholder("float", [None, M])
s3 = tf.placeholder("float", [None, M])

num_nodes = [24,32,48]

def decoder1(y):
    s = tf.layers.dense(y, units=num_nodes[0], activation=act_func)
    s = tf.layers.dense(s, units=int(num_nodes[0]/2), activation=act_func)
    s = tf.layers.dense(s, units=M, activation=tf.nn.softmax)
    return s

def decoder2(y):
    s = tf.layers.dense(y, units=num_nodes[1], activation=act_func)
    s = tf.layers.dense(s, units=int(num_nodes[1]/2), activation=act_func)
    s = tf.layers.dense(s, units=M, activation=tf.nn.softmax)
    return s

def decoder3(y):
    s = tf.layers.dense(y, units=num_nodes[2], activation=act_func)
    s = tf.layers.dense(s, units=int(num_nodes[2]/2), activation=act_func)
    s = tf.layers.dense(s, units=M, activation=tf.nn.softmax)
    return s

p1 = decoder1(y)
z2 = tf.concat([y, p1], axis=-1)
p2 = decoder2(z2)
z3 = tf.concat([z2,p2], axis=-1)
p3 = decoder3(z3)

# loss = tf.reduce_mean(0*tf.pow(p1 - s1, 2) + 0*tf.pow(p2 - s2, 2) + tf.pow(p3 - s3, 2))
# loss = tf.reduce_mean(1*cce(p1,s1)+ 1*cce(p2,s2)+ cce(p3,s3))
loss = - tf.reduce_mean(0*s1*tf.log(p1)+0*s2*tf.log(p2)+s3*tf.log(p3))
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


# generate data
def qam_constellation(M):
    a = 1/np.sqrt(2)
    b = np.sqrt(3)
    if M==4:
        QAM = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=complex) /np.sqrt(2) # gray mapping
    elif M==8:
        QAM = np.array([1, a+a*1j, -a+a*1j, 1j, a-a*1j, -1j, -1, -a-a*1j], dtype=complex) # 8PSK, not 8QAM indeed
        QAM = np.array([-3+1j, -3-1j, -1+1j, -1-1j, 3+1j, 3-1j, 1+1j, 1-1j], dtype=complex)  #gray qam

    elif M==16:
        QAM = np.array([-3+3j, -3+1j, -3-3j, -3-1j,
                        -1+3j, -1+1j, -1-3j, -1-1j,
                        3+3j, 3+1j, 3-3j, 3-1j,
                        1+3j, 1+1j, 1-3j, 1-1j], dtype=complex)

    elif M==2:
        QAM = np.array([-1, 1], dtype=complex) #BPSK
    else:
        raise ValueError('Modulation order must be in {2,4,8,16}.')
    return QAM

def generate_data_qam_symbols(M=4, N=16, num_samples=5): 
    m = int(np.log2(M))
    QAM = qam_constellation(M)
    bits = np.random.binomial(n=1, p=0.5, size=(num_samples,N,m))
    sym_one_hot = np.zeros((num_samples,N,M), dtype=int)
    sym_com = np.zeros((num_samples,N), dtype=complex)
    for i in range(num_samples):
        bit = bits[i]
        sym = np.zeros((N,), dtype=complex)
        for j in range(N):
            sym_id = bit[j].dot(2**np.arange(bit[j].size)[::-1])
            sym[j] = QAM[sym_id]
            sym_one_hot[i,j,sym_id] = 1
        sym_com[i] = sym          
    return bits, sym_com, sym_one_hot

def generate_data(M, K, noise_std, h, num_symbols):
    bits, sym_com, sym_one_hot = generate_data_qam_symbols(M=M, N=K, num_samples=num_symbols)
    bits3 = bits[:,2,:]

    x1 = sym_com[:,0]
    x2 = sym_com[:,1]
    x3 = sym_com[:,2]
    x = x3 + c2*x2 + c1*x1

    if noise_type == 'awgn':
        noise = noise_std * (np.random.normal(0, 1, x.shape) + 1j*np.random.normal(0, 1, x.shape))
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

    y_ = np.stack([np.real(y), np.imag(y)], axis=1)
    s1_ = sym_one_hot[:,0,:]
    s2_ = sym_one_hot[:,1,:]
    s3_ = sym_one_hot[:,2,:]
    return y_, s1_, s2_, s3_, bits3


ber_coded = [None] * len(EbNodB)  # ber of coded bits
ser = [None] * len(EbNodB)  # ber of uncoded bits


with tf.Session() as sess:
    init.run()
    snr_train = 10 ** (snr_train_db / 10.0)
    noise_std_train = np.sqrt(1/(snr_train)/2)
    h_train = h + np.sqrt(epsilon/2)*(np.random.normal(0, 1, num_symbols_train) + 1j*np.random.normal(0, 1, num_symbols_train))
    y_, s1_, s2_, s3_, _ = generate_data(M, K, noise_std_train, h_train, num_symbols_train)

    n_batches = len(y_) // batch_size
    for epoch in range(epochs):
        for batch_index in range(n_batches):
            sys.stdout.flush()
            _y, _s1, _s2, _s3 = fetch_batch(epoch, batch_index, batch_size, y_, s1_, s2_, s3_)
            sess.run(optimizer, feed_dict={y: _y, s1: _s1, s2: _s2, s3: _s3})
        loss_train = loss.eval(feed_dict={y: _y, s1: _s1, s2: _s2, s3: _s3})  # not shown
        if epoch % 10 == 0: 
            print("\r{}".format(epoch), 'Train loss:', loss_train)

  
    for ii in range(len(EbNodB)):
        EbNo = 10 ** (EbNodB[ii] / 10.0)
        noise_std = np.sqrt(1/2/(EbNo))
        num_errors_uncoded = 0
        num_errors_coded = 0
        uncoded_bit_count = 0
        coded_bit_count = 0
        
        h_test = h 
        y_, _, _, s3_, bits3 = generate_data(M, K, noise_std, h_test, num_symbols_test)
        s3_est = sess.run(p3, feed_dict={y: y_})
        s3_dec = np.argmax(s3_, axis=1)
        s3_dec_est = np.argmax(s3_est, axis=1)
        num_errors_uncoded = sum(s3_dec!=s3_dec_est)
        ser[ii] = num_errors_uncoded / num_symbols_test


print('SER: ', np.round(ser, 6))
np.reshape(ber_coded, (len(ser),))

plt.plot(EbNodB, ser,'-bo',label='Uncoded bits')
plt.yscale('log')
plt.xlabel('EbNo (dB)')
plt.ylabel('SER')
plt.title('NOMA 3 users under channel, noise type: '+noise_type)
plt.grid()
fig = plt.gcf()
plt.show()

