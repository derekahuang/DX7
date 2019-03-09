#!/usr/bin/env ipython
#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:




# time parameters
fs = 8000 # sampling rate
T = 1.0/fs # sampling period
dur = 1 # duration in seconds
t = np.linspace(0,(dur-T),num=fs*dur) # time vector


# In[5]:


def adsr_env(dur,fs,a,d,s,r):
    
    if a+d+r < 1.0:
        a_samps = int(dur*fs*a)
        d_samps = int(dur*fs*d)        
        r_samps = int(dur*fs*r)
        s_samps = dur*fs - a_samps - d_samps - r_samps

        adsr = np.concatenate((np.linspace(0,1,num=a_samps), 
        np.linspace(1,s,num=d_samps),
        s*np.ones(s_samps),                       
        np.linspace(s,0,num=r_samps)))
        
    else:
        print("a+d+r must be less than 1")

    return adsr

def fm_sample(dur,fs,t,A1,A2,A3,A4,A5,A6,f1,f2,f3,f4,f5,f6,a1,d1,s1,r1,a2,d2,s2,r2,a3,d3,s3,r3,a4,d4,s4,r4,a5,d5,s5,r5,a6,d6,s6,r6):
    
    adsr1 = adsr_env(dur,fs,a1,d1,s1,r1)
    x1 = np.multiply(A1*np.sin(2*np.pi*f1*t),adsr1)
    adsr2 = adsr_env(dur,fs,a2,d2,s2,r2)
    x2 = np.multiply(A2*np.sin(2*np.pi*f2*t + x1),adsr2)
    adsr3 = adsr_env(dur,fs,a3,d3,s3,r3)
    x3 = np.multiply(A3*np.sin(2*np.pi*f3*t + x2),adsr3)

    adsr4 = adsr_env(dur,fs,a4,d4,s4,r4)
    x4 = np.multiply(A4*np.sin(2*np.pi*f4*t),adsr4)
    adsr5 = adsr_env(dur,fs,a5,d5,s5,r5)
    x5 = np.multiply(A5*np.sin(2*np.pi*f5*t + x4),adsr5)
    adsr6 = adsr_env(dur,fs,a6,d6,s6,r6)
    x6 = np.multiply(A6*np.sin(2*np.pi*f6*t + x5),adsr6)
    
    x = x3+x6
    
    return x


# In[9]:

def generate_fixed_samples(n, var_id):

    data = np.zeros((1, 8000))
    param = np.zeros((1, 36))

    # oscillators' parameters (all real numbers possible, but some may not give results that can be heard by the human ear)
    A = np.random.rand(6)
    A1 = A[0] # between 0 and 1
    A2 = A[1]# between 0 and 1
    A3 = A[2]# between 0 and 1
    A4 = A[3]# between 0 and 1
    A5 = A[4]# between 0 and 1
    A6 = A[5]# between 0 and 1
    f1 = np.random.uniform(0,100) #110 # frequency in herz (not above 100)
    f2 = np.random.uniform(0,100) # frequency in herz (not above 1000)
    f3 = np.random.uniform(50,200) # frequency in herz (not above 2000, not below 50)
    f4 = np.random.uniform(0,100) # frequency in herz (not above 100)
    f5 = np.random.uniform(0,100)  # frequency in herz (not above 1000)
    f6 = np.random.uniform(50,200)  # frequency in herz (not above 2000, not below 50)

    # adsr parameters (a,d,s,r must be positive numbers between 0 and 1. Important: a+d+r must be less than 1)
    # a is attack time (as a fraction of the total duration of the adsr envelope)
    # d is decay time (as a fraction of the total duration of the adsr envelope)
    # s is the sustain level, as a fraction of the largest value in the adsr envelope (largest value is always 1)
    # r is release time (as a fraction of the total duration of the adsr envelope)
    vals = np.random.uniform(.1, .3, 3)
    div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
    vals = np.divide(vals, div)
    a1 = vals[0]
    d1 = vals[1]
    s1 = np.random.uniform(.5, 1) # (not below 0.5)
    r1 = vals[2]
    vals = np.random.uniform(.1, .3, 3)
    div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
    vals = np.divide(vals, div)
    a2 = vals[0]
    d2 = vals[1]
    s2 = np.random.uniform(.5, 1) # (not below 0.5)
    r2 = vals[2]
    vals = np.random.uniform(.1, .3, 3)
    div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
    vals = np.divide(vals, div)
    a3 = vals[0]
    d3 = vals[1]
    s3 = np.random.uniform(.5, 1) # (not below 0.5)
    r3 = vals[2]
    vals = np.random.uniform(.1, .3, 3)
    div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
    vals = np.divide(vals, div)
    a4 = vals[0]
    d4 = vals[1]
    s4 = np.random.uniform(.5, 1) # (not below 0.5)
    r4 = vals[2]
    vals = np.random.uniform(.1, .3, 3)
    div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
    vals = np.divide(vals, div)
    a5 = vals[0]
    d5 = vals[1]
    s5 = np.random.uniform(.5, 1) # (not below 0.5)
    r5 = vals[2]
    vals = np.random.uniform(.1, .3, 3)
    div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
    vals = np.divide(vals, div)
    a6 = vals[0]
    d6 = vals[1]
    s6 = np.random.uniform(.5, 1) # (not below 0.5)
    r6 = vals[2]
    x = fm_sample(dur,fs,t,A1,A2,A3,A4,A5,A6,f1,f2,f3,f4,f5,f6,a1,d1,s1,r1,a2,d2,s2,r2,a3,d3,s3,r3,a4,d4,s4,r4,a5,d5,s5,r5,a6,d6,s6,r6) # time domain
    x = x/np.max(np.abs(x)) # normalize
    data = np.r_[data, x.reshape(8000,1).T]
    cur_param = np.array([A1,A2,A3,A4,A5,A6,f1,f2,f3,f4,f5,f6,a1,d1,s1,r1,a2,d2,s2,r2,a3,d3,s3,r3,a4,d4,s4,r4,a5,d5,s5,r5,a6,d6,s6,r6]).reshape(36, 1)
    param = np.r_[param, cur_param.T]
    # X = np.abs(np.fft.fft(x)) # spectrum magnitude
    # X = X[0:int(len(X)/2)] # visualize only positive frequencies
    # f = np.linspace(0,fs/2,num=len(X)) # frequency vector

    # plt.subplot(2,1,1)
    # plt.plot(t,x)
    # plt.subplot(2,1,2)
    # plt.plot(f,X)
    # plt.show()
    # print(cur_param.T)
    return np.delete(data, (0), axis=0), np.delete(param, (0), axis=0)
        # np.save('data.npy', np.delete(data, (0), axis=0))
        # np.save('params.npy', np.delete(param, (0), axis=0))

# visualization of results
def generate_samples(n):

    data = np.zeros((1, 8000))
    param = np.zeros((1, 36))

    for i in range(n):
        # oscillators' parameters (all real numbers possible, but some may not give results that can be heard by the human ear)
        A = np.random.rand(6)
        A1 = A[0] # between 0 and 1
        A2 = A[1]# between 0 and 1
        A3 = A[2]# between 0 and 1
        A4 = A[3]# between 0 and 1
        A5 = A[4]# between 0 and 1
        A6 = A[5]# between 0 and 1
        f1 = np.random.uniform(0,100) #110 # frequency in herz (not above 100)
        f2 = np.random.uniform(0,100) # frequency in herz (not above 1000)
        f3 = np.random.uniform(50,200) # frequency in herz (not above 2000, not below 50)
        f4 = np.random.uniform(0,100) # frequency in herz (not above 100)
        f5 = np.random.uniform(0,100)  # frequency in herz (not above 1000)
        f6 = np.random.uniform(50,200)  # frequency in herz (not above 2000, not below 50)

        # adsr parameters (a,d,s,r must be positive numbers between 0 and 1. Important: a+d+r must be less than 1)
        # a is attack time (as a fraction of the total duration of the adsr envelope)
        # d is decay time (as a fraction of the total duration of the adsr envelope)
        # s is the sustain level, as a fraction of the largest value in the adsr envelope (largest value is always 1)
        # r is release time (as a fraction of the total duration of the adsr envelope)
        vals = np.random.uniform(.1, .3, 3)
        div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
        vals = np.divide(vals, div)
        a1 = vals[0]
        d1 = vals[1]
        s1 = np.random.uniform(.5, 1) # (not below 0.5)
        r1 = vals[2]
        vals = np.random.uniform(.1, .3, 3)
        div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
        vals = np.divide(vals, div)
        a2 = vals[0]
        d2 = vals[1]
        s2 = np.random.uniform(.5, 1) # (not below 0.5)
        r2 = vals[2]
        vals = np.random.uniform(.1, .3, 3)
        div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
        vals = np.divide(vals, div)
        a3 = vals[0]
        d3 = vals[1]
        s3 = np.random.uniform(.5, 1) # (not below 0.5)
        r3 = vals[2]
        vals = np.random.uniform(.1, .3, 3)
        div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
        vals = np.divide(vals, div)
        a4 = vals[0]
        d4 = vals[1]
        s4 = np.random.uniform(.5, 1) # (not below 0.5)
        r4 = vals[2]
        vals = np.random.uniform(.1, .3, 3)
        div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
        vals = np.divide(vals, div)
        a5 = vals[0]
        d5 = vals[1]
        s5 = np.random.uniform(.5, 1) # (not below 0.5)
        r5 = vals[2]
        vals = np.random.uniform(.1, .3, 3)
        div = np.random.uniform(1.5 * np.sum(vals), np.sum(vals))
        vals = np.divide(vals, div)
        a6 = vals[0]
        d6 = vals[1]
        s6 = np.random.uniform(.5, 1) # (not below 0.5)
        r6 = vals[2]
        x = fm_sample(dur,fs,t,A1,A2,A3,A4,A5,A6,f1,f2,f3,f4,f5,f6,a1,d1,s1,r1,a2,d2,s2,r2,a3,d3,s3,r3,a4,d4,s4,r4,a5,d5,s5,r5,a6,d6,s6,r6) # time domain
        x = x/np.max(np.abs(x)) # normalize
        data = np.r_[data, x.reshape(8000,1).T]
        cur_param = np.array([A1,A2,A3,A4,A5,A6,f1,f2,f3,f4,f5,f6,a1,d1,s1,r1,a2,d2,s2,r2,a3,d3,s3,r3,a4,d4,s4,r4,a5,d5,s5,r5,a6,d6,s6,r6]).reshape(36, 1)
        param = np.r_[param, cur_param.T]
        # X = np.abs(np.fft.fft(x)) # spectrum magnitude
        # X = X[0:int(len(X)/2)] # visualize only positive frequencies
        # f = np.linspace(0,fs/2,num=len(X)) # frequency vector

        # plt.subplot(2,1,1)
        # plt.plot(t,x)
        # plt.subplot(2,1,2)
        # plt.plot(f,X)
        # plt.show()
        # print(cur_param.T)
    return np.delete(data, (0), axis=0), np.delete(param, (0), axis=0)
        # np.save('data.npy', np.delete(data, (0), axis=0))
        # np.save('params.npy', np.delete(param, (0), axis=0))





