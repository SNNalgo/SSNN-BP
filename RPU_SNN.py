import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

def thresholding(v_in, th_val, th):
    th.assign(th_val*tf.ones_like(v_in[:,:,0,:]))
    shape = v_in.get_shape()
    op_list = []
    for i in range(shape[2]):
        sl = v_in[:,:,i,:] > th
        y = tf.to_float(sl)
        th.assign_add(th_val*y)
        op_list.append(y)
    return tf.stack(op_list, axis=2)

def spike_multiply(del_post, s_pre):
    # Assuming input shape : [batch_size, num_iter, N]
    shape = del_post.get_shape()
    b_sz = shape[0]
    n_it = shape[1]
    dW = tf.matmul(tf.transpose(s_pre[0,:,:]), del_post[0,:,:])/tf.to_float(n_it)
    for i in range(1,b_sz):
        dW += tf.matmul(tf.transpose(s_pre[i,:,:]), del_post[i,:,:])/tf.to_float(n_it)
    return dW/tf.to_float(b_sz)

def apply_relu_grad(del_post, s_post):
    # Assuming input shape : [batch_size, num_iter, N]
    shape = del_post.get_shape()
    b_sz = shape[0]
    n_it = shape[1]
    op_list = []
    for i in range(n_it):
        mean_s = tf.reduce_mean(s_post[:,:i+1,:], axis=[1])
        del_relu = del_post[:,i,:]*tf.to_float(mean_s>0)
        op_list.append(del_relu)
    return tf.stack(op_list, axis=1)

def apply_relu_grad_full(del_post, s_post):
    # Assuming input shape : [batch_size, 1, num_iter, N]
    shape = del_post.get_shape()
    b_sz = shape[0]
    n_iter = shape[2]
    op_list = []
    mean_s = tf.reduce_mean(s_post, axis=[2])
    for i in range(n_iter):
        del_relu = del_post[:,:,i,:]*tf.to_float(mean_s>0)
        op_list.append(del_relu)
    return tf.stack(op_list, axis=2)

def forward_pass(X, mask, W, th_val, kernel):
    masked_in = X*mask
    s_in = tf.matmul(masked_in, W)
    v_mem = tf.nn.depthwise_conv2d(s_in, kernel, [1,1,1,1], 'SAME')
    th = tf.Variable(th_val*tf.ones_like(v_mem[:,:,0,:]))
    s_out = thresholding(v_mem, th_val, th)
    return masked_in, s_out

def y_WTA(X, mask, W, th_val, kernel, W_lat, s_it):
    masked_in = X*mask
    s_in = tf.matmul(masked_in, W)
    v_mem = tf.nn.depthwise_conv2d(s_in, kernel, [1,1,1,1], 'SAME')
    th = tf.Variable(th_val*tf.ones_like(v_mem[:,:,0,:]))
    s_out = thresholding(v_mem, th_val, th)
    for i in range(s_it):
        s_inh = tf.matmul(s_out, W_lat)
        s_in = s_in - s_inh
        v_mem = tf.nn.depthwise_conv2d(s_in, kernel, [1,1,1,1], 'SAME')
        s_out = thresholding(v_mem, th_val, th)
    return masked_in, s_out

def finalW_update(s_y, s_pre, label, th_val, kernel):
    err_p = 1.5*th_val*(label - s_y)
    err_m = 1.5*th_val*(s_y - label)
    v_p = tf.nn.depthwise_conv2d(err_p, kernel, [1,1,1,1], 'SAME')
    v_m = tf.nn.depthwise_conv2d(err_m, kernel, [1,1,1,1], 'SAME')
    th = tf.Variable(th_val*tf.ones_like(v_m[:,:,0,:]))
    del_p = thresholding(v_p, th_val, th)
    del_m = thresholding(v_m, th_val, th)
    s_pre_av = tf.reduce_mean(s_pre, axis=[1])
    del_y_av = tf.reduce_mean(del_p - del_m, axis=[1])
    dfinalW = spike_multiply(del_y_av, s_pre_av)
    return del_p, del_m, dfinalW

def innerW_update(s_post, s_pre, del_post_p, del_post_m, W_post, th_val, kernel):
    n_post = tf.to_float(s_post.get_shape()[-1])
    n_pre = tf.to_float(s_pre.get_shape()[-1])
    th_val_b = th_val*n_post/n_pre
    pre_relu_p = tf.matmul(0.5*(del_post_p - del_post_m), tf.transpose(W_post))
    pre_relu_m = tf.matmul(0.5*(del_post_m - del_post_p), tf.transpose(W_post))
    del_p = apply_relu_grad_full(pre_relu_p, s_post)
    del_m = apply_relu_grad_full(pre_relu_m, s_post)
    v_p = tf.nn.depthwise_conv2d(del_p, kernel, [1,1,1,1], 'SAME')
    v_m = tf.nn.depthwise_conv2d(del_m, kernel, [1,1,1,1], 'SAME')
    th = tf.Variable(th_val*tf.ones_like(v_m[:,:,0,:]))
    sdel_p = thresholding(v_p, th_val_b, th)
    sdel_m = thresholding(v_m, th_val_b, th)
    s_pre_av = tf.reduce_mean(s_pre, axis=[1])
    del_av = tf.reduce_mean(sdel_p - sdel_m, axis=[1])
    dW = spike_multiply(del_av, s_pre_av)
    return sdel_p, sdel_m, dW

def gen_kernel(kernel_size, layer_size, tau):
    min_v = 1.0
    max_v = 1.0
    kernel = np.zeros(kernel_size)
    for i in range(kernel_size//2):
        #kernel[i] = 1 - np.exp((i-(k_size//2))/5)
        kernel[i] = (min_v + i*(max_v-min_v)/(kernel_size//2))*(1.0 - np.exp((i-(kernel_size//2))/tau))

    kernels = [kernel for _ in range(layer_size)]
    out_kernel = np.stack(kernels, axis=-1)
    out_kernel = np.expand_dims(out_kernel, axis=0)
    out_kernel = np.expand_dims(out_kernel, axis=3)
    return out_kernel