
# coding: utf-8

# In[1]:


import tensorflow as tf
import itertools
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tensorflow.python.lib.io import file_io
import argparse

# In[2]:


def generator(z, isTraining=True):
    with tf.variable_scope('generator'):
        
        gz1=tf.layers.conv2d_transpose(inputs=z,filters=64, kernel_size=[4,4], strides=(1,1), padding='valid')
        ga1= tf.nn.relu(tf.layers.batch_normalization(gz1, training=isTraining))
        gz2=tf.layers.conv2d_transpose(inputs=ga1,filters=64, kernel_size=[4,4], strides=(2,2), padding='same')
        ga2= tf.nn.relu(tf.layers.batch_normalization(gz2, training=isTraining))
        gz3=tf.layers.conv2d_transpose(inputs=ga2,filters=32, kernel_size=[4,4], strides=(2,2), padding='same')
        ga3= tf.nn.relu(tf.layers.batch_normalization(gz3, training=isTraining))
        gz4=tf.layers.conv2d_transpose(inputs=ga3,filters=32, kernel_size=[4,4], strides=(2,2), padding='same')
        ga4= tf.nn.relu(tf.layers.batch_normalization(gz4, training=isTraining))
        gz5=tf.layers.conv2d_transpose(inputs=ga4,filters=1, kernel_size=[4,4], strides=(2,2), padding='same')
        ga5=tf.nn.tanh(gz5)
        output= ga5
        return output


# In[3]:


def discriminator(x, isTraining=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        
        dz1=tf.layers.conv2d(inputs=x,filters=16, kernel_size=[4,4], strides=(2,2), padding='same')
        da1= tf.nn.leaky_relu(dz1,.2)
        dz2=tf.layers.conv2d(inputs=da1,filters=32, kernel_size=[4,4], strides=(2,2), padding='same')
        da2= tf.nn.leaky_relu(tf.layers.batch_normalization(dz2, training=isTraining),.2)
        dz3=tf.layers.conv2d(inputs=da2,filters=64, kernel_size=[4,4], strides=(2,2), padding='same')
        da3= tf.nn.leaky_relu(tf.layers.batch_normalization(dz3, training=isTraining),.2)
        dz4=tf.layers.conv2d(inputs=da3,filters=64, kernel_size=[4,4], strides=(2,2), padding='same')
        da4= tf.nn.leaky_relu(tf.layers.batch_normalization(dz4, training=isTraining),.2)
        dz5=tf.layers.conv2d(inputs=da4,filters=1, kernel_size=[4,4], strides=(1,1), padding='valid')
        da5=tf.nn.sigmoid(dz5)
        output=da5
        return output, dz5
    




# In[5]:
def main(job_dir,**args):

    with tf.device('/device:GPU:0'):
        batch_size = 100
        lr = 0.0002
        epochs = 10
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
        x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
        z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
        isTraining = tf.placeholder(dtype=tf.bool)
        G_z=generator(z, isTraining)
        realD, real_logits=discriminator(x,isTraining)
        fakeD, fake_logits=discriminator(G_z,isTraining, True)
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

        #trainable params
        T_vars = tf.trainable_variables()
        D_vars = [var for var in T_vars if 'discriminator' in var.name]
        G_vars = [var for var in T_vars if 'generator' in var.name]
        #we get all vars and then update all functions like relu before calculating loss after each training
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
            G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, 
                                            log_device_placement=True))
        tf.global_variables_initializer().run()

        #summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
        train_set = (train_set - 0.5) / 0.5  #-1 to 1 normalize

        root = job_dir+'MNIST_DCGAN_results/'
        model = 'MNIST_DCGAN_'
        
        saver = tf.train.Saver()

        # In[7]:
    
    
        for epoch in range(epochs):
            G_losses = []
            D_losses = []
            #epoch_start_time = time.time()
            for iter in range(mnist.train.num_examples // batch_size):
                # update discriminator
                x_ = train_set[iter*batch_size:(iter+1)*batch_size]
                z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

                loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTraining: True})
                D_losses.append(loss_d_)

                # update generator
                z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
                loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTraining: True})
                G_losses.append(loss_g_)

            
            saver.save(sess, './modelDCGAN'+str(epoch)+'.ckpt')
            with file_io.FileIO('./modelDCGAN'+str(epoch)+'.ckpt.data-00000-of-00001', mode='rb') as input_f :
                with file_io.FileIO(job_dir+'modelDCGAN'+str(epoch)+'.ckpt.data-00000-of-00001', mode='w+') as output_f:
                    output_f.write(input_f.read())
            with file_io.FileIO('./modelDCGAN'+str(epoch)+'.ckpt.index', mode='rb') as input_f :
                with file_io.FileIO(job_dir+'modelDCGAN'+str(epoch)+'.ckpt.index', mode='w+') as output_f:
                    output_f.write(input_f.read())
            with file_io.FileIO('./modelDCGAN'+str(epoch)+'.ckpt.meta', mode='rb') as input_f :
                with file_io.FileIO(job_dir+'modelDCGAN'+str(epoch)+'.ckpt.meta', mode='w+') as output_f:
                    output_f.write(input_f.read())
            if epoch==9 :
                fixedZ = np.random.normal(0, 1, (25, 1, 1, 100))
                test_images = sess.run(G_z, {z: fixedZ, isTraining: False})
                k=0
                for im in test_images:
                    im=np.reshape(im, (64, 64))
                    k=k+1
                    
                    fixed_p = './'+model + str(k) + str(epoch + 1) + '.png'
                    plt.imsave(fixed_p, im, cmap='gray')
                    with file_io.FileIO(fixed_p, mode='rb') as input_f :
                        with file_io.FileIO(job_dir+'result/imgepoch'+ str(epoch+1)+str(k)+ '.png' , mode='w+') as output_f:
                            output_f.write(input_f.read())
        
        sess.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
