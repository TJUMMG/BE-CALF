import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import cv2
import glob
import tensorflow as tf
import sys
import time
from tqdm import tqdm
from becalf import becalf


def normalize(images):
    return np.array([image/65535.0 for image in images])
    

def downscale(images):
    #print images
    downs = [[[[0 for p in range(3)] for k in range(1024)] for j in range(436)] for i in range(len(images))]
    for ii in range(len(images)):
        for j in range(len(images[ii])):
            for k in range(len(images[ii][j])):
                for p in range(len(images[ii][j][k])):
                    tmp = bin(images[ii][j][k][p])
                    tmp_quan = tmp[:-10]+'0000000000'
                    downs[ii][j][k][p] = int (tmp_quan,2)
    downs_np = np.array(downs, dtype=np.uint16)
    return downs_np
    

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True) 
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True 

x = tf.placeholder(tf.float32, [None,436, 1024, 3])
downscaled = tf.placeholder(tf.float32, [None,436, 1024, 3])
is_training = tf.placeholder(tf.bool, [])

model = becalf(x, downscaled, is_training, 1)
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, './latest')

#pic = './677.png'
pics = glob.glob('../Source50/*')

tt=0
for i in tqdm(range(len(pics))):#
#for i in range(1):#len(pics)):#
    x_t1 = cv2.imread(pics[i],3)
    #x_t1 = cv2.imread(pic,3)
    x_t1n = x_t1[np.newaxis,:,:,:]

    downs = downscale(x_t1n)
    
    cv2.imwrite('tmp.png',downs[0])

    starttime=time.time()
    x_t1 = cv2.imread('tmp.png',3)
    downs = x_t1[np.newaxis,:,:,:]
    

    downs_f = normalize(downs)
    
    fake = sess.run(model.imitation,
        feed_dict={x: downs_f, downscaled: downs_f, is_training: False})

    adda = fake + downs_f


    full_name = pics[i].split('/')[-1]
    #full_name = pic.split('/')[-1]
    pure_name = full_name.split('.')[0]
    clipped = np.clip(adda[0],0,1)
    im = np.uint16(clipped*65535.0)
    
    #cv2.imwrite(pure_name+'_dsp_416.png',im)
    cv2.imwrite('./results_616/'+pure_name+'_becalf_616.png',im)

    print time.time()-starttime


        
        