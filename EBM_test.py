from models import ResNet128
import numpy as np
import os.path as osp
from tensorflow.python.platform import flags
import tensorflow as tf
import imageio
import scipy.io as io
import cv2
import matplotlib.pyplot as plt
from utils import optimistic_restore
from skimage.measure import compare_psnr,compare_ssim
import glob
import time

flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_steps', 300, 'num of steps for conditional imagenet sampling')
flags.DEFINE_float('step_lr', 300., 'step size for Langevin dynamics')
flags.DEFINE_integer('batch_size', 1, 'number of steps to run')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('spec_norm', True, 'whether to use spectral normalization in weights in a model')
flags.DEFINE_bool('cclass', True, 'conditional models')
flags.DEFINE_bool('use_attention', False, 'using attention')

FLAGS = flags.FLAGS
def show(image):
    plt.imshow(image,cmap='gray')
    plt.xticks([])
    plt.yticks([])
def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)
def rescale_im(im):
    return np.clip(im * 256, 0, 255)

def compare_hfen(rec,ori):
    operation = np.array(io.loadmat("./input_data/loglvbo.mat")['h1'],dtype=np.float32)
    ori = cv2.filter2D(ori.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    rec = cv2.filter2D(rec.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    hfen = np.linalg.norm(ori-rec, ord = 'fro')
    return hfen

def write_all(result_all):
    filename="result_all_radial70.txt"
    with open(osp.join('./result/siat_rec/radial70/',filename),"w+") as f:
        for k in range(32):
            psnr = result_all[k,0]
            ssim = result_all[k,1]
            hfen = result_all[k,2]
            if k+1 == 32:
                f.writelines('=================ave=====================')
                f.write('\n')
            f.writelines(str(round(psnr, 3))+' '+str(round(ssim, 4))+' '+str(round(hfen, 3)))
            f.write('\n')

   
def write_Data(i,pic,psnr,ssim,hfen):
    filedir="result_psnr"+str(pic)+".txt"
    with open(osp.join('./result/siat_rec/radial70/',filedir),"w+") as f:
        f.writelines('step='+str(i)+' '+'['+str(round(psnr, 3))+' '+str(round(ssim, 5))+' '+str(round(hfen, 3))+']')
        f.write('\n')

def write_zero_Data(psnr,ssim,hfen):
    with open(osp.join('./result/siat_rec/radial70/',"zero_psnr1.txt"),"w+") as f:
        f.writelines('['+str(round(psnr, 3))+' '+str(round(ssim, 5))+' '+str(round(hfen, 3))+']')
        f.write('\n')


if __name__ == "__main__":
    #========================================================================================

    model = ResNet128(num_filters=64)
    X_NOISE = tf.placeholder(shape=(None, 256, 256, 2), dtype=tf.float32)
    LABEL = tf.placeholder(shape=(None, 1), dtype=tf.float32)

    DATA_REAL=tf.placeholder(shape=(256, 256), dtype=tf.float32)
    DATA_IMAG=tf.placeholder(shape=(256, 256), dtype=tf.float32)
    MASK=tf.placeholder(shape=(256, 256),dtype=tf.complex64)

    data=tf.complex(DATA_REAL,DATA_IMAG)
    kdata=tf.fft2d(data)
    ksample=tf.multiply(MASK,kdata)
    sdata=tf.ifft2d(ksample)
    sess = tf.InteractiveSession()

    # Langevin dynamics algorithm
    weights = model.construct_weights("context_0")
    x_mod = X_NOISE
    x_mod1 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=0.005)
    x_mod2 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=0.005)
    x_mod3 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=0.005)
    x_mod4 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=0.005)
    x_mod5 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=0.005)
    x_mod6 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=0.005)
    x_mod7 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=0.005)
    x_mod8 = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=0.005)  
    
    energy_noise1 = energy_start = model.forward(x_mod1, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad1 = tf.gradients(energy_noise1, [x_mod1])[0] 
    energy_noise2 = energy_start = model.forward(x_mod2, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad2 = tf.gradients(energy_noise2, [x_mod2])[0]  
    energy_noise3 = energy_start = model.forward(x_mod3, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad3 = tf.gradients(energy_noise3, [x_mod3])[0] 
    energy_noise4 = energy_start = model.forward(x_mod4, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad4 = tf.gradients(energy_noise4, [x_mod4])[0]   
    energy_noise5 = energy_start = model.forward(x_mod5, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad5 = tf.gradients(energy_noise5, [x_mod5])[0]   
    energy_noise6 = energy_start = model.forward(x_mod6, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad6 = tf.gradients(energy_noise6, [x_mod6])[0]  
    energy_noise7 = energy_start = model.forward(x_mod7, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad7 = tf.gradients(energy_noise7, [x_mod7])[0] 
    energy_noise8 = energy_start = model.forward(x_mod8, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad8 = tf.gradients(energy_noise8, [x_mod8])[0]

    energy_noise_old = energy_noise1
    energy_noise = energy_noise1
    
    lr = FLAGS.step_lr
    x_last = x_mod - (lr) * (x_grad1 + x_grad2 + x_grad3 + x_grad4 + x_grad5 +x_grad6 +x_grad7 +x_grad8)/8

    x_mod = x_last
    x_mod = tf.clip_by_value(x_mod, -0.3, 1)

    # channel mean
    x_real=x_mod[:,:,:,0]
    x_imag=x_mod[:,:,:,1]
    x_complex = tf.complex(x_real,x_imag)

    # data consistance
    iterkspace = tf.fft2d(x_complex)
    iterkspace = ksample + iterkspace*(1-MASK)

    x_output  = tf.ifft2d(iterkspace)

    sess.run(tf.global_variables_initializer())
    saver = loader = tf.train.Saver()
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
    saver.restore(sess, model_file)

#============================================================================================
    result_all = np.zeros((32,3),dtype=np.float64)
    start_outout = time.time()
    for pp in range(31):
        write_psnr=0
        write_ssim=0
        write_hfen=9999
        np.random.seed(1)

        lx = np.random.randint(0, 1, (FLAGS.batch_size))

        ims = []
        PSNR=[]
        im_complex=np.zeros((FLAGS.batch_size,256,256),dtype=np.complex64)
        im_best=np.zeros((256,256),dtype=np.complex64)
        
        pic= pp + 1
        print('picture:',pic)
        if pic<10:
            file_path='./input_data/SIAT_test_image31/test_data_0'+str(pic)+'.mat'
        else:
            file_path='./input_data/SIAT_test_image31/test_data_'+str(pic)+'.mat'
            
        ori_data = np.zeros([256,256],dtype=np.complex64)
        ori_data = io.loadmat(file_path)['Img']

        ori_data=ori_data/np.max(np.abs(ori_data))

        data_real=ori_data.real
        data_imag=ori_data.imag

        data_real = data_real.astype(np.float32)
        data_imag = data_imag.astype(np.float32)



        mask = io.loadmat('./input_data/mask/mask_radial70.mat')['mask_radial70']
        #mask=np.fft.fftshift(mask)
        #plt.imshow(mask,cmap='gray')
        #plt.show()
        
        ori_Image=data_real+1j*data_imag
        write_images(abs(ori_data),osp.join('./result/siat_rec/radial70/Image/'+'ori_Image'+str(pic)+'.png'))
        
        Kdata=np.fft.fft2(ori_Image)
        Ksample=np.multiply(mask,Kdata)
        zeorfilled_data=np.fft.ifft2(Ksample)
        psnr_zero=compare_psnr(255*abs(zeorfilled_data),255*abs(ori_Image),data_range=255)
        ssim_zero=compare_ssim(abs(zeorfilled_data),abs(ori_Image),data_range=1)
        hfen_zero=compare_hfen(abs(zeorfilled_data),abs(ori_Image))
        write_zero_Data(psnr_zero,ssim_zero,hfen_zero)

        x_mod = np.random.uniform(-1, 1, size=(FLAGS.batch_size, 256, 256, 2))
        labels = np.eye(1)[lx]

        start_out = time.time()
        
        for i in range(FLAGS.num_steps):
            start_in = time.time()
            e, im_complex= sess.run([energy_noise,x_output],{X_NOISE:x_mod, LABEL:labels, DATA_REAL:data_real, DATA_IMAG:data_imag, MASK:mask})
            

            ################################################################################ PSNR
            for k in range(FLAGS.batch_size):
                psnr=compare_psnr(255*abs(im_complex[k]),255*abs(ori_Image),data_range=255)
                PSNR.append(psnr)
            psnr=max(PSNR)
            id=PSNR.index(max(PSNR))
            PSNR=[]

            x_mod[:,:,:,0],x_mod[:,:,:,1]=np.real(im_complex),np.imag(im_complex)
            im_rec=x_mod[:,:,:,0]+1j*x_mod[:,:,:,1]

            end_in = time.time()
            print("%.2f s"%(end_in-start_in))
            ################################################################################# SSIM
            ssim=compare_ssim(abs(im_complex[id]),abs(ori_Image),data_range=1)
            
            if write_ssim<ssim:
                write_ssim=ssim
                result_all[pp,1] =ssim
            ################################################################################# HFEN
            hfen=compare_hfen(abs(im_complex[id]),abs(ori_Image))
            
            if write_hfen>hfen:
                write_hfen=hfen
                result_all[pp,2] =hfen
            ################################################################################ PSNR
            if write_psnr<psnr:
                write_psnr=psnr
                result_all[pp,0] =psnr
                write_images(abs(im_complex[id]),osp.join('./result/siat_rec/radial70/Image/'+'EBM_rec_'+str(pic)+'.png'))
                io.savemat(osp.join('./result/siat_rec/radial70/Image/'+'EBM_rec_'+str(pic)),{'img':im_complex[id]})
                
                write_Data(i,pic,write_psnr,ssim,hfen)
                               
            print("step:{}".format(i),' id:',id,' PSNR:', psnr,' SSIM:', ssim,' HFEN:', hfen)
            
        end_out = time.time()
        print('================',str(pic),'========================')
        print("Time to rebuild a MR image:%.2f s"%(end_out-start_out))
        print('====================================================')
    
    end_outout = time.time()
    print("Time to rebuild the 31 MR image:%.2f s"%(end_outout-start_outout))
    print("Ave time:%.2f s"%((end_outout-start_outout)/31))
    result_all[31,0] = sum(result_all[:31,0])/31
    result_all[31,1] = sum(result_all[:31,1])/31
    result_all[31,2] = sum(result_all[:31,2])/31
    write_all(result_all)      

        
