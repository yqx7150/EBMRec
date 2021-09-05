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
from US_pattern import US_pattern
import h5py

flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_steps', 800, 'num of steps for conditional imagenet sampling')
flags.DEFINE_float('step_lr', 100., 'step size for Langevin dynamics')
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
    return np.clip(im * 256, 0, 255)#.astype(np.uint8)
    
def compare_hfen(rec,ori):
    operation = np.array(io.loadmat("./input_data/loglvbo.mat")['h1'],dtype=np.float32)
    ori = cv2.filter2D(ori.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    rec = cv2.filter2D(rec.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    hfen = np.linalg.norm(ori-rec, ord = 'fro')
    return hfen
    
def write_Data(psnr,ssim,hfen):
    with open(osp.join('./result/compare_ddp/',"psnr_CompareDDP.txt"),"w+") as f:
        f.writelines('['+str(round(psnr, 3))+' '+str(round(ssim, 4))+' '+str(round(hfen, 3))+']')
        f.write('\n')

def write_Data2(step,psnr,ssim,hfen):
    with open(osp.join('./result/compare_ddp/',"psnr_T.txt"),"w+") as f:
        f.writelines('step='+str(step)+' ['+str(round(psnr, 3))+' '+str(round(ssim, 4))+' '+str(round(hfen, 3))+']')
        f.write('\n')
        
def FT (x):   
    #inp: [nx, ny]
    #out: [nx, ny, ns]
    return np.fft.fftshift(np.fft.fft2(sensmaps*np.tile(x[:,:,np.newaxis],[1,1,sensmaps.shape[2]]),axes=(0,1)),axes=(0,1))
    # np.fft.fftshift jiang dipin zhuanyi dao tuxiang zhongxin
    # np.fft.fft2 erwei fuliye

def tFT (x):
    #inp: [nx, ny, ns]
    #out: [nx, ny]
    temp = np.fft.ifft2(np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
    return np.sum( temp*np.conjugate(sensmaps) , axis=2)  / np.sum(sensmaps*np.conjugate(sensmaps),axis=2)


def UFT(x, uspat):
    #inp: [nx, ny], [nx, ny]
    #out: [nx, ny, ns] 
    return np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])*FT(x)

def tUFT(x, uspat):
    #inp: [nx, ny], [nx, ny]
    #out: [nx, ny]
    tmp1 = np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])
    return tFT(tmp1*x)

#load the test image and the coil maps
#-----------------------------------------------------        
#unfortunately, due to complications in saving complex valued data, we save
#and load the complex and real parts seperately
f = h5py.File('./DDP_share_data/acq_im_real.h5', 'r')
kspr = np.array((f['DS1']))
f = h5py.File('./DDP_share_data/acq_im_imag.h5', 'r')
kspi = np.array((f['DS1']))
ksp = np.rot90(np.transpose(kspr+1j*kspi),3)
#get the k-space data
ksp = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(ksp,axes=[0,1]),axes=[0,1]),axes=[0,1])


#again we save and load the complex and real parts seperately for coil maps
f = h5py.File('./DDP_share_data/acq_coilmaps_espirit_real.h5', 'r')
espsr = np.array((f['DS1']))
f = h5py.File('./DDP_share_data/acq_coilmaps_espirit_imag.h5', 'r')
espsi = np.array((f['DS1']))

esps= np.rot90(np.transpose(espsr+1j*espsi),3)
sensmaps = esps.copy()

#rotate images for canonical orientation
sensmaps=np.rot90(np.rot90(sensmaps))
ksp=np.rot90(np.rot90(ksp))
#normalize the espirit coil maps
sensmaps=sensmaps/np.tile(np.sum(sensmaps*np.conjugate(sensmaps),axis=2)[:,:,np.newaxis],[1, 1, sensmaps.shape[2]])

#load the undersampling pattern  original mask
#patt = pickle.load(open('./DDP_share_data/acq_im_us_patt_R2','rb'))
#make the undersampled kspace
#usksp = ksp * np.tile(patt[:,:,np.newaxis],[1, 1, ksp.shape[2]])


orim = tFT(ksp) # the fully sampled image 

# to make the mr image divisible by four
orim_ = np.zeros([216,256],dtype=np.complex128)
orim_[5:210,:] = orim
orim = orim_

sensmaps_ = np.ones([216,256,15],dtype=np.complex128)
sensmaps_[5:210,:,:] = sensmaps
sensmaps = sensmaps_

ksp = FT(orim)


#load the undersampling pattern
USp = US_pattern();
patt = USp.generate_opt_US_pattern_1D([orim.shape[0], orim.shape[1]], R=3, max_iter=100, no_of_training_profs=15)
#misc.imsave('DDP_usp_patt_R3_%s.png'%str((np.sum(patt))/orim.shape[0]/orim.shape[1]),np.abs(patt))
#make the undersampled kspace
usksp = ksp * np.tile(patt[:,:,np.newaxis],[1, 1, ksp.shape[2]])

# normalize the kspace
tmp = tFT(usksp)
usksp=usksp/np.percentile(  np.abs(tmp).flatten()   ,99)


if __name__ == "__main__":

    model = ResNet128(num_filters=64)
    X_NOISE = tf.placeholder(shape=(None, 256, 256, 2), dtype=tf.float32)
    LABEL = tf.placeholder(shape=(None, 1), dtype=tf.float32)
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
    x_last = x_mod - (lr) * (x_grad1 + x_grad2 + x_grad3 + x_grad4 + x_grad5 + x_grad6 + x_grad7 + x_grad8)/8
    
    x_mod = x_last
    x_mod = tf.clip_by_value(x_mod, -1, 1)

    # channel mean
    x_real=x_mod[:,:,:,0]
    x_imag=x_mod[:,:,:,1]
    x_complex = tf.complex(x_real,x_imag)
    x_output  = x_complex

    sess.run(tf.global_variables_initializer())
    saver = loader = tf.train.Saver()
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
    saver.restore(sess, model_file)
#============================================================================================
    write_psnr=0
    write_ssim=0
    write_hfen=9999
    np.random.seed(1)
    #lx = np.random.permutation(1)[:16] #16个0~1000的随机数
    lx = np.random.randint(0, 1, (FLAGS.batch_size))

    ims = []
    PSNR=[]
    im_complex=np.zeros((FLAGS.batch_size,216,256),dtype=np.complex128)

    #=====================================================================================
    
    ori_complex = orim
    ori_complex = ori_complex/np.max(np.abs(ori_complex))
    io.savemat(osp.join('./result/compare_ddp/'+'ori'),{'img':ori_complex}) 

    ksp = FT(ori_complex)
    #mask = patt
    #==================================

    #mask = io.loadmat('./DDP_share_data/mask/cart_mask_R2.mat')['mask']
    mask = io.loadmat('./DDP_share_data/mask/cart_mask_R3.mat')['mask']
    
    #==================================
    #io.savemat(osp.join('./DDP_share_data/'+'cart_mask_R3'),{'mask':mask})
    print(np.sum(mask)/(216*256)) # R=2 (0.5)   R=3 (0.33)
    
    #undersample multi coil kspace
    usksp = ksp * np.tile(mask[:,:,np.newaxis],[1, 1, ksp.shape[2]])
    zero_fiiled = tFT(usksp)
    write_images(abs(zero_fiiled),osp.join('./result/compare_ddp/'+'zero_fiiled'+'.png'))
    io.savemat(osp.join('./result/compare_ddp/'+'zero_fiiled'),{'img':zero_fiiled}) 

    psnr_zerofill = compare_psnr(255*abs(zero_fiiled),255*abs(ori_complex),data_range=255)
    print('psnr_zerofill = ',psnr_zerofill) #26.966455615003746

    # undersample_kspace
    undersample_kspace = usksp


    write_images(abs(ori_complex),osp.join('./result/compare_ddp/'+'ori'+'.png'))

    x_mod = np.random.uniform(-1, 1, size=(FLAGS.batch_size, 256, 256, 2))
    #x_mod[:,:,:,0] = np.real(zero_fiiled)
    #x_mod[:,:,:,1] = np.imag(zero_fiiled)

    labels = np.eye(1)[lx]


    for i in range(FLAGS.num_steps):
        e, im_complex= sess.run([energy_noise,x_output],{X_NOISE:x_mod, LABEL:labels})
        #print(im_complex.shape) (1, 256, 256)
        
        im_complex = np.squeeze(im_complex)
        im_complex = im_complex[0:216,:] # 256-->216

        # data consistance

        iterkspace = undersample_kspace + UFT(im_complex,(1-mask))
        im_complex  = tFT(iterkspace)        
        
        im_back = np.zeros((256,256),dtype=np.complex128)
        im_back[0:216,:] = im_complex
        
        im_back = np.expand_dims(im_back, 0)
        x_mod[:,:,:,0],x_mod[:,:,:,1]=np.real(im_back),np.imag(im_back)
          
        ################################################################################# SSIM
        
        ssim=compare_ssim(abs(im_complex),abs(ori_complex),data_range=1)
        
        if write_ssim<ssim:
            write_ssim=ssim
        ################################################################################# HFEN
        
        hfen=compare_hfen(abs(im_complex),abs(ori_complex))
        
        if write_hfen>hfen:
            write_hfen=hfen
        ################################################################################# PSNR

        err = abs(im_complex) -abs(ori_complex)

        psnr=compare_psnr(255*abs(im_complex),255*abs(ori_complex),data_range=255)
        
        if write_psnr<psnr:
            write_psnr=psnr
            write_images(abs(im_complex),osp.join('./result/compare_ddp/'+'EBMrec_'+'.png'))
            io.savemat(osp.join('./result/compare_ddp/'+'EBMrec_'),{'img':im_complex})
            write_Data2(i,psnr,ssim,hfen)

        print("step:{}".format(i),' PSNR:', psnr,' SSIM:', ssim,' HFEN:', hfen)
        write_Data(write_psnr,write_ssim,write_hfen)
        
