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
import h5py as h5

flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_steps', 800, 'num of steps for conditional imagenet sampling')
flags.DEFINE_float('step_lr', 10., 'step size for Langevin dynamics')
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
    with open(osp.join('./result/compare_modl/',"psnr_compare_modl.txt"),"w+") as f:
        f.writelines('['+str(round(psnr, 3))+' '+str(round(ssim, 4))+' '+str(round(hfen, 3))+']')
        f.write('\n')

def write_Data2(step,psnr,ssim,hfen):
    with open(osp.join('./result/compare_modl/',"psnr_T.txt"),"w+") as f:
        f.writelines('step='+str(step)+' ['+str(round(psnr, 3))+' '+str(round(ssim, 4))+' '+str(round(hfen, 3))+']')
        f.write('\n')

def FT(x,csm):
    """ This is a the A operator as defined in the paper"""
    ncoil,nrow,ncol = csm.shape
    ccImg=np.reshape(x,(nrow,ncol) )
    coilImages=np.tile(ccImg,[ncoil,1,1])*csm;
    kspace=np.fft.fft2(coilImages)/np.sqrt(nrow * ncol)
    return kspace

def tFT(kspaceUnder,csm):
    """ This is a the A^T operator as defined in the paper"""
    ncoil,nrow,ncol = csm.shape
    #temp=np.zeros((ncoil,nrow,ncol),dtype=np.complex64)
    img=np.fft.ifft2(kspaceUnder)*np.sqrt(nrow*ncol)
    coilComb=np.sum(img*np.conj(csm),axis=0).astype(np.complex64)
    #coilComb=coilComb.ravel();
    return coilComb

 
filename='./MoDL_share_data/demoImage.hdf5' #set the correct path here

with h5.File(filename,'r') as f:
    org,csm,mask=f['tstOrg'][:],f['tstCsm'][:],f['tstMask'][:]

#print(org.shape,csm.shape,mask.shape)
#(1, 256, 232) (1, 12, 256, 232) (1, 256, 232)
orim = org[0]
csm = csm[0]
patt = mask[0]

if __name__ == "__main__":

    model = ResNet128(num_filters=64)
    #model = ResNet32Large(num_filters=128)
    X_NOISE = tf.placeholder(shape=(None, 256, 232, 2), dtype=tf.float32)
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
    lx = np.random.randint(0, 1, (FLAGS.batch_size))

    ims = []
    PSNR=[]
    im_complex=np.zeros((FLAGS.batch_size,256,232),dtype=np.complex128)

    #==========================================================================
    ori_complex = orim
    ori_complex = ori_complex/np.max(np.abs(ori_complex))
    write_images(abs(ori_complex),osp.join('./result/compare_modl/'+'ori'+'.png'))
    io.savemat(osp.join('./result/compare_modl/'+'ori'),{'img':ori_complex})

    mask = patt #0.1666217672413793# R=6
    io.savemat(osp.join('./MoDL_share_data/'+'random_mask_R6'),{'mask':mask})
    print('====================')
    print(np.sum(mask)/(256*232))
    ksp = FT(ori_complex,csm)  

    if len(mask.shape)==2:
        mask=np.tile(mask,(csm.shape[0],1,1))  
 
    #get multi coil undersample kspace by mask
    usksp = np.multiply(ksp,mask)
    undersample_kspace = usksp
    zero_fiiled = tFT(usksp,csm)
    write_images(abs(zero_fiiled),osp.join('./result/compare_modl/'+'zero_fiiled'+'.png'))
    io.savemat(osp.join('./result/compare_modl/'+'zero_fiiled'),{'img':zero_fiiled})

    # use for getting degrade img and psnr,ssim,hfen in iteration
    psnr_zerofill = compare_psnr(255*abs(zero_fiiled),255*abs(ori_complex),data_range=255)
    print('psnr_zerofill = ',psnr_zerofill) #25.95079970708028    
    
    x_mod = np.random.uniform(-1, 1, size=(FLAGS.batch_size, 256, 232, 2))
    labels = np.eye(1)[lx]

    for i in range(FLAGS.num_steps):
        e, im_complex= sess.run([energy_noise,x_output],{X_NOISE:x_mod, LABEL:labels})  
        im_complex = np.squeeze(im_complex)
        
        # data consistance 
        iterkspace = undersample_kspace + FT(im_complex,csm)*(1-mask)
        im_complex  = tFT(iterkspace,csm)

        #temp_complex = np.zeros((1,256,232),dtype=np.complex64)
        #temp_complex[0,:,:] = im_complex
        im_complex = np.expand_dims(im_complex, 0)

        x_mod[:,:,:,0],x_mod[:,:,:,1]=np.real(im_complex),np.imag(im_complex)
        im_rec=x_mod[:,:,:,0]+1j*x_mod[:,:,:,1]
        im_complex = im_complex[0]
        #im_complex=im_complex/np.max(abs(im_complex))
        #print(np.max(abs(im_complex)),np.min(abs(im_complex)))

        ################################################################################# SSIM
   
        ssim=compare_ssim(abs(im_complex),abs(ori_complex),data_range=1)
        
        if write_ssim<ssim:
            write_ssim=ssim
        ################################################################################# HFEN
        
        hfen=compare_hfen(abs(im_complex),abs(ori_complex))
        
        if write_hfen>hfen:
            write_hfen=hfen
        ################################################################################# PSNR
        
        psnr=compare_psnr(255*abs(im_complex),255*abs(ori_complex),data_range=255)
        err = abs(im_complex) -abs(ori_complex)

        if write_psnr<psnr:
            write_psnr=psnr
            write_Data2(i,psnr,ssim,hfen)
            write_images(abs(im_complex),osp.join('./result/compare_modl/'+'EBMrec_'+'.png'))
            write_images(abs(err)*5,osp.join('./result/compare_modl/'+'erro_CompareMmodl'+'.png'))
            io.savemat(osp.join('./result/compare_modl/'+'EBM_rec'),{'img':im_complex})
                                 
        print("step:{}".format(i),' PSNR:', psnr,' SSIM:', ssim,' HFEN:', hfen)
        write_Data(write_psnr,write_ssim,write_hfen)
    
        
