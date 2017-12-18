from __future__ import division
import numpy as np
import cv2, sys, os, urllib2, io, binascii
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from PIL import Image
import PIL.Image
from tensorflow.python.ops import control_flow_ops
from Tkinter import *

def show_images(images, cols = 1, titles = None):
  assert((titles is None)or (len(images) == len(titles)))
  n_images = len(images)
  if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
  fig = plt.figure()
  for n, (image, title) in enumerate(zip(images, titles)):
      a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
      if image.ndim == 2:
          plt.gray()
      plt.imshow(image)
      a.set_title(title)
  fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
  figManager = plt.get_current_fig_manager()
  figManager.resize(*figManager.window.maxsize())
  plt.show(block=False)
  time.sleep(1)
  plt.close()

def call(vidFile):
  cap = cv2.VideoCapture(vidFile)
  fourcc_orig = cv2.cv.CV_FOURCC(*'XVID')
  writer_orig = None
  (h, w) = (None, None)
  cnt = 0
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
  checkpoints_dir = "/home/va/Documents/Schools/Columbia/Courses/Sem1/COMS_4771_ML/git/checkpoints"
  image_size = vgg.vgg_16.default_image_size
  fourcc = cv2.cv.CV_FOURCC(*'XVID')
  #print vidFile[:-4] + "_processed" + vidFile[-4:]
  out = cv2.VideoWriter(vidFile[:-4] + "_processed" + vidFile[-4:],fourcc, 20.0, (image_size, image_size))
  names = imagenet.create_readable_names_for_imagenet_labels()
  list1 = []
  list2 = []
  while(cap.isOpened()):
    ret, frame_orig = cap.read()
    cnt = cnt + 1
    if ret == True:
      if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)
      slim = tf.contrib.slim
      with tf.Graph().as_default():
        frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(frame, dtype=tf.int32)
        processed_image = vgg_preprocessing.preprocess_image(frame, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)
        with slim.arg_scope(vgg.vgg_arg_scope()):
          logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
        probabilities = tf.nn.softmax(logits)
        init_fn = slim.assign_from_checkpoint_fn(
          os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
          slim.get_model_variables('vgg_16'))
        
        with tf.Session() as sess:
          init_fn(sess)
          np_image, network_input, probabilities = sess.run([image, processed_image, probabilities])
          probabilities = probabilities[0, 0:]
          sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
        #show_images([frame, network_input.astype(np.uint8)], titles = ["Frame" + str(cnt), "Resized, Cropped and Mean-Centered input to network"])
        op = network_input.astype(np.uint8)
        if writer_orig is None:
          (h, w) = frame_orig.shape[:2]
          #print vidFile[:-4] + "_original" + vidFile[-4:]
          writer_orig = cv2.VideoWriter(vidFile[:-4] + "_original" + vidFile[-4:], fourcc_orig, 20, (w, h)) 
        writer_orig.write(frame_orig)
        out.write(op)
        print "Frame" + str(cnt)
        for i in range(5):
          index = sorted_inds[i]
          print('%0.2f => [%s]' % (probabilities[index], names[index+1]))
        res = slim.get_model_variables()
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
      sys.stdout.flush()
    else: 
      break
    '''
    if(cnt >= 60):
      break
    '''
    print ""
  cap.release()
  out.release()
  writer_orig.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = '0'
  sys.path.append("/home/va/Documents/Schools/Columbia/Courses/Sem1/COMS_4771_ML/git/models/research/slim")
  from datasets import imagenet
  from datasets import dataset_utils
  from nets import vgg
  from preprocessing import vgg_preprocessing
  vidFile=sys.argv[1]
  call(vidFile)
