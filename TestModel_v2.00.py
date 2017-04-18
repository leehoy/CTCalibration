import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt
import os
import sys
import glob
import time
float32=np.float32
pi=np.pi
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

#if os.path.isfile(caffe_root+ 'Exp-SRCNN-1606/Model/SRCNN_v0.02_iter_7.caffemodel'):
#	print 'CaffeNet found.'
#else:
#	print 'Downloading pre-trained CaffeNet model...'
#	os.system('../scripts/download_model_binary.py ../models/bvlc_reference_caffenet')

def vis_square(data):
	"""Take an array of shape (n, height, width) or (n, height, width, 3)
		and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)""" 
	# normalize data for display
	print data.min(),data.max(),data.shape
	data = (data - data.min()) / (data.max() - data.min()) 
	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = (((0, n ** 2 - data.shape[0]),
		(0, 1), (0, 1))                 # add some space between filters
		+ ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
	data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
	# tile the filters into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
	print data.shape
	plt.imshow(data)
	plt.show()
	plt.axis('off')

start_time=time.time()
caffe.set_mode_cpu()
model_def='./Network_Calibration_v2.00_deploy.prototxt'
model_weights ='./Model/Calibration_v2.00_iter_413416.caffemodel'


net=caffe.Net(model_def,model_weights,caffe.TEST)

# load the mean ImageNet image (as distributed with Caffe) for subtraction


# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'in_data': net.blobs['in_data'].data.shape})

#net.blobs['in_data'].reshape(1,        # batch size
#                          1,         # 3-channel (BGR) images
#                          1024,768)  # image size is 227x227
# testing files are 2 to 430 with stride 5
NumberOfProjectionData=10000
Sinogram=range(0,NumberOfProjectionData,3)
kk=0
TestSinogram=Sinogram[kk]
path1='../../tensorflow/data/Calibration/CalibrationPhantomRandomShiftNoiseAdd/'
f=open('../../tensorflow/data/Calibration/CalibrationPhantomRandomShiftNoiseAdd/offset.txt')
line=f.readline()
TrueLabel=np.zeros([7,NumberOfProjectionData],dtype=np.float32)
for i in range(NumberOfProjectionData):
	line=f.readline()
	if not line: break
	tmp=line.strip().split('\\')
	translation=tmp[1]
	angle_offset=tmp[2]
	translation=translation.split(':')[1].strip()
	angle_offset=angle_offset.split(':')[1].strip()
	translation=translation.replace('(','').replace(')','').split()
	trnaslation=[float(t) for t in translation]
	angle_offset=angle_offset.replace('(','').replace(')','').split()
	angle_offset=[float(t)*180/pi for t in angle_offset]
	TrueLabel[0:3,i]=translation
	TrueLabel[3:6,i]=angle_offset
	#if i==TestSinogram:
	#	print(TrueLabel[:,i])
f.close()

total_loss=0
x=0
y=0
nx=1024
ny=768
#image=np.fromfile(path1,dtype=float32).reshape([1,1,nx,ny])
#plt.imshow(image,cmap='gray')
#plt.show()
for i in range(0,len(Sinogram)):
	image=np.fromfile(''.join([path1,'sino_%06d.dat'%Sinogram[i]]),dtype=float32).reshape([1,1,nx,ny])
	gt=np.array(TrueLabel[:,Sinogram[i]],dtype=float32)
	gt=gt.reshape([1,7])
	net.blobs['in_data'].data[...]=image
	net.blobs['labels'].data[...]=gt
	net.forward()
	result=net.blobs['fc04'].data
	loss=net.blobs['loss'].data
	total_loss+=loss
	print loss
#print result
#print gt.reshape([1,7])
print total_loss/len(Sinogram)
end_time=time.time()
print(end_time-start_time,'seconds')
