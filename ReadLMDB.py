import numpy as np
import lmdb
import caffe
import matplotlib.pyplot as plt

ttt1=np.zeros([10,1024,768])
env=lmdb.open('../../tensorflow/data/CalibrationProjection_lmdb',readonly=True)
k=0
with env.begin() as txn:
	cursor=txn.cursor()
	for key,value in cursor:
		if(k>=10):
			break
#		print(key)
		raw_datum=txn.get(key)
		datum=caffe.proto.caffe_pb2.Datum()
		datum.ParseFromString(value)
		x=caffe.io.datum_to_array(datum)
#		x=x.reshape([datum.cahnnel,datum.height,datum.width])
#		flat_x=np.fromstring(datum.data,dtype=np.float32)
#		x=flat_x.reshape(datum.channels,datum.height,datum.width)
#		y=datum.label
#		print y
#		print x.shape
		#x=x.transpose()
		ttt1[k,:,:]=x[0,:,:]
		k+=1
#		print x.shape
kk=0
plt.imshow(ttt1[kk,:,:,],cmap='gray')
plt.show()
env=lmdb.open('../../tensorflow/data/CalibrationLabel_lmdb',readonly=True)
k=0
ttt2=np.zeros([10,7,1])
with env.begin() as txn:
	cursor=txn.cursor()
	for key,value in cursor:
#		print(key)
		if k>=10:
			break
		raw_datum=txn.get(key)
		datum=caffe.proto.caffe_pb2.Datum()
		datum.ParseFromString(value)
		x=caffe.io.datum_to_array(datum)
		#print x.shape
#		x=x.reshape([datum.channel,datum.height,datum.width])
#		flat_x=np.fromstring(datum.data,dtype=np.float32)
#		x=flat_x.reshape(datum.channels,datum.height,datum.width)
#		y=datum.label
#		print y
#		print x.shape
		#x=x.transpose()
		ttt2[k,:,:]=x[0,:,:]
		k+=1
#		print x.shape
#kk=40
#plt.imshow(ttt1[kk,:,:],cmap='gray')
#plt.show()
print(ttt2[kk,:,:])


