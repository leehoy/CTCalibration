import lmdb
import numpy as np
#import cv2
import caffe
from caffe.proto import caffe_pb2
import glob
import os
import shutil
import matplotlib.pyplot as plt
#import random
float32=np.float32
pi=np.pi

NumberOfCenterProjection=10000
NumberOfTruncatedProjection=5000
ny=768
nx=1024
GroundParameters_center=np.zeros([7,NumberOfCenterProjection],dtype=np.float32)
GroundParameters_trun=np.zeros([7,NumberOfTruncatedProjection],dtype=np.float32)

offset_file_center='../../tensorflow/data/Calibration/CalibrationPhantomRandomShiftNoiseAdd/offset.txt'
f=open(offset_file_center)
#index=0
label_index=0
line=f.readline()
for i in range(NumberOfCenterProjection):
	line=f.readline()
	if not line: break
	tmp=line.strip().split('\\')
	translation=tmp[1]
	angle_offset=tmp[2]
	#ev_offset=tmp[2]
	translation=translation.split(':')[1].strip()
	angle_offset=angle_offset.split(':')[1].strip()
	#ev_offset=ev_offset.split(':')[1].strip()
	translation=translation.replace('(','').replace(')','').split()
	translation=[float(t) for t in translation]
	angle_offset=angle_offset.replace('(','').replace(')','').split()
	angle_offset=[float(t)*180/pi for t in angle_offset]
	#ev_offset=ev_offset.replace('(','').replace(')','').split()
	#ev_offset=[float(t) for t in ev_offset]
	#print translation , eu_offset, ev_offset
	#print GroundParameters[:,index]
	GroundParameters_center[0:3,label_index]=translation
	GroundParameters_center[3:6,label_index]=angle_offset
	label_index+=1
	#GroundParameters[5:7,i]=ev_offset
	#index+=1
f.close()
label_index=0
offset_file_truncated='../../tensorflow/data/Calibration/CalibrationPhantomRandomShiftTruncationNoiseAdded/offset.txt'
f=open(offset_file_truncated)
#index=0
line=f.readline()
for i in range(NumberOfTruncatedProjection):
	line=f.readline()
	if not line: break
	tmp=line.strip().split('\\')
	translation=tmp[1]
	angle_offset=tmp[2]
	#ev_offset=tmp[2]
	translation=translation.split(':')[1].strip()
	angle_offset=angle_offset.split(':')[1].strip()
	#ev_offset=ev_offset.split(':')[1].strip()
	translation=translation.replace('(','').replace(')','').split()
	translation=[float(t) for t in translation]
	angle_offset=angle_offset.replace('(','').replace(')','').split()
	angle_offset=[float(t)*180/pi for t in angle_offset]
	#ev_offset=ev_offset.replace('(','').replace(')','').split()
	#ev_offset=[float(t) for t in ev_offset]
	#print translation , eu_offset, ev_offset
	#print GroundParameters[:,index]
	GroundParameters_trun[0:3,label_index]=translation
	GroundParameters_trun[3:6,label_index]=angle_offset
	label_index+=1
	#GroundParameters[5:7,i]=ev_offset
	#index+=1
f.close()
path_center='../../tensorflow/data/Calibration/CalibrationPhantomRandomShiftNoiseAdd/'
path_truncated='../../tensorflow/data/Calibration/CalibrationPhantomRandomShiftTruncationNoiseAdded/'
lmdb_file='../../tensorflow/data/CalibrationProjection_lmdb/'
if os.path.isdir(lmdb_file):
	shutil.rmtree(lmdb_file)


CenterProjectionNumber=range(0,NumberOfCenterProjection,3)
TrunProjectionNumber=range(0,NumberOfTruncatedProjection,3)
#print(len(sliceNumber),len(patchNumber))
filepath=''.join([path_center,'sino_%06d.dat'%(CenterProjectionNumber[0])])
print filepath
#filepath=''.join([path1,'%04d/%06d.dat'%(sliceNumber[0],patchNumber[19])])
s=np.fromfile(filepath,dtype=float32).reshape([1,nx,ny])
print s.shape,s.nbytes
map_size_projection=s.nbytes*((len(CenterProjectionNumber)+len(TrunProjectionNumber))*100)
print map_size_projection
c=0
env=lmdb.open(lmdb_file,map_size=map_size_projection)
with env.begin(write=True) as txn:
	for i in range(len(CenterProjectionNumber)):
		filepath=''.join([path_center,'sino_%06d.dat'%(CenterProjectionNumber[i])])
		s=np.fromfile(filepath,dtype=float32)
		s=s.reshape([1,nx,ny])
		datum=caffe.io.array_to_datum(s.astype(float))
		str_id='{:08}'.format(c)
		txn.put(str_id,datum.SerializeToString())
		#print c
		c+=1
	for i in range(len(TrunProjectionNumber)):
		filepath=''.join([path_truncated,'sino_%06d.dat'%(TrunProjectionNumber[i])])
		s=np.fromfile(filepath,dtype=float32)
		s=s.reshape([1,nx,ny])
		datum=caffe.io.array_to_datum(s.astype(float))
		str_id='{:08}'.format(c)
		txn.put(str_id,datum.SerializeToString())
		#print c
		c+=1
#print c
env.close()
label_lmdb='../../tensorflow/data/CalibrationLabel_lmdb/'
if os.path.isdir(label_lmdb):
	shutil.rmtree(label_lmdb)
#map_size_label=np.array(GroundParameters_center[:,0],dtype=float32).nbytes*((len(CenterProjectionNumber)+len(TrunProjectionNumber))*100)
#print map_size
c=0
env=lmdb.open(label_lmdb)
with env.begin(write=True) as txn:
	for i in range(len(CenterProjectionNumber)):
		label=np.array(GroundParameters_center[:,CenterProjectionNumber[i]],dtype=float32).reshape([1,7,1])

		datum=caffe.io.array_to_datum(label.astype(float))
		str_id='{:08}'.format(c)
		txn.put(str_id,datum.SerializeToString())
		c+=1
	for i in range(len(TrunProjectionNumber)):
		label=np.array(GroundParameters_trun[:,TrunProjectionNumber[i]],dtype=float32).reshape([1,7,1])
		datum=caffe.io.array_to_datum(label.astype(float))
		str_id='{:08}'.format(c)
		txn.put(str_id,datum.SerializeToString())
		c+=1
env.close()

lmdb_file='../../tensorflow/data/CalibrationProjection_Test_lmdb/'
if os.path.isdir(lmdb_file):
	shutil.rmtree(lmdb_file)
#path1='../data/SinoPatchWholeBody/SinoPatch170WholeBody/Test/'
CenterProjectionNumber_test=range(1,NumberOfCenterProjection,3)
TrunProjectionNumber_test=range(1,NumberOfTruncatedProjection,3)
#PatchNumber=range(18447)
#batch_size=len(sliceNumber)
#patch_size=len(PatchNumber)
#TestSize=10
#s=np.fromfile(''.join([path1,'0190/000000.dat']),dtype=float32)
filepath=''.join([path_center,'sino_%06d.dat'%(CenterProjectionNumber_test[0])])
s=np.fromfile(filepath,dtype=float32).reshape([1,nx,ny])
map_size=s.nbytes*(len(CenterProjectionNumber_test)+len(TrunProjectionNumber_test))*100
env=lmdb.open(lmdb_file,map_size=map_size_projection)
c=0
with env.begin(write=True) as txn:
	for i in range(len(CenterProjectionNumber_test)):
		filepath=''.join([path_center,'sino_%06d.dat'%(CenterProjectionNumber_test[i])])
		s=np.fromfile(filepath,dtype=float32)
		s=s.reshape([1,nx,ny])
		datum=caffe.io.array_to_datum(s.astype(float))
		str_id='{:08}'.format(c)
		txn.put(str_id,datum.SerializeToString())
		#print c
		c+=1
	for i in range(len(TrunProjectionNumber_test)):
		filepath=''.join([path_truncated,'sino_%06d.dat'%(TrunProjectionNumber_test[i])])
		s=np.fromfile(filepath,dtype=float32)
		s=s.reshape([1,nx,ny])
		datum=caffe.io.array_to_datum(s.astype(float))
		str_id='{:08}'.format(c)
		txn.put(str_id,datum.SerializeToString())
		#print c
		c+=1
env.close()
label_lmdb='../../tensorflow/data/CalibrationLabel_Test_lmdb/'
if os.path.isdir(label_lmdb):
	shutil.rmtree(label_lmdb)
c=0
env=lmdb.open(label_lmdb)
with env.begin(write=True) as txn:
	for i in range(len(CenterProjectionNumber_test)):
		label=np.array(GroundParameters_center[:,CenterProjectionNumber_test[i]],dtype=float32).reshape([1,7,1])
		datum=caffe.io.array_to_datum(np.array(label).astype(float))
		str_id='{:08}'.format(c)
		txn.put(str_id,datum.SerializeToString())
		c+=1
	for i in range(len(TrunProjectionNumber_test)):
		label=np.array(GroundParameters_trun[:,TrunProjectionNumber_test[i]],dtype=float32).reshape([1,7,1])
		datum=caffe.io.array_to_datum(np.array(label).astype(float))
		str_id='{:08}'.format(c)
		txn.put(str_id,datum.SerializeToString())
		c+=1
#print c
env.close()
