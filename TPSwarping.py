import numpy as np
import cv2

def WarpImage_TPS(source,target,img):
	tps = cv2.createThinPlateSplineShapeTransformer()

	source=source.reshape(-1,len(source),2)
	target=target.reshape(-1,len(target),2)

	matches=list()
	for i in range(0,len(source[0])):

		matches.append(cv2.DMatch(i,i,0))

	tps.estimateTransformation(target, source, matches)  # note it is target --> source

	new_img = tps.warpImage(img)

	# get the warp kps in for source and target
	tps.estimateTransformation(source, target, matches)  # note it is source --> target
	# there is a bug here, applyTransformation must receive np.float32 data type
	f32_pts = source.astype(np.float32)#np.zeros(source.shape, dtype=np.float32)
	f32_pts[:] = source[:]
	transform_cost, new_pts1 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2
	f32_pts = np.zeros(target.shape, dtype=np.float32)
	f32_pts[:] = target[:]
	transform_cost, new_pts2 = tps.applyTransformation(f32_pts)  # e.g., 1 x 4 x 2

	return new_img, new_pts1, new_pts2

def thin_plate_transform(x,y,offw,offh,imshape,shift_l=-0.05,shift_r=0.05,num_points=5,offsetMatrix=False):
	rand_p=np.random.choice(x.size,num_points,replace=False)
	movingPoints=np.zeros((1,num_points,2),dtype='float32')
	fixedPoints=np.zeros((1,num_points,2),dtype='float32')

	movingPoints[:,:,0]=x[rand_p]
	movingPoints[:,:,1]=y[rand_p]
	fixedPoints[:,:,0]=movingPoints[:,:,0]+offw*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)
	fixedPoints[:,:,1]=movingPoints[:,:,1]+offh*(np.random.rand(num_points)*(shift_r-shift_l)+shift_l)

	tps=cv2.createThinPlateSplineShapeTransformer()
	good_matches=[cv2.DMatch(i,i,0) for i in range(num_points)]
	tps.estimateTransformation(movingPoints,fixedPoints,good_matches)

	imh,imw=imshape
	x,y=np.meshgrid(np.arange(imw),np.arange(imh))
	x,y=x.astype('float32'),y.astype('float32')
	# there is a bug here, applyTransformation must receive np.float32 data type
	newxy=tps.applyTransformation(np.dstack((x.ravel(),y.ravel())))[1]
	newxy=newxy.reshape([imh,imw,2])

	if offsetMatrix:
		return newxy,newxy-np.dstack((x,y))
	else:
		return newxy
