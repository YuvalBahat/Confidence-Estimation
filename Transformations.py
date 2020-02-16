import cv2
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
from scipy.ndimage import rotate
import copy
# import re
# from example_utils import Softmax #Used for producing outputs based on ensemble of transformations
# from scipy.ndimage.interpolation import shift
# from skimage.transform import warp, AffineTransform
# import sys

# Implemented transformations:
#   'horFlip':              Horizontal image flip
#   'vertFlip':              Vertical image flip
#   'increaseContrast<x>':  Increase image contrast by 0.1*x (e.g. 'increaseContrast3': Increase contrast by 0.3)
#   'BW':                   Convert image to gray-scale
#   'gamma<x>':             Gamma transformation. Each pixel value is raised to the power of 0.1*x, after being normalized to the range [0,1]
#   'blur<x>':              Horizontally blur the image with a magnitude of x pixels.
#   'zoomin<x>':            Zoom-in to 0.01*x center of image, then resize (using bilinear interolation) to original image size
#   'unisoZoom<x>_<y>'      Zoom-in to 0.01*x center of image in the horizontal axis and 0.01*y in the vertical axis, then resize (using bilinear interolation) to original image size
#   'crop<x>_<y>':          Zoom-in to 0.01*x portion of image, 0.01*y right and down from the upper left corner, then resize (using bilinear interolation) to original image size
#   'rotate<x>':            Zoom-in to avoid invalid pixels, then rotate image counter-clockwise by x degrees
#   'shift<x>_<y>':         Translate image by x pixels to the right and y pixels down, filling out-of-input boundaries pixels with constant 0. Supports sub-pixel shifts.
#                           When using random parameters, passing positive x and y results in randomly (and independently) choosing negative x and y as well.
#   'GaussianNoise<x>':     Adding Gaussian noise with STD x, assuming pixels are in the range [0,255]
#   'GN<x>':                Same as GaussianNoise
#   'masking<x>':           Masking 0.01*x of image pixels (randomly picked), by changing their value to (self.max_pixel_value-self.min_pixel_value)/2

RANDOMLY_NEGATED_PARAMS_TRANSFORMATIONS = ['shift']
TRANSFORMATIONS_ALLOWING_RANDOM_PLACEHOLDER = ['GaussianNoise','GN']

class Transformer():
    def __init__(self, transformations, batch_operation=True,min_pixel_value=0,max_pixel_value=255,deterministic_randomness=None):
        # Inputs:
        #   transformations:                    List of either strings or lists of string (out of the optional tranformations listed above) defining the
        #                                       transformations to be applied on all images. Each string may optionally contain a parameter, in the form
        #                                       of <transformation_name><parameter> (e.g. gamma8.5). When a transformation is a list of strings, these transformations
        #                                       are applied one after the other on each image.
        #                                       Parameters are separated by an ubderscore. For random parameters, pass a range, separated by an asterix, e.g. 'gamma7.1*8.5
        #   batch_operation:                    Whether the transformer should operate on a single images tensor or a batch of images tensor (default)
        #   min_pixel_value,max_pixel_value:    Pixels' dynamic range is expected to be in [min_pixel_value,max_pixel_value]
        #   deterministic_randomness:           When using random parameters, draw values at initializtion and use them consistently throughout the run.
        #                                       Added to allow iterative computation of expectation over random parameters, for white-box attacks.
        # Output:
        #   A transformer object

        self.batch_operation = batch_operation
        self.min_pixel_value = float(min_pixel_value)
        self.max_pixel_value = float(max_pixel_value)
        self.pixels_value_range = max_pixel_value-min_pixel_value
        self.transformations = [t if isinstance(t,list) else [t] for t in copy.deepcopy(transformations)]
        self.num_transformations = len(self.transformations)
        self.per_image_copies = len(self.transformations) + 1
        self.transformation_param = [[[] for i_sub in range(len(self.transformations[i]))] for i in range(self.num_transformations)]
        self.random_transformation = [np.zeros([len(self.transformations[i])]).astype(np.bool) for i in range(self.num_transformations)]
        for ind, cur_transformation in enumerate(self.transformations):
            for sub_ind,sub_transformation in enumerate (cur_transformation):
                self.transformations[ind][sub_ind], self.transformation_param[ind][sub_ind], self.random_transformation[ind][sub_ind] =\
                    ParseParameters(sub_transformation,deterministic_randomness)
        self.PrintInitialization()

    def PrintInitialization(self):
        print('Initialized transformer with the following transformations:')
        for i,t in enumerate(self.transformations):
            print('(%d) %s:'%(i,'->'.join(t)),'\t%s'%(' -> '.join(['None' if self.transformation_param[i][sub_t] is None else ', '.join([('-'.join([str(param) for param in \
                self.transformation_param[i][sub_t][param_num]]) if self.random_transformation[i][sub_t] else\
                str(self.transformation_param[i][sub_t][param_num])) for param_num in range(len(self.transformation_param[i][sub_t]))]) for\
                sub_t in range(len(self.transformation_param[i]))])))

    def TransformationParameter(self,ind,shape=1,TF=True):
        if self.random_transformation[ind[0]][ind[1]]:
            if TF:
                if shape==1:
                    returnable = [tf.random_uniform([],          minval=par[0],maxval=par[1]) for par in self.transformation_param[ind[0]][ind[1]]]
                else:
                    returnable =  [tf.random_uniform(shape=shape, minval=par[0],maxval=par[1]) for par in self.transformation_param[ind[0]][ind[1]]]
                if self.transformations[ind[0]][ind[1]] in RANDOMLY_NEGATED_PARAMS_TRANSFORMATIONS:
                    returnable = [2*(tf.cast(tf.greater(tf.random_uniform(shape=tensor.get_shape(),minval=0,maxval=1),0.5),dtype=tensor.dtype)-0.5)*tensor for tensor in returnable]
                return returnable
            else:
                # shape = 0 if shape is None else shape
                returnable = [np.random.uniform(low=par[0], high=par[1],size=[shape]) for par in self.transformation_param[ind[0]][ind[1]]]
                if self.transformations[ind[0]][ind[1]] in RANDOMLY_NEGATED_PARAMS_TRANSFORMATIONS:
                    returnable = [2 * ((np.random.uniform(size=value.shape, low=0, high=1)> 0.5).astype(value.dtype) - 0.5) * value for value in returnable]
                return returnable
        else:
            return [par*np.ones([shape]) for par in self.transformation_param[ind[0]][ind[1]]]

    def TransformImages(self,images,labels=None):
        # Creating the transformed images and labels TensorFlow operator.
        # Inputs:
        #   images: A single image (HxWxC) or a batch of images (NxHxWxC) tensor (depending on batch_operation)
        #   labels: A 1-D tensor of corresponding image labels
        # Outputs:
        #   output_images:  A batch of images and their transformed versions. Each image is followed by its transformed versions, then by the next image (if batch_operation).
        #   output_labels:  A 1-D batch of corresponding labels
        assert (len(images.shape)==3 and not self.batch_operation) or (len(images.shape)==4 and self.batch_operation),'Incorrect shape of images input'
        if not self.batch_operation:
            images = np.expand_dims(images,axis=0)
        image_shape = np.array(list(images.shape)[1:3])
        num_pixels = np.prod(image_shape[:2])
        non_modified_images = images.astype(np.float32)
        if any([any([('Contrast' in T) for T in Ts]) for Ts in self.transformations]):
            image_mean = np.mean(images,axis=(1,2),keepdims=True)
        images2use = np.expand_dims(images,axis=1)
        for ind,cur_chained_transformation in enumerate(self.transformations):
            modified_image = 1.*non_modified_images
            for sub_ind,cur_transformation in enumerate(cur_chained_transformation):
                if any([name in cur_transformation for name in ['GaussianNoise','GN']]):
                    assert not self.random_transformation[ind][sub_ind],'Unsupported yet'
                    modified_image = np.maximum(self.min_pixel_value,np.minimum(self.max_pixel_value,modified_image+np.random.normal(loc=0,
                        scale=self.TransformationParameter((ind,sub_ind),TF=False)[0]/255*self.pixels_value_range,size=modified_image.shape)))
                elif 'masking' in cur_transformation:
                    assert not self.random_transformation[ind][sub_ind],'Unsupported yet'
                    masked_pixels = [np.random.permutation(num_pixels)[:int(0.01*self.TransformationParameter((ind,sub_ind),TF=False)[0]*num_pixels)] for i in range(len(modified_image))]
                    for im_num in range(len(modified_image)):
                        masked_pixels_y, masked_pixels_x = np.unravel_index(masked_pixels[im_num], dims=image_shape)
                        for channel_num in range(modified_image.shape[-1]):
                            modified_image[im_num][masked_pixels_y,masked_pixels_x,channel_num] = (self.max_pixel_value-self.min_pixel_value)/2
                elif 'increaseContrast' in cur_transformation:
                    modified_image = np.maximum(self.min_pixel_value,np.minimum((modified_image-image_mean)*(1+0.1*self.TransformationParameter((ind,sub_ind),TF=False)[0])+image_mean,self.max_pixel_value))
                elif 'horFlip' in cur_transformation:
                    modified_image = modified_image[:,:,::-1,:]
                elif 'placibo' in cur_transformation:
                    pass
                elif 'vertFlip' in cur_transformation:
                    modified_image = modified_image[:, ::-1, :, :]
                elif cur_transformation=='colorSwap':
                    modified_image = modified_image[:, :, :,[1,2,0]]
                elif 'blur' in cur_transformation:
                    blur_pixels = int(self.TransformationParameter((ind,sub_ind),TF=False)[0])
                    assert blur_pixels>=2,'Blurring the image with blur kernel of size %d makes no difference'%(blur_pixels)
                    pre_blur_images = np.pad(modified_image,pad_width=((0,0),(0,0),(int((blur_pixels-1)/2),int((blur_pixels-1)/2)),(0,0)),mode='reflect')
                    modified_image = np.zeros_like(modified_image)
                    for pixel_num in range(blur_pixels):
                        modified_image += (pre_blur_images/blur_pixels)[:,:,pixel_num:(pixel_num+image_shape[1]),:]
                elif 'BW' in cur_transformation:
                    modified_image = np.repeat(np.sum(modified_image*np.reshape([0.299,0.587,0.114],newshape=[1,1,1,3]),axis=3,keepdims=True),repeats=3,axis=3)
                elif 'gamma' in cur_transformation:
                    modified_image = np.clip(modified_image,a_min=self.min_pixel_value,a_max=self.max_pixel_value)
                    # tf.Assert(tf.reduce_all(tf.greater_equal(modified_image,0)),[tf.reduce_min(modified_image)])
                    modified_image = (((modified_image-self.min_pixel_value)/self.pixels_value_range)**\
                        np.reshape(0.1*self.TransformationParameter((ind,sub_ind), shape=len(images),TF=False)[0],[-1,1,1,1]))*self.pixels_value_range+self.min_pixel_value
                elif 'resize' in cur_transformation:
                    resize_param = self.TransformationParameter((ind, sub_ind), shape=len(images),TF=False)
                    modified_image = np.stack([cv2.resize(cv2.resize(im,tuple(np.round(resize_param[0][i]/100*image_shape).astype(np.int32)),interpolation=cv2.INTER_LINEAR),
                                                          tuple(image_shape),interpolation=cv2.INTER_LINEAR)
                                               for i, im in enumerate(modified_image)], axis=0).astype(images.dtype)
                elif 'zoomin' in cur_transformation:
                    crop_params = self.TransformationParameter((ind, sub_ind), shape=len(images),TF=False)
                    modified_image = np.stack([imzoom(im,f=[100 / crop_params[0][i]],min_pix=self.min_pixel_value,max_pix=self.max_pixel_value)
                                               for i, im in enumerate(modified_image)], axis=0).astype(images.dtype)
                elif 'unisoZoom' in cur_transformation:
                    crop_params = self.TransformationParameter((ind, sub_ind), shape=len(images),TF=False)
                    modified_image = np.stack([imzoom(im,f=[100 / par[i] for par in crop_params],min_pix=self.min_pixel_value,max_pix=self.max_pixel_value)
                                               for i, im in enumerate(modified_image)], axis=0).astype(images.dtype)
                elif 'shift' in cur_transformation:
                    shift_params = self.TransformationParameter((ind, sub_ind),shape=len(images), TF=False)
                    if self.random_transformation[ind][sub_ind]:#Introducing random translation direction (avoid moving just to one quarter):
                        shift_params = [((np.random.uniform(low=0,high=1,size=len(param))>0.5)-0.5)*2*param for param in shift_params]
                    h, w = images.shape[1:3]
                    transformations = [np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32) for tx, ty in zip(*shift_params)]
                    warped_images = [cv2.warpAffine(img, M, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                     for img, M in zip(modified_image, transformations)]
                    modified_image = np.stack(warped_images, axis=0)
                elif 'rotate' in cur_transformation:
                    rotation_angle = self.TransformationParameter((ind,sub_ind),TF=False)[0]
                    pre_rot_rows = modified_image.shape[1]
                    modified_image = rotate(modified_image,angle=rotation_angle,axes=(1,2),reshape=True)
                    rotation_angle = np.mod(rotation_angle,90)
                    rows2remove = 2*modified_image.shape[1]*pre_rot_rows*np.sin(np.deg2rad(rotation_angle))/\
                          (modified_image.shape[1]+modified_image.shape[2]*np.tan(np.deg2rad(rotation_angle)))
                    upscaled_size = (np.ceil(pre_rot_rows/(modified_image.shape[1]-rows2remove)*modified_image.shape[1]/2)*2*np.ones([2])).astype(int)
                    modified_image = np.stack([imresize(im,upscaled_size,interp='bicubic') for im in modified_image],axis=0)
                    modified_image = modified_image/255*self.max_pixel_value
                    margins2crop = ((np.array(modified_image.shape[1:3])-np.array(non_modified_images.shape[1:3]))/2).astype(np.int32)
                    if np.any(margins2crop==0):
                        if margins2crop[0]!=0:
                            modified_image = modified_image[:, margins2crop[0]:-margins2crop[0],:, :]
                        elif margins2crop[1]!=0:
                            modified_image = modified_image[:,:,margins2crop[1]:-margins2crop[1],:]
                    else:
                        modified_image = modified_image[:,margins2crop[0]:-margins2crop[0],margins2crop[1]:-margins2crop[1],:]
                elif 'crop' in cur_transformation:
                    crop_params = self.TransformationParameter((ind, sub_ind), shape=len(images),TF=False)
                    crop_params[1] = np.minimum(crop_params[1],100-crop_params[0])
                    crop_params[1] = (crop_params[1]/crop_params[0]*image_shape[0]).astype(np.int32)
                    modified_image = np.stack([imresize(im,size=100/crop_params[0][i],interp='bicubic')[crop_params[1][i]:crop_params[1][i]+image_shape[0],
                        crop_params[1][i]:crop_params[1][i]+image_shape[1],:] for i,im in enumerate(modified_image)],axis=0)
                else:
                    raise Exception('Transformation %s not implemented'%(cur_transformation))
            images2use = np.concatenate((images2use,np.expand_dims(modified_image.astype(images.dtype),axis=1)),axis=1)
        output_images =  np.reshape(images2use,[-1]+list(images.shape)[1:])
        output_labels = np.reshape(np.repeat(np.expand_dims(labels, axis=1),repeats=self.per_image_copies,axis=1),[-1]) if labels is not None else None
        if output_labels is None:
            return output_images
        else:
            return output_images,output_labels

    def TransformImages_TF_OP(self,images,labels=None,drawn_randoms=None):
        # Creating the transformed images and labels TensorFlow operator.
        # Inputs:
    #   images:             A single image (HxWxC) or a batch of images (NxHxWxC) tensor (depending on batch_operation)
        #   labels:         A 1-D tensor of corresponding image labels
        #   drawn_randoms:  tf placeholder. If passed when using transformations with inherent randomness (e.g. Gaussian noise), using the given values instead of drawing randomly.
        # Outputs:
        #   output_images:  A batch of images and their transformed versions. Each image is followed by its transformed versions, then by the next image (if batch_operation).
        #   output_labels:  A 1-D batch of corresponding labels
        assert (len(images.get_shape())==3 and not self.batch_operation) or (len(images.get_shape())==4 and self.batch_operation),'Incorrect shape of images input'
        if not self.batch_operation:
            images = tf.expand_dims(images,axis=0)
        image_shape = np.array(images.get_shape().as_list()[1:3])
        non_modified_images = tf.cast(images,tf.float32)
        if any([any([('Contrast' in T) for T in Ts]) for Ts in self.transformations]):
            image_mean = tf.reduce_mean(images,axis=(1,2),keep_dims=True)
        images2use = tf.expand_dims(images,axis=1)
        for ind,cur_chained_transformation in enumerate(self.transformations):
            modified_image = 1.*non_modified_images
            for sub_ind,cur_transformation in enumerate(cur_chained_transformation):
                if any([name in cur_transformation for name in ['GaussianNoise','GN']]):
                    assert not self.random_transformation[ind][sub_ind],'Unsupported yet'
                    if drawn_randoms is None:
                        additive_noise = tf.random_normal(shape=modified_image.get_shape())
                    else:
                        additive_noise = drawn_randoms
                    modified_image = tf.maximum(self.min_pixel_value, tf.minimum(self.max_pixel_value,modified_image+
                        self.TransformationParameter((ind,sub_ind))[0]/255*self.pixels_value_range*additive_noise))
                elif 'increaseContrast' in cur_transformation:
                    modified_image = tf.maximum(self.min_pixel_value,tf.minimum((modified_image-image_mean)*(1+0.1*self.TransformationParameter((ind,sub_ind))[0])+image_mean,self.max_pixel_value))
                elif 'horFlip' in cur_transformation:
                    # modified_image = tf.image.flip_left_right(modified_image)
                    modified_image = tf.map_fn(lambda image: tf.image.flip_left_right(image), modified_image)
                elif 'identity' in cur_transformation:
                    # modified_image = tf.image.flip_left_right(modified_image)
                    modified_image = tf.map_fn(lambda image: tf.identity(image), modified_image)
                elif 'vertFlip' in cur_transformation:
                    modified_image = tf.map_fn(lambda image: tf.image.flip_up_down(image), modified_image)
                elif 'blur' in cur_transformation:
                    blur_pixels = int(self.TransformationParameter((ind,sub_ind))[0])
                    assert blur_pixels>=2,'Blurring the image with blur kernel of size %d makes no difference'%(blur_pixels)
                    pre_blur_images = tf.pad(modified_image,paddings=((0,0),(0,0),(int((blur_pixels-1)/2),int((blur_pixels-1)/2)),(0,0)),mode='SYMMETRIC')
                    modified_image = tf.zeros_like(modified_image)
                    for pixel_num in range(blur_pixels):
                        modified_image = tf.add(modified_image,tf.slice(pre_blur_images/blur_pixels,begin=[0,0,pixel_num,0],size=[-1,-1,tf.shape(modified_image)[1],-1]))
                elif 'BW' in cur_transformation:
                    modified_image = tf.tile(tf.reduce_sum(tf.multiply(modified_image,tf.reshape(tf.constant([0.299,0.587,0.114]),[1,1,1,3])),axis=3,keep_dims=True),multiples=[1,1,1,3])
                elif 'gamma' in cur_transformation:
                    modified_image = tf.clip_by_value(modified_image,clip_value_min=self.min_pixel_value,clip_value_max=self.max_pixel_value)
                    # tf.Assert(tf.reduce_all(tf.greater_equal(modified_image,0)),[tf.reduce_min(modified_image)])
                    modified_image = tf.pow((modified_image-self.min_pixel_value)/self.pixels_value_range,0.1*self.TransformationParameter((ind,sub_ind))[0])*self.pixels_value_range+self.min_pixel_value
                elif 'shift' in cur_transformation:
                    if self.random_transformation[ind][sub_ind]:
                        shift_params = self.TransformationParameter((ind, sub_ind),shape=[int(modified_image.get_shape()[0])])
                        mat = tf.stack([1.+0.*shift_params[0], 0.*shift_params[0], -shift_params[0],0.*shift_params[0], 1.+0.*shift_params[0],-shift_params[1],
                                         0. * shift_params[0],0.*shift_params[0]],axis=1)
                    else:
                        shift_params = self.TransformationParameter((ind, sub_ind))
                        mat = np.array([[1, 0, -shift_params[0],0, 1,-shift_params[1],0,0]], dtype=np.float32)
                    modified_image = tf.contrib.image.transform(images=modified_image,transforms=mat,interpolation='BILINEAR')
                elif 'unisoZoom' in cur_transformation:
                    assert not self.random_transformation[ind][sub_ind], 'Unsupported yet'
                    tx, ty = image_shape[0] // 2, image_shape[1] // 2
                    crop_params = self.TransformationParameter((ind, sub_ind))
                    mat =  np.array([crop_params[0]/100,0,(1-crop_params[0]/100)*tx,0,crop_params[1]/100,(1-crop_params[1]/100)*ty,0,0], dtype=np.float32)
                    modified_image = tf.contrib.image.transform(images=modified_image, transforms=mat, interpolation='BILINEAR')
                elif 'resize' in cur_transformation:
                    assert not self.random_transformation[ind][sub_ind], 'Unsupported yet'
                    resize_param = self.TransformationParameter((ind, sub_ind))
                    modified_image = tf.image.resize_bilinear(tf.image.resize_bilinear(modified_image,size=np.round(resize_param[0]/100*image_shape).astype(np.int32),align_corners=True),
                                                              size=image_shape,align_corners=True)
                elif 'zoomin' in cur_transformation:
                    tx, ty = image_shape[0] // 2, image_shape[1] // 2
                    if self.random_transformation[ind][sub_ind]:
                        crop_params = self.TransformationParameter((ind, sub_ind), shape=tf.reshape(tf.shape(images)[0], [1]))
                        mat =  tf.stack([crop_params[0]/100,0.*crop_params[0],(1-crop_params[0]/100)*tx,0.*crop_params[0],crop_params[0]/100,
                                         (1-crop_params[0]/100)*ty,0.*crop_params[0],0.*crop_params[0]],axis=1)
                    else:
                        crop_params = self.TransformationParameter((ind, sub_ind))
                        mat =  np.array([crop_params[0]/100,0,(1-crop_params[0]/100)*tx,0,crop_params[0]/100,(1-crop_params[0]/100)*ty,0,0], dtype=np.float32)
                    modified_image = tf.contrib.image.transform(images=modified_image, transforms=mat, interpolation='BILINEAR')
                elif 'crop' in cur_transformation:
                    if self.random_transformation[ind][sub_ind]:
                        crop_params = self.TransformationParameter((ind, sub_ind), shape=tf.reshape(tf.shape(images)[0], [1]))
                        crop_params[1] = tf.minimum(crop_params[1],100-crop_params[0])
                        boxes = 0.01*np.ones([1,4])*tf.reshape(tf.cast(crop_params[1],dtype=tf.float32),[-1,1])+0.01*np.reshape([0,0,1,1],[1,4])*tf.reshape(tf.cast(crop_params[0],dtype=tf.float32),[-1,1])
                    else:
                        crop_params = self.TransformationParameter((ind, sub_ind))
                        boxes = 0.01*(np.ones([1,4])*crop_params[1]+np.reshape([0,0,1,1],[1,4])*crop_params[0])*tf.reshape(tf.ones(shape=tf.reshape(tf.shape(images)[0],[1])),[-1,1])
                    box_ind = tf.cast(tf.cumsum(tf.ones(shape=tf.reshape(tf.shape(images)[0], [1])), axis=0) - 1,dtype=tf.int32)
                    crop_size = tf.constant(image_shape, dtype=tf.int32)
                    modified_image = tf.image.crop_and_resize(image=modified_image, boxes=boxes, box_ind=box_ind,crop_size=crop_size)
                else:
                    raise Exception('Transformation %s not implemented'%(cur_transformation))
            images2use = tf.concat((images2use,tf.expand_dims(tf.cast(modified_image,images.dtype),axis=1)),axis=1)
        output_images =  tf.reshape(images2use,[-1]+images.get_shape().as_list()[1:])
        if labels is not None:
            if len(labels.get_shape())>1:#Labels in a "one-hot" configuration:
                output_labels = tf.reshape(tf.tile(tf.expand_dims(labels, axis=1), multiples=[1,self.per_image_copies,1]),[-1]+[int(labels.get_shape()[1])])
            else:
                output_labels = tf.reshape(tf.tile(tf.expand_dims(labels, axis=1), multiples=[1, self.per_image_copies]),[-1])

            return output_images,output_labels
        else:
            return output_images

    def Process_NonLogits(self,input_array):
        # After running a classifier on the perturbed images, all outputs (but the logits) repeat themselves per_image_copies number
            #  of times. In all outputs but the logits, we are only interested in the output for the original images. This function gets such output
            #   and returns the relevant portion of it.
        input_array_shape = input_array.shape
        assert np.mod(input_array_shape[0],self.per_image_copies)==0,'Input size is not an integer multiplication of number of transformations'
        # assert input_array.shape[0]==self.num_of_output_images,'Expected %d rows but got %d'%(self.num_of_output_images,input_array_shape[0])
        return np.reshape(input_array,[input_array_shape[0]//self.per_image_copies,self.per_image_copies]+list(input_array_shape[1:]))[:,0,...]

    def Process_NonLogits_TF_OP(self,input_tensor,avoid_gradients_calc=False):
        # The outputs of a classifier fed with transformed images will often correspond to its input batch size, which was modified by the transformer.
        # This function disimply scards the outputs that correspond to the transformed images and outputs only those corresponding to the original images.
        # To be used on all output tensors (e.g. "correct classification" tensor) but the logits tensor
        # Input:
        #   A tensor whose first dimension corresponds to the classifier's input batch size
        # Output: (I apply tf.stop_gradient on the output to avoid the unnecessary gradients calculation when training a detector)
        #  The input tensor after filtering out values corresponding to the transformed images (along the first dimension)

        input_tensor_shape = input_tensor.get_shape().as_list()
        if len(input_tensor_shape)>1:
            input_tensor = tf.reshape(input_tensor,[-1,self.per_image_copies]+input_tensor_shape[1:])
            tensor2return = tf.reshape(tf.slice(input_tensor,begin=[0,0]+list(np.zeros([len(input_tensor_shape)-1]).astype(np.int32)),size=[-1,1]+list(-1*np.ones([len(input_tensor_shape)-1]).astype(np.int32))),
                [-1]+input_tensor_shape[1:])
        else:
            input_tensor = tf.reshape(input_tensor,[-1,self.per_image_copies])
            tensor2return = tf.reshape(tf.slice(input_tensor,begin=[0,0]+list(np.zeros([len(input_tensor_shape)-1]).astype(np.int32)),size=[-1,1]+list(-1*np.ones([len(input_tensor_shape)-1]).astype(np.int32))),[-1])
        if avoid_gradients_calc:
            return tf.stop_gradient(tensor2return)
        else:
            return tensor2return

    def Process_Logits(self,input_logits,reorder_logits=None,num_logits_per_transformation=-1,use_ensemble=False):
        # Converts the logit output of a classifier fed by transformed images into a logits vector corresponding to the original image and a features vector.
        # Inputs:
        #   input_logits:                   The logits output of a classifier of interest, in the shape of NxNUM_CLASSES,
        #                                   where N is the original batch size X (number of transformations+1)
        #   reorder_logits:                 If True (default), the features vector has the logits corresponding to all transformations (including the original
        #                                   non-transformed image) ordered according to a descending order of the logits corresponding to the original image.
        #   num_logits_per_transformation:  (optional) Using only the logits corresponding to the top num_logits_per_transformation logits of the original image.
        #                                   To use this option, pass an integer between 1 and NUM_CLASSES-1
        #   use_ensemble:                   Outputing logits corresponding to the average over the softmax of all transformations, rather than to original version. When
        #                                   reorder_logits, order is based on ensemble too.
        # Outputs:
        #   logits_output:  Logits tensors corresponding to the original or ensemble (see use_ensemble input), non-transformed, image.
        #                   The logits corresponding to the transformed versions are removed.
        #   features_vect:  Feature vectors tensor of shape N x (number of transformation+1) x min(NUM_CLASSES,num_logits_per_transformation)
        input_logits_shape = list(input_logits.shape)
        assert len(input_logits_shape)==2,'Unrecognized logits shape'
        assert not (num_logits_per_transformation>0 and reorder_logits is None),'Cannot keep k logits per transformation without reordering them'
        assert not num_logits_per_transformation>input_logits_shape[1],'Cannot keep more logits (%d) than there are originally (%d)'%(num_logits_per_transformation,input_logits_shape[1])
        input_logits = input_logits.reshape([-1,self.per_image_copies,input_logits_shape[1]])
        if use_ensemble: #Computing softmax here, but doing so without increasing precision to float64, since this needs to support both ndarray and torch.Tensor. See if it is good enough.
            logits_output = input_logits.exp()
            logits_output = logits_output/logits_output.sum(2,keepdim=True)
            logits_output = logits_output.mean(1).log()
        else:
            logits_output = (input_logits[:,:1,:]).reshape([-1,input_logits_shape[-1]])
        if reorder_logits:
            logits_output_shape = list(logits_output.shape)
            descending_order = logits_output.argsort(1,descending=True)
            if num_logits_per_transformation > 0:
                descending_order = descending_order[:,:num_logits_per_transformation]
            input_logits = input_logits[np.arange(logits_output_shape[0]).reshape([-1,1,1]),np.arange(self.per_image_copies).reshape([1,-1,1]),descending_order.unsqueeze(1)]
            features_vect = input_logits.reshape([logits_output_shape[0],self.per_image_copies*descending_order.shape[1]])
        else:
            features_vect = input_logits.reshape([-1, self.per_image_copies * input_logits_shape[-1]])
        return logits_output,features_vect

    def Process_Logits_TF_OP(self,input_logits,reorder_logits=True,num_logits_per_transformation=-1,avoid_gradients_calc=True):
        # Converts the logit output of a classifier fed by transformed images into a logits vector corresponding to the original image and a features vector.
        # Inputs:
        #   input_logits:                   The logits output of a classifier of interest, in the shape of NxNUM_CLASSES,
        #                                   where N is the original batch size X (number of transformations+1)
        #   reorder_logits:                 If True (default), the features vector has the logits corresponding to all transformations (including the original
        #                                   non-transformed image) ordered according to a descending order of the logits corresponding to the original image.
        #   num_logits_per_transformation:  (optional) Using only the logits corresponding to the top num_logits_per_transformation logits of the original image.
        #                                   To use this option, pass an integer between 1 and NUM_CLASSES-1
        # Outputs: (I apply tf.stop_gradient on both outputs to avoid the unnecessary gradients calculation when training a detector)
        #   logits_of_original: Logits tensors corresponding to the original, non-transformed, image. The logits corresponding to the transformed versions are removed.
        #   features_vect:      Feature vectors tensor of shape N x (number of transformation+1) x min(NUM_CLASSES,num_logits_per_transformation)
        input_logits_shape = input_logits.get_shape().as_list()
        assert len(input_logits_shape)==2,'Unrecognized logits shape'
        assert not (num_logits_per_transformation>0 and not reorder_logits),'Cannot keep k logits per transformation without reordering them'
        assert not num_logits_per_transformation>input_logits_shape[1],'Cannot keep more logits (%d) than there are originally (%d)'%(num_logits_per_transformation,input_logits_shape[1])
        input_logits = tf.reshape(input_logits,[-1,self.per_image_copies,input_logits_shape[1]])
        logits_of_original = tf.reshape(tf.slice(input_logits,begin=[0,0,0],size=[-1,1,-1]),[-1,input_logits_shape[-1]])
        if reorder_logits:
            org_logits_shape = logits_of_original.get_shape().as_list()
            org_logits_shape[1] = num_logits_per_transformation if num_logits_per_transformation>0 else org_logits_shape[1]
            _,descending_order = tf.nn.top_k(logits_of_original,k=org_logits_shape[1])
            descending_order = tf.tile(tf.reshape(descending_order,shape=[-1,1,org_logits_shape[1],1]),[1,self.per_image_copies,1,1])
            image_indices = tf.tile(tf.reshape(tf.range(org_logits_shape[0]),[org_logits_shape[0],1,1,1]),multiples=[1,self.per_image_copies,org_logits_shape[1],1])
            permutation_indices = tf.tile(tf.reshape(tf.range(self.per_image_copies),[1,self.per_image_copies,1,1]),[org_logits_shape[0],1,org_logits_shape[1],1])
            combined_indices = tf.concat([image_indices,permutation_indices,descending_order],axis=3)
            descending_values = tf.gather_nd(params=input_logits,indices=combined_indices)
            # if num_logits_per_transformation>0:
            #     descending_values = tf.slice(descending_values,begin=[0,0,0],size=[-1,-1,num_logits_per_transformation])
            features_vect = tf.reshape(descending_values,[int(input_logits_shape[0]/self.per_image_copies),-1])
        else:
            features_vect = tf.reshape(input_logits,[-1, self.per_image_copies * input_logits_shape[-1]])
        if avoid_gradients_calc:
            return tf.stop_gradient(logits_of_original),tf.stop_gradient(features_vect)
        else:
            return logits_of_original,features_vect

def ParseParameters(cur_transformation,deterministic_randomness):
    is_digit = [character.isdigit() or character=='-' for character in cur_transformation]
    is_asterisk = [character == '*' for character in cur_transformation]
    # is_underscore = [character == '_' for character in cur_transformation]
    transformation_name,transformation_param,random_transformation = cur_transformation,None,False
    if np.any(np.logical_or(is_asterisk,is_digit)):
        transformation_param = []
        params_first_ind = np.argwhere(np.logical_or(is_asterisk,is_digit))[0][0]
        params = transformation_name[params_first_ind:].split('_')
        transformation_name = cur_transformation[:params_first_ind]
        for param in params:
            if '*' in param:
                assert deterministic_randomness is not None,'Should determine whether to use deterministic or random randomness'
                if deterministic_randomness:
                    transformation_param.append(float(np.random.uniform(low=float(param[:param.find('*')]),high=float(param[param.find('*')+1:]))))
                    # if transformation_name in RANDOMLY_NEGATED_PARAMS_TRANSFORMATIONS and np.random.uniform(0,1)>0.5:
                    #     transformation_param[-1] *= -1.
                else:
                    random_transformation = True
                    transformation_param.append([float(param[:param.find('*')]),float(param[param.find('*')+1:])])
            else:
                transformation_param.append(float(param))
    return transformation_name,transformation_param,random_transformation

def imzoom(image, f,min_pix,max_pix):
    h, w = image.shape[:2]
    tx, ty = w // 2, h // 2
    f = [f[0],f[0]] if len(f)==1 else f
    mat = np.array([[f[0], 0, (1 - f[0]) * tx], [0, f[1], (1 - f[1]) * ty]], dtype=np.float32)
    warped_image = cv2.warpAffine(image, mat, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # warped_image = np.clip(warped_image, min_pix, max_pix)
    return warped_image

def Recursive_String_Replace(string,old,new):
    assert len(old)==len(new),'The two expressions lists should have the same length'
    for i in range(len(old)):
        if i==0:
            string = '_'.join([string.split('_')[i].replace(old[i], new[i]),
                               ''.join(string.split('_')[i + 1:])])
        elif i==len(old)-1:
            string = '_'.join([''.join(string.split('_')[:i]), string.split('_')[i].replace(old[i], new[i])])
        else:
            string = '_'.join([''.join(string.split('_')[:i]),string.split('_')[i].replace(old[i],new[i]),''.join(string.split('_')[i+1:])])
        # string = string.replace(old[i],new[i],1)
    return string