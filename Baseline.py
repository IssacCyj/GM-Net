from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
import sys 
sys.setrecursionlimit(50000)

def ConvBNAct(x,channels,conv_stride_w,conv_stride_h,weight_decay=1e-4):
	x = Convolution2D(channels, conv_stride_w, conv_stride_h,
	                  init="he_uniform",
	                  border_mode="same",
	                  bias=False,
	                  W_regularizer=l2(weight_decay))(x)
	x = BatchNormalization(mode=0,
	                       axis=-1,
	                       gamma_regularizer=l2(weight_decay),
	                       beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)
	return x

def BottleNeck(x,channels,compression,weight_decay=1e-4):
	x = Convolution2D(channels*compression, 1, 1,
	                  init="he_uniform",
	                  border_mode="same",
	                  bias=False,
	                  W_regularizer=l2(weight_decay))(x)
	x = BatchNormalization(mode=0,
	                       axis=-1,
	                       gamma_regularizer=l2(weight_decay),
	                       beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)

	x = Convolution2D(channels*compression, 3, 3,
	                  init="he_uniform",
	                  border_mode="same",
	                  bias=False,
	                  W_regularizer=l2(weight_decay))(x)
	x = BatchNormalization(mode=0,
	                       axis=-1,
	                       gamma_regularizer=l2(weight_decay),
	                       beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)

	x = Convolution2D(channels, 1, 1,
	                  init="he_uniform",
	                  border_mode="same",
	                  bias=False,
	                  W_regularizer=l2(weight_decay))(x)
	x = BatchNormalization(mode=0,
	                       axis=-1,
	                       gamma_regularizer=l2(weight_decay),
	                       beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)
	return x


def Baseline(nb_classes, img_dim, depth, nb_dense_block, growth_rate,
             nb_filter, dropout_rate=None, weight_decay=1E-4, compression=0.5):
	if K.image_dim_ordering() == "th":
		concat_axis = 1
	elif K.image_dim_ordering() == "tf":
		concat_axis = -1
	model_input = Input(shape=(224,224,3))
	#init block
	print("init block...")
	#==================================================================================
	# x = ConvBNAct(model_input,32,3,3)
	# x = ConvBNAct(x,64,3,3)
	# if dropout_rate:
	#     x = Dropout(dropout_rate)(x)
	# x_1 = AveragePooling2D((2, 2), strides=(2,2))(x)
	#==================================================================================
	x = Convolution2D(32, 5, 5,
	                  init="he_uniform",
	                  border_mode="same",
	                  bias=False,
	                  W_regularizer=l2(weight_decay),
	                  subsample=(2,2))(model_input)
	x = BatchNormalization(mode=0,
	                       axis=-1,
	                       gamma_regularizer=l2(weight_decay),
	                       beta_regularizer=l2(weight_decay))(x)
	x_1 = Activation('relu')(x)
	#==================================================================================

	#main flow
	print("main flow...")
	#==================================================================================
	#nb_groups = 32
	#slices = [4,8,16]
	nb_block = 4
	c_init = 8
	c_factor = [1,1,2,4]
	cs = [64,128,256,512]

	#stage 1
	#---------------------------------------------------------------
	for g in range(nb_block):
		print("nb block%d"%g)
		merge_adapting_unit = []
		merge_path = []
		merge_block = []
		c = c_init * c_factor[g]
		# merge_block.append(x)
		for i in range(int(cs[g]/c)):
			print("nb block %d nb group %d"%(g,i))
			# res_1 = x
			x = Convolution2D(c, 1, 1,
	                  init="he_uniform",
	                  border_mode="same",
	                  bias=False,
	                  W_regularizer=l2(weight_decay))(x_1)
			merge_block.append(x)
			x = BatchNormalization(mode=0,
	                       axis=-1,
	                       gamma_regularizer=l2(weight_decay),
	                       beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			# x = ConvBNAct(x_1,slices[g],1,1)
			
			#merge_unit = []
			#merge_unit.append(x)

			x = BottleNeck(x,c,0.5)
			#x = BottleNeck(x,c,0.5)

			#merge_unit.append(x)
			#x = merge(merge_unit, mode='sum')
			#merge_unit = []
			#merge_unit.append(x)

			x = BottleNeck(x,c,0.5)
			x = BottleNeck(x,c,0.5)

			#merge_unit.append(x)
			#x = merge(merge_unit, mode='sum')
			merge_path.append(x)
		x = merge(merge_path, mode='concat', concat_axis=concat_axis)
		block_input = merge(merge_block, mode='concat', concat_axis=concat_axis)
		x_1 = merge([x,block_input], mode='sum')


		if g != nb_block-1:
			merge_adapting_unit.append(x_1)
			x = BottleNeck(x_1,int(cs[g]),0.5)
			x = BottleNeck(x,int(cs[g]),0.5)
			merge_adapting_unit.append(x)
			x = merge(merge_adapting_unit, mode='sum')
			# x = BottleNeck(x,slices[g]*nb_groups*2,0.5)
			x_1 = AveragePooling2D((2, 2), strides=(2,2))(x)


	#end flow
	print("end flow...")
	#==================================================================================
	merge_end = []
	merge_end.append(x_1)
	x = BottleNeck(x_1,int(cs[g]),0.5)
	x = BottleNeck(x,int(cs[g]),0.5)
	merge_end.append(x)
	x = merge(merge_end, mode='sum')
	x = GlobalAveragePooling2D(dim_ordering="tf")(x)
	x = Dense(nb_classes,
	          activation='softmax',
	          W_regularizer=l2(weight_decay),
	          b_regularizer=l2(weight_decay))(x)

	Baseline = Model(input=[model_input], output=[x], name="Baseline")
	#==================================================================================
	#==================================================================================
	#==================================================================================
	#==================================================================================
	#==================================================================================
	#==================================================================================
	return Baseline







