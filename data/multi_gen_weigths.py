#   Copyright (c) 2019, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without 
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright 
#       notice, this list of conditions and the following disclaimer in the 
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
import numpy as np
import os
import sys
import random 
import subprocess

outFileWeights = open("memdata.h" , "wt")
outFileWeights.write("#ifndef PARAMS_HPP\n")
outFileWeights.write("#define PARAMS_HPP\n")
outFileWeights.write("\n\n//Convolutional Layers\n\n")

outFileWeights.write("namespace PARAM{ \n")
outFileConfig = open("config.h" , "wt")

# kernel_dim = 3
# stride = 1
# input_precision = 8
# ifm_channels = 4
# ofm_channels = 16
# ifm_dimension = 3
# ofm_dimension = 1

# activation_precision = 16
# expand = 1
# simd = 2
# pe = 2
# w_precision = 1
# mmv=1

conv_layers = 2
fc_layers = 0
# [kernel_dim, stride, input_precision, ifm_channels, ofm_channels, ifm_dimension, ofm_dimension, activation_precision, expand, simd, pe, w_precision, mmv]
# [k, s, ip, ifmc, ofmc, ifmd, ofmd, ap, e, simd, pe, wp, mmv]
conv_params = [
#	[k,	s,	ip,	ifmc,	ofmc,	ifmd,	ofmd,	ap,	e,	simd,	pe,	wp,	mmv]
	[3,	1, 	8,	3,		16,		10,		8,		8, 1,	3,		2, 8,	1],
	[3,	1, 	8,	16,		16,		8,		6,		8, 1,	2,		2, 8,	1],
]

# fc_params = [
# #	[MatrixW, MatrixH, SIMD, PE, MMV, IP, WP, AP]
# 	[576, 10, 2, 2, 1, 8, 8, 8],
# ]

for layer in range(conv_layers):
	kernel_dim = conv_params[layer][0]
	stride = conv_params[layer][1]
	input_precision = conv_params[layer][2]
	ifm_channels = conv_params[layer][3]
	ofm_channels = conv_params[layer][4]
	ifm_dimension = conv_params[layer][5]
	ofm_dimension = conv_params[layer][6]
	activation_precision = conv_params[layer][7]
	expand = conv_params[layer][8]
	simd = conv_params[layer][9]
	pe = conv_params[layer][10]
	w_precision = conv_params[layer][11]
	mmv = conv_params[layer][12]

	tile = ifm_channels *kernel_dim*kernel_dim * ofm_channels // (simd*pe)

	# write layer
	outFileConfig.write("\n // Convolutional Layer L%d \n\n\n" % layer)

	# write config
	outFileConfig.write("#define L%d_KERNEL_DIM %d \n" % (layer, kernel_dim))
	outFileConfig.write("#define L%d_SIMD %d \n" % (layer, simd))
	outFileConfig.write("#define L%d_PE %d \n" % (layer, pe))
	outFileConfig.write("#define L%d_MMV %d \n" % (layer, mmv))
	outFileConfig.write("#define L%d_WIDTH %d \n" % (layer, w_precision))

	outFileConfig.write("#define L%d_IFM_Channels %d \n" % (layer, ifm_channels))
	outFileConfig.write("#define L%d_OFM_Channels %d \n" % (layer, ofm_channels))
	outFileConfig.write("#define L%d_IFMDim %d \n" % (layer, ifm_dimension))
	outFileConfig.write("#define L%d_OFMDim %d \n" % (layer, ofm_dimension))
	outFileConfig.write("#define L%d_STRIDE %d \n" % (layer, stride))
	outFileConfig.write("#define L%d_INPUT_PRECISION %d \n" % (layer, input_precision))
	outFileConfig.write("#define L%d_TILE %d \n" % (layer, tile))

	outFileConfig.write("#define L%d_ACTIVATION_PRECISION %d \n" % (layer, activation_precision))

	# write weights

	if (w_precision == 1):
		outFileWeights.write("static BinaryWeights<%d,%d,%d> weights_%d= {\n {\n" %(simd,pe,tile,layer))
	else:
		outFileWeights.write("static FixedPointWeights<%d,ap_int<%d>,%d,%d> weights_%d= {\n {\n" %(simd,w_precision,pe,tile,layer))

	for p in range(pe):
		outFileWeights.write("  {\n")
		for t in range(tile):
			width = simd*w_precision;
			val = random.randint(0, 1<<width-1)
			outFileWeights.write("   %s" % hex(val))
			if t!=tile-1:
				outFileWeights.write(",\n")
		outFileWeights.write("\n  }")
		if p!=pe-1:
			outFileWeights.write(",\n")

	outFileWeights.write("\n  }\n };\n")

outFileWeights.write("\n\n//Fully Connected Layers Layers\n\n")

for layer in range(conv_layers, conv_layers+fc_layers):
	MatrixW = fc_params[layer-conv_layers][0]
	MatrixH = fc_params[layer-conv_layers][1]
	SIMD = fc_params[layer-conv_layers][2]
	PE = fc_params[layer-conv_layers][3]
	MMV = fc_params[layer-conv_layers][4]
	IP = fc_params[layer-conv_layers][5]
	WP = fc_params[layer-conv_layers][6]
	AP = fc_params[layer-conv_layers][7]

	tile = MatrixW * MatrixH // (SIMD*PE)

	# write layer
	outFileConfig.write("\n // Fully Connected Layer L%d \n\n\n" % layer)

	# write config
	outFileConfig.write("#define L%d_MATRIXW %d \n" % (layer, MatrixW))
	outFileConfig.write("#define L%d_MATRIXH %d \n" % (layer, MatrixH))
	outFileConfig.write("#define L%d_SIMD %d \n" % (layer, SIMD))
	outFileConfig.write("#define L%d_PE %d \n" % (layer, PE))
	outFileConfig.write("#define L%d_MMV %d \n" % (layer, MMV))
	outFileConfig.write("#define L%d_WIDTH %d \n" % (layer, WP))

	outFileConfig.write("#define L%d_TILE %d \n" % (layer, tile))

	outFileConfig.write("#define L%d_INPUT_PRECISION %d \n" % (layer, IP))
	outFileConfig.write("#define L%d_ACTIVATION_PRECISION %d \n" % (layer, AP))

	# write weights

	if (WP == 1):
		outFileWeights.write("static BinaryWeights<%d,%d,%d> weights_%d= {\n {\n" %(SIMD,PE,tile,layer))
	else:
		outFileWeights.write("static FixedPointWeights<%d,ap_int<%d>,%d,%d> weights_%d= {\n {\n" %(SIMD,WP,PE,tile,layer))

	for p in range(PE):
		outFileWeights.write("  {\n")
		for t in range(tile):
			width = SIMD*WP;
			val = random.randint(0, 1<<width-1)
			outFileWeights.write("   %s" % hex(val))
			if t!=tile-1:
				outFileWeights.write(",\n")
		outFileWeights.write("\n  }")
		if p!=PE-1:
			outFileWeights.write(",\n")

	outFileWeights.write("\n  }\n };\n")



outFileWeights.write("}\n#endif \n")
outFileWeights.close()
outFileConfig.close()



