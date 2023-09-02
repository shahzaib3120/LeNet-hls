#include <iostream>
#include <cmath>
#include <ctime>
#include <cstring>
#include <hls_stream.h>
#include <cstdlib>
#define AP_INT_MAX_W 8191
#include "ap_int.h"
#include "weights.hpp"
#include "bnn-library.h"
#include "data/memdata.h"
#include "data/config.h"
#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"
#include "conv.hpp"
#include "utils.hpp"
#include "fc_sw.hpp"
#include "pool_tb.hpp"
using namespace hls;
using namespace std;

#define MAX_IMAGES 1
void LeNet(stream<ap_uint<L0_IFM_Channels * L0_INPUT_PRECISION>> &in, stream<ap_uint<L3_PE * L3_ACTIVATION_PRECISION>> &out, unsigned int numReps);

int main()
{
	// create_memdata();
	static ap_uint<L0_INPUT_PRECISION> IMAGE[MAX_IMAGES][L0_IFMDim * L0_IFMDim][L0_IFM_Channels];
	static ap_uint<L0_ACTIVATION_PRECISION> CNV1[MAX_IMAGES][L0_OFMDim][L0_OFMDim][L0_OFM_Channels];
	static ap_uint<L0_ACTIVATION_PRECISION> POOL1[MAX_IMAGES][L0_OFMDim/2][L0_OFMDim/2][L0_OFM_Channels];
	static ap_uint<L1_ACTIVATION_PRECISION> CNV2[MAX_IMAGES][L1_OFMDim][L1_OFMDim][L1_OFM_Channels];
	static ap_uint<L0_ACTIVATION_PRECISION> POOL2[MAX_IMAGES][L1_OFMDim/2][L1_OFMDim/2][L1_OFM_Channels];
	static ap_uint<L2_ACTIVATION_PRECISION> FC1[MAX_IMAGES][L2_MATRIXH];
	static ap_uint<L3_ACTIVATION_PRECISION> FC2[MAX_IMAGES][L3_MATRIXH];
	stream<ap_uint<L0_IFM_Channels * L0_INPUT_PRECISION>> input_stream("input_stream");
	//	stream<ap_uint<L1_OFM_Channels * L1_ACTIVATION_PRECISION>> output_stream("output_stream");
	stream<ap_uint<L3_PE * L3_ACTIVATION_PRECISION>> output_stream("output_stream");

	unsigned int counter = 0;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++)
	{
		for (unsigned int oy = 0; oy < L0_IFMDim; oy++)
		{
			for (unsigned int ox = 0; ox < L0_IFMDim; ox++)
			{
				// value of a pixel across all channels = input_channel
				ap_uint<L0_INPUT_PRECISION * L0_IFM_Channels> input_channel = 0;
				// loop over all channels
				for (unsigned int channel = 0; channel < L0_IFM_Channels; channel++)
				{
					// value of a pixel for single channel = input
					ap_uint<L0_INPUT_PRECISION> input = (ap_uint<L0_INPUT_PRECISION>)(counter);
					// store respective pixel and channel in image
					IMAGE[n_image][oy * L0_IFMDim + ox][channel] = input;
					// shift the whole channel value to lower 8 bits
					input_channel = input_channel >> L0_INPUT_PRECISION;
					// write the current channel value in MSB of input_channel (31,24)
					input_channel(L0_IFM_Channels * L0_INPUT_PRECISION - 1, (L0_IFM_Channels - 1) * L0_INPUT_PRECISION) = input;
					counter++;
				}
				input_stream.write(input_channel);
			}
		}
	}
	// initialize the weights
	// cnv weights
	// template <unsigned int WIDTH, unsigned int IFMCh, unsigned int OFMCh, unsigned int KDim, unsigned int SIMD, unsigned int PE, typename TW>
	// void loadCNVWeights(ap_uint<WIDTH> weights[OFMCh][KDim][KDim][IFMCh], TW const &weights_in)

	// lfc weights
	// template <unsigned int WIDTH, unsigned int MatrixW, unsigned int MatrixH, unsigned int SIMD, unsigned int PE, typename TW>
	// void loadFCWeights(ap_uint<WIDTH> weights[MatrixH][MatrixW], TW const &weights_in)
	static ap_uint<L0_WIDTH> W1[L0_OFM_Channels][L0_KERNEL_DIM][L0_KERNEL_DIM][L0_IFM_Channels];
	static ap_uint<L1_WIDTH> W2[L1_OFM_Channels][L1_KERNEL_DIM][L1_KERNEL_DIM][L1_IFM_Channels];
	static ap_uint<L2_WIDTH> W3[L2_MATRIXH][L2_MATRIXW];
	static ap_uint<L3_WIDTH> W4[L3_MATRIXH][L3_MATRIXW];

	loadCNVWeights<L0_WIDTH, L0_IFM_Channels, L0_OFM_Channels, L0_KERNEL_DIM, L0_SIMD, L0_PE>(W1, PARAM::weights_0);
	loadCNVWeights<L1_WIDTH, L1_IFM_Channels, L1_OFM_Channels, L1_KERNEL_DIM, L1_SIMD, L1_PE>(W2, PARAM::weights_1);
	loadFCWeights<L2_WIDTH, L2_MATRIXW, L2_MATRIXH, L2_SIMD, L2_PE>(W3, PARAM::weights_2);
	loadFCWeights<L3_WIDTH, L3_MATRIXW, L3_MATRIXH, L3_SIMD, L3_PE>(W4, PARAM::weights_3);

	conv<MAX_IMAGES, L0_IFMDim, L0_OFMDim, L0_IFM_Channels, L0_OFM_Channels, L0_KERNEL_DIM, 1, ap_uint<L0_INPUT_PRECISION>>(IMAGE, W1, CNV1);

	// max pool
//	template<int MAX_IMAGE,
//		int IFMDim,
//		int OFMDim,
//		int FMCh,
//		int kernel,
//		int stride,
//		typename TI>
//		void pool(TI const img[MAX_IMAGE][IFMDim][IFMDim][FMCh], TI out[MAX_IMAGE][OFMDim][OFMDim][FMCh])

	pool<MAX_IMAGES, L0_OFMDim, L0_OFMDim/2, L0_OFM_Channels, 2, 2, ap_uint<L0_ACTIVATION_PRECISION>>(CNV1, POOL1);

	conv_inter<MAX_IMAGES, L1_IFMDim, L1_OFMDim, L1_IFM_Channels, L1_OFM_Channels, L1_KERNEL_DIM, 1, ap_uint<L1_ACTIVATION_PRECISION>>(POOL1, W2, CNV2);

	pool<MAX_IMAGES, L1_OFMDim, L1_OFMDim/2, L1_OFM_Channels, 2, 2, ap_uint<L1_ACTIVATION_PRECISION>>(CNV2, POOL2);

	static ap_uint<L2_INPUT_PRECISION> IMAGE_FC[MAX_IMAGES][L2_MATRIXW];
	// flatten the output of conv layer

	// template <unsigned int MAX_IMAGES, unsigned int IFMDim, unsigned int IFMChannels, typename TI>
	// void flatten(TI input[MAX_IMAGES][IFMChannels][IFMDim][IFMDim], TI output[MAX_IMAGES][IFMChannels * IFMDim * IFMDim])

	flatten<MAX_IMAGES, L1_OFMDim/2, L1_OFM_Channels, ap_uint<L1_ACTIVATION_PRECISION>>(POOL2, IMAGE_FC);

	// fully connected layers

	// template <int MAX_IMAGE,
	//       int MATRIXW,
	//       int MATRIXH,
	//       typename TI,
	//       typename TO,
	//       typename TW>
	// void fc_sw(TI const input[MAX_IMAGE][MATRIXW], TW const weights[MATRIXH][MATRIXW], TO output[MAX_IMAGE][MATRIXH])

	fc_sw<MAX_IMAGES, L2_MATRIXW, L2_MATRIXH, ap_uint<L2_INPUT_PRECISION>, ap_uint<L2_ACTIVATION_PRECISION>, ap_uint<L2_WIDTH>>(IMAGE_FC, W3, FC1);
	fc_sw<MAX_IMAGES, L3_MATRIXW, L3_MATRIXH, ap_uint<L3_INPUT_PRECISION>, ap_uint<L3_ACTIVATION_PRECISION>, ap_uint<L2_WIDTH>>(FC1, W4, FC2);

	LeNet(input_stream, output_stream, MAX_IMAGES);
	bool error = false;

	unsigned int bit = 0;
	unsigned int output_reads = 0;
	// if(output_stream.empty()){
	// 	cout << "Output Stream was empty !!" << endl;
	// 	return 1;
	// }
	while (!output_stream.empty())
	{
		auto output = output_stream.read();
		for (unsigned int i = 0; i < L3_PE; i++)
		{
			auto outElem = output((L3_ACTIVATION_PRECISION) * (i + 1) - 1, i * L3_ACTIVATION_PRECISION);
			if (outElem != FC2[0][bit])
			{
				error = true;
				cout << "ERROR: ";
			}
			cout << "output[" << bit << "] = " << outElem;
			cout << "\t expected[" << bit << "] = " << FC2[0][bit] << endl;
			bit++;
		}
		output_reads++;
	}
	cout << "Output Reads = " << output_reads << endl;
	if (error)
	{
		cout << "Test failed!" << endl;
		return 1;
	}
	else
	{
		cout << "Test passed!" << endl;
	}
	return 0;
}
