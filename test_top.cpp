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
using namespace hls;
using namespace std;

#define MAX_IMAGES 1
void Testbench_conv(stream<ap_uint<L0_IFM_Channels * L0_INPUT_PRECISION>> &in, stream<ap_uint<L1_OFM_Channels * L1_ACTIVATION_PRECISION>> &out, unsigned int numReps);

int main()
{
	// create_memdata();
	static ap_uint<L0_INPUT_PRECISION> IMAGE[MAX_IMAGES][L0_IFMDim * L0_IFMDim][L0_IFM_Channels];
	static ap_int<L0_ACTIVATION_PRECISION> TEST_INTER[MAX_IMAGES][L0_OFMDim][L0_OFMDim][L0_OFM_Channels];
	static ap_int<L1_ACTIVATION_PRECISION> TEST[MAX_IMAGES][L1_OFMDim][L1_OFMDim][L1_OFM_Channels];
	stream<ap_uint<L0_IFM_Channels * L0_INPUT_PRECISION>> input_stream("input_stream");
	stream<ap_uint<L1_OFM_Channels * L1_ACTIVATION_PRECISION>> output_stream("output_stream");
	unsigned int counter = 0;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++)
	{
		for (unsigned int oy = 0; oy < L0_IFMDim; oy++)
		{
			for (unsigned int ox = 0; ox < L0_IFMDim; ox++)
			{
				// value of a pixel across all channels = input_channel
				ap_uint<L0_INPUT_PRECISION *L0_IFM_Channels> input_channel = 0;
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
	static ap_uint<L0_WIDTH> W1[L0_OFM_Channels][L0_KERNEL_DIM][L0_KERNEL_DIM][L0_IFM_Channels];
	// initialize the weights
	int TX = (L0_IFM_Channels * L0_KERNEL_DIM * L0_KERNEL_DIM) / L0_SIMD; // iterations required for kernel
	int TY = L0_OFM_Channels / L0_PE;									  // outputs
	unsigned int kx = 0;
	unsigned int ky = 0;
	unsigned int chan_count = 0;
	unsigned int out_chan_count = 0;

	for (unsigned int oy = 0; oy < TY; oy++)
	{
		for (unsigned int pe = 0; pe < L0_PE; pe++)
		{
			for (unsigned int ox = 0; ox < TX; ox++)
			{
				for (unsigned int simd = 0; simd < L0_SIMD; simd++)
				{
					W1[out_chan_count][kx][ky][chan_count] = PARAM::weights_0.weights(oy * TX + ox)[pe][simd];
					// cout << "TILE " << oy*TX + ox << " PE " << pe << " SIMD " << simd << endl;
					// cout << "IFM " << chan_count << " KX " << kx << " KY " << ky << " OFM " << out_chan_count << endl;
					chan_count++;
					if (chan_count == L0_IFM_Channels)
					{
						chan_count = 0;
						kx++;
						if (kx == L0_KERNEL_DIM)
						{
							kx = 0;
							ky++;
							if (ky == L0_KERNEL_DIM)
							{
								ky = 0;
								out_chan_count++;
								if (out_chan_count == L0_OFM_Channels)
								{
									out_chan_count = 0;
								}
							}
						}
					}
				}
			}
		}
	}

	static ap_uint<L0_WIDTH> W2[L1_OFM_Channels][L1_KERNEL_DIM][L1_KERNEL_DIM][L1_IFM_Channels];
	// initialize the weights
	TX = (L1_IFM_Channels * L1_KERNEL_DIM * L1_KERNEL_DIM) / L1_SIMD; // iterations required for kernel
	TY = L1_OFM_Channels / L1_PE;									  // outputs
	kx = 0;
	ky = 0;
	chan_count = 0;
	out_chan_count = 0;

	for (unsigned int oy = 0; oy < TY; oy++)
	{
		for (unsigned int pe = 0; pe < L1_PE; pe++)
		{
			for (unsigned int ox = 0; ox < TX; ox++)
			{
				for (unsigned int simd = 0; simd < L1_SIMD; simd++)
				{
					W2[out_chan_count][kx][ky][chan_count] = PARAM::weights_1.weights(oy * TX + ox)[pe][simd];
					// cout << "TILE " << oy*TX + ox << " PE " << pe << " SIMD " << simd << endl;
					// cout << "IFM " << chan_count << " KX " << kx << " KY " << ky << " OFM " << out_chan_count << endl;
					chan_count++;
					if (chan_count == L1_IFM_Channels)
					{
						chan_count = 0;
						kx++;
						if (kx == L1_KERNEL_DIM)
						{
							kx = 0;
							ky++;
							if (ky == L1_KERNEL_DIM)
							{
								ky = 0;
								out_chan_count++;
								if (out_chan_count == L1_OFM_Channels)
								{
									out_chan_count = 0;
								}
							}
						}
					}
				}
			}
		}
	}

	conv<MAX_IMAGES, L0_IFMDim, L0_OFMDim, L0_IFM_Channels, L0_OFM_Channels, L0_KERNEL_DIM, 1, ap_uint<L0_INPUT_PRECISION>>(IMAGE, W1, TEST_INTER);
	// test_inter = [MAX_IMAGES][L0_OFMDim][L0_OFMDim][L0_OFM_Channels]
	// required= 	[MAX_IMAGE][IFMDim * IFMDim][IFMCh]

	conv_inter<MAX_IMAGES, L1_IFMDim, L1_OFMDim, L1_IFM_Channels, L1_OFM_Channels, L1_KERNEL_DIM, 1, ap_int<L1_ACTIVATION_PRECISION>>(TEST_INTER, W2, TEST);
	Testbench_conv(input_stream, output_stream, MAX_IMAGES);
	int err_counter = 0, err_perimage = 0;
	ap_int<L1_ACTIVATION_PRECISION> out_chan;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++)
	{
		for (unsigned int oy = 0; oy < L1_OFMDim; oy++)
		{
			for (unsigned int ox = 0; ox < L1_OFMDim; ox++)
			{
				for (int e = 0; e < 1; e++)
				{
					ap_uint<L1_OFM_Channels *L1_ACTIVATION_PRECISION> outElem = output_stream.read();
					for (unsigned int channel = 0; channel < L1_OFM_Channels; channel++)
					{
						ap_int<L1_ACTIVATION_PRECISION> EXP = TEST[n_image][ox][oy][channel + e * L1_OFM_Channels];
						out_chan(L1_ACTIVATION_PRECISION - 1, 0) = outElem((channel + 1) * L1_ACTIVATION_PRECISION - 1, channel * L1_ACTIVATION_PRECISION);

						if (EXP != out_chan)
						{
							std::cout << "ERROR: Expected[" << oy << "][" << ox << "][" << channel << "]=" << EXP << " actual " << out_chan << std::endl;
							err_counter++;
							err_perimage++;
						}
						else
						{
							std::cout << "Expected[" << oy << "][" << ox << "][" << channel << "]=" << EXP << " actual " << out_chan << std::endl;
						}
					}
				}
			}
		}
		if (err_perimage == 0)
		{
			std::cout << "Image # " << n_image << " passed the testing." << std::endl;
		}
		else
		{
			err_perimage = 0;
			std::cout << "Image # " << n_image << " failed the testing." << std::endl;
		}
	}
	if (err_counter == 0)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}
