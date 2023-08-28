#include <hls_stream.h>
using namespace hls;
#include "ap_int.h"
#include "bnn-library.h"
#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"
#include "conv.hpp"
#include "data/memdata.h"
#include "data/config.h"

void Testbench_conv(stream<ap_uint<L0_IFM_Channels * L0_INPUT_PRECISION>> &in, stream<ap_uint<L1_OFM_Channels * L1_ACTIVATION_PRECISION>> &out, unsigned int numReps)
{
	stream<ap_uint<L0_OFM_Channels * L0_ACTIVATION_PRECISION>> inter_1("CNV1");
#pragma HLS DATAFLOW
	ConvLayer_Batch<L0_KERNEL_DIM, L0_IFM_Channels, L0_IFMDim, L0_OFM_Channels, L0_OFMDim, L0_SIMD, L0_PE, Slice<ap_uint<L0_INPUT_PRECISION>>, Slice<ap_uint<L0_ACTIVATION_PRECISION>>, Identity>(in, inter_1, PARAM::weights_0, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_dsp());
	ConvLayer_Batch<L1_KERNEL_DIM, L1_IFM_Channels, L1_IFMDim, L1_OFM_Channels, L1_OFMDim, L1_SIMD, L1_PE, Slice<ap_uint<L1_INPUT_PRECISION>>, Slice<ap_uint<L0_ACTIVATION_PRECISION>>, Identity>(inter_1, out, PARAM::weights_1, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_dsp());
}
