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

void Testbench_conv(stream<ap_uint<IFM_Channels1*INPUT_PRECISION> > & in, stream<ap_uint<OFM_Channels1*ACTIVATION_PRECISION> > & out, unsigned int numReps){
#pragma HLS DATAFLOW
	ConvLayer_Batch<KERNEL_DIM, IFM_Channels1, IFMDim1, OFM_Channels1, OFMDim1, SIMD1, PE1, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_uint<ACTIVATION_PRECISION> >, Identity >(in, out, PARAM::weights, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_dsp());
}
