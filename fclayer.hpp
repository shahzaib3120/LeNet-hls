// imported from BNN-PYNQ

#ifndef FCLAYER_H
#define FCLAYER_H

#include <ap_int.h>
#include <hls_stream.h>

#include "streamtools.h"
#include "mvau.hpp"

using namespace std;

template <
    unsigned int MatrixW, unsigned int MatrixH, // geometry must be specified
    unsigned int SIMD, unsigned int PE, unsigned int MMV,

    typename TSrcI = Identity, // redefine I/O interpretation as needed
    typename TDstI = Identity,
    typename TWeightI = Identity, // redefine I/O interpretation as needed for weigths

    int InStreamW, int OutStreamW, // safely deducible (stream width must be int though!)
    typename TW, typename TA, typename R>
void StreamingFCLayer_Batch(hls::stream<ap_uint<InStreamW>> &in,
                            hls::stream<ap_uint<OutStreamW>> &out,
                            TW const &weights,
                            TA const &activation,
                            unsigned const reps,
                            R const &r)
{
#pragma HLS INLINE
  unsigned const InpPerImage = MatrixW / InStreamW * TSrcI::width;
  unsigned const OutPerImage = MatrixH / PE;

  WidthAdjustedInputStream<InStreamW, SIMD * TSrcI::width, InpPerImage> wa_in(in, reps);
  WidthAdjustedOutputStream<PE * TDstI::width, OutStreamW, OutPerImage> wa_out(out, reps);

//  hls::stream<ap_uint<SIMD * TSrcI::width>>& adjustedStream = wa_in;
//  while(!adjustedStream.empty()){
//	  cout << "adj stream = " << adjustedStream.read() << endl;
//  }
  cout << "Adjusted" << endl;

  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, MMV, TSrcI, TDstI, TWeightI>(static_cast<hls::stream<ap_uint<SIMD * TSrcI::width>> &>(wa_in),
                                                                                        static_cast<hls::stream<ap_uint<PE * TDstI::width>> &>(wa_out),
																						weights, activation, reps, r);

  cout << "Matrix activation" << endl;
}

#endif
