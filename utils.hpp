#ifndef UTILS_FC
#define UTILS_FC

#include <ap_int.h>
using namespace std;

// write a function to load weights
template <unsigned int WIDTH, unsigned int MatrixW, unsigned int MatrixH, unsigned int SIMD, unsigned int PE, typename TW>
void loadFCWeights(ap_uint<WIDTH> weights[MatrixH][MatrixW], TW const &weights_in)
{
    const int TX = MatrixW / SIMD;
    const int TY = MatrixH / PE;
    int kx = 0;
    int ky = 0;

    for (unsigned int oy = 0; oy < TY; oy++)
    {
        for (unsigned int pe = 0; pe < PE; pe++)
        {
            for (unsigned int ox = 0; ox < TX; ox++)
            {
                for (unsigned int simd = 0; simd < SIMD; simd++)
                {
                    weights[ky][kx] = weights_in.weights(oy * TX + ox)[pe][simd];
                    // cout << "TILE " << oy*TX + ox << " PE " << pe << " SIMD " << simd << endl;
                    if (kx == MatrixW - 1)
                    {
                        kx = 0;
                        ky++;
                    }
                    else
                    {
                        kx++;
                    }
                }
            }
        }
    }
}

template <unsigned int WIDTH, unsigned int IFMCh, unsigned int OFMCh, unsigned int KDim, unsigned int SIMD, unsigned int PE, typename TW>
void loadCNVWeights(ap_uint<WIDTH> weights[OFMCh][KDim][KDim][IFMCh], TW const &weights_in)
{
    // load cnv weights
    int TX = (IFMCh * KDim * KDim) / SIMD; // iterations required for kernel
    int TY = OFMCh / PE;                   // outputs
    unsigned int kx = 0;
    unsigned int ky = 0;
    unsigned int chan_count = 0;
    unsigned int out_chan_count = 0;

    for (unsigned int oy = 0; oy < TY; oy++)
    {
        for (unsigned int pe = 0; pe < PE; pe++)
        {
            for (unsigned int ox = 0; ox < TX; ox++)
            {
                for (unsigned int simd = 0; simd < SIMD; simd++)
                {
                    weights[out_chan_count][kx][ky][chan_count] = weights_in.weights(oy * TX + ox)[pe][simd];
                    // cout << "TILE " << oy*TX + ox << " PE " << pe << " SIMD " << simd << endl;
                    // cout << "IFM " << chan_count << " KX " << kx << " KY " << ky << " OFM " << out_chan_count << endl;
                    chan_count++;
                    if (chan_count == IFMCh)
                    {
                        chan_count = 0;
                        kx++;
                        if (kx == KDim)
                        {
                            kx = 0;
                            ky++;
                            if (ky == KDim)
                            {
                                ky = 0;
                                out_chan_count++;
                                if (out_chan_count == OFMCh)
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
}

template <unsigned int MAX_IMAGES, unsigned int MatrixH, typename TI>
void printResult(TI matrix[MAX_IMAGES][MatrixH])
{
    for (int i = 0; i < MAX_IMAGES; i++)
    {
        for (int j = 0; j < MatrixH; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// flatten the input image
template <unsigned int MAX_IMAGES, unsigned int IFMDim, unsigned int IFMChannels, typename TI>
void flatten(TI input[MAX_IMAGES][IFMDim][IFMDim][IFMChannels], TI output[MAX_IMAGES][IFMChannels * IFMDim * IFMDim])
{
    for (int n = 0; n < MAX_IMAGES; n++)
    {
        for (int h = 0; h < IFMDim; h++)
        {
            for (int w = 0; w < IFMDim; w++)
            {
                for (int c = 0; c < IFMChannels; c++)
                {
                    output[n][h * IFMDim * IFMChannels + w * IFMChannels + c] = input[n][w][h][c];
                }
            }
        }
    }
}
#endif
