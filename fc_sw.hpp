#ifndef FC_SW
#define FC_SW

// write a function that takes in input stream, weights, and output stream
// and performs the matrix vector activation

// MATRIXH is the number of output neurons
// MATRIXW is the number of input neurons
// output shape of single matrix vector activation is MATRIXH
// input shape of single matrix vector activation is MATRIXW
template <int MAX_IMAGE,
          int MATRIXW,
          int MATRIXH,
          typename TI,
          typename TO,
          typename TW>
void fc_sw(TI const input[MAX_IMAGE][MATRIXW], TW const weights[MATRIXH][MATRIXW], TO output[MAX_IMAGE][MATRIXH])
{
    for (int n = 0; n < MAX_IMAGE; n++)
        for (int h = 0; h < MATRIXH; h++)
        {
            TO tmp = 0;
            for (int w = 0; w < MATRIXW; w++)
                tmp += input[n][w] * weights[h][w];
            output[n][h] = tmp;
        }
}

#endif
