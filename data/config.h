
 // Convolutional Layer L0 


#define L0_KERNEL_DIM 3 
#define L0_SIMD 3 
#define L0_PE 2 
#define L0_MMV 1 
#define L0_WIDTH 8 
#define L0_IFM_Channels 3 
#define L0_OFM_Channels 16 
#define L0_IFMDim 10 
#define L0_OFMDim 8 
#define L0_STRIDE 1 
#define L0_INPUT_PRECISION 8 
#define L0_TILE 72 
#define L0_ACTIVATION_PRECISION 8 

 // Convolutional Layer L1 


#define L1_KERNEL_DIM 3 
#define L1_SIMD 2 
#define L1_PE 2 
#define L1_MMV 1 
#define L1_WIDTH 8 
#define L1_IFM_Channels 16 
#define L1_OFM_Channels 16 
#define L1_IFMDim 8 
#define L1_OFMDim 6 
#define L1_STRIDE 1 
#define L1_INPUT_PRECISION 8 
#define L1_TILE 576 
#define L1_ACTIVATION_PRECISION 8 
