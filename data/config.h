
 // Convolutional Layer L0 


#define L0_KERNEL_DIM 3 
#define L0_SIMD 3 
#define L0_PE 8 
#define L0_MMV 1 
#define L0_WIDTH 1 
#define L0_IFM_Channels 3 
#define L0_OFM_Channels 16 
#define L0_IFMDim 10 
#define L0_OFMDim 8 
#define L0_STRIDE 1 
#define L0_INPUT_PRECISION 8 
#define L0_TILE 18 
#define L0_ACTIVATION_PRECISION 8 

 // Convolutional Layer L1 


#define L1_KERNEL_DIM 3 
#define L1_SIMD 8 
#define L1_PE 8 
#define L1_MMV 1 
#define L1_WIDTH 1 
#define L1_IFM_Channels 16 
#define L1_OFM_Channels 16 
#define L1_IFMDim 4 
#define L1_OFMDim 2 
#define L1_STRIDE 1 
#define L1_INPUT_PRECISION 8 
#define L1_TILE 36 
#define L1_ACTIVATION_PRECISION 8 

 // Fully Connected Layer L2 


#define L2_MATRIXW 16 
#define L2_MATRIXH 64 
#define L2_SIMD 16 
#define L2_PE 8 
#define L2_MMV 1 
#define L2_WIDTH 1 
#define L2_TILE 8 
#define L2_INPUT_PRECISION 8 
#define L2_ACTIVATION_PRECISION 8 

 // Fully Connected Layer L3 


#define L3_MATRIXW 64 
#define L3_MATRIXH 10 
#define L3_SIMD 16 
#define L3_PE 10 
#define L3_MMV 1 
#define L3_WIDTH 1 
#define L3_TILE 4 
#define L3_INPUT_PRECISION 8 
#define L3_ACTIVATION_PRECISION 8 
