#include "spmv.h"

//void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
//		DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE])
//{
//L1: for (int i = 0; i < NUM_ROWS; i++) {
//		DTYPE y0 = 0;
//	L2: for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {
//#pragma HLS loop_tripcount min=1 max=4 avg=2
////#pragma HLS unroll factor=4
//#pragma HLS pipeline
//			y0 += values[k] * x[columnIndex[k]];
//		}
//		y[i] = y0;
//	}
//}

void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
		DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE])
{
#pragma HLS ARRAY_PARTITION variable=rowPtr cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=columnIndex cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=values cyclic factor=4 dim=1

L1: for (int i = 0; i < NUM_ROWS; i++) {
	DTYPE yt[SIZE];
#pragma HLS ARRAY_PARTITION variable=yt complete dim=1
	DTYPE y0 = 0;
	L2_1: for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {
#pragma HLS loop_tripcount min=1 max=4 avg=2
#pragma HLS unroll factor=4
#pragma HLS pipeline
			yt[k] = values[k] * x[columnIndex[k]];
		}
	L2_2: for (int k = 0; k < SIZE; k++) {
#pragma HLS unroll
			y0 += yt[k];
		}
		y[i] = y0;
	}
}

//const static int S = 4;
//
//void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
//          DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE])
//{
//#pragma HLS ARRAY_PARTITION variable=rowPtr block factor=4 dim=1
//#pragma HLS ARRAY_PARTITION variable=columnIndex block factor=4 dim=1
//#pragma HLS ARRAY_PARTITION variable=values block factor=4 dim=1
//#pragma HLS ARRAY_PARTITION variable=y block factor=4 dim=1
//  L1: for (int i = 0; i < NUM_ROWS; i++) {
//	  DTYPE y0 = 0;
//    L2_1: for (int k = rowPtr[i]; k < rowPtr[i+1]; k += S) {
//#pragma HLS loop_tripcount min=1 max=4 avg=2
//#pragma HLS pipeline II=S
//
//    	DTYPE yt[S];
//	  L2_2: for(int j = 0; j < S; j++) {
//			  if(k+j < rowPtr[i+1]) {
//				  yt[j] = values[k+j] * x[columnIndex[k+j]];
//				  y0 += yt[j];
//			  }
//			  else
//			  {
//				  yt[j] = 0;
//			  }
//		  }
//	  }
//    y[i] = y0;
//  }
//}


//void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
//          DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE])
//{
//#pragma HLS ARRAY_PARTITION variable=rowPtr cyclic factor=2 dim=1
//#pragma HLS ARRAY_PARTITION variable=columnIndex cyclic factor=2 dim=1
//#pragma HLS ARRAY_PARTITION variable=values cyclic factor=2 dim=1
//  L1: for (int i = 0; i < NUM_ROWS; i++) {
//	  DTYPE y0 = 0;
//    L2_1: for (int k = rowPtr[i]; k < rowPtr[i+1]; k += 2) {
//#pragma HLS loop_tripcount min=1 max=4 avg=2
//#pragma HLS pipeline
//    	DTYPE yt1, yt2=0;
//		  yt1 = values[k] * x[columnIndex[k]];
//		  if(k+1 < rowPtr[i+1])
//			  yt2 = values[k+1] * x[columnIndex[k+1]];
//		  y0 += yt1 + yt2;
//	  }
//    y[i] = y0;
//  }
//}
