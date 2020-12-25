#include <math.h>					//Required for cos and sin functions
typedef float IN_TYPE;		// Data type for the input signal
typedef float TEMP_TYPE; // Data type for the temporary variables
#define N 32				// DFT Size

TEMP_TYPE cos_table[N];
TEMP_TYPE sin_table[N];


//void dft(IN_TYPE sample_real[N], IN_TYPE sample_imag[N]) {
//	int i, j;
//	TEMP_TYPE w;
//	TEMP_TYPE c, s;
//
//	// Temporary arrays to hold the intermediate frequency domain results
//	TEMP_TYPE temp_real[N];
//	TEMP_TYPE temp_imag[N];
//
//	// Calculate each frequency domain sample iteratively
//	L1:for (i = 0; i < N; i += 1) {
//		temp_real[i] = 0;
//		temp_imag[i] = 0;
//
//		// (2 * pi * i)/N
//		w = (2.0 * 3.141592653589  / N) * (TEMP_TYPE)i;
//
//		// Calculate the jth frequency sample sequentially
//		L2:for (j = 0; j < N; j += 1) {
//#pragma HLS pipeline
//			// Utilize HLS tool to calculate sine and cosine values
//			c = cos(j * w);
//			s = -sin(j * w);
//
//			// Multiply the current phasor with the appropriate input sample and keep
//			// running sum
//			TEMP_TYPE rj = sample_real[j];
//			TEMP_TYPE imj = sample_imag[j];
//			temp_real[i] += (rj * c - imj * s);
//			temp_imag[i] += (rj * s + imj * c);
//		}
//	}
//
//	// Perform an inplace DFT, i.e., copy result into the input arrays
//	for (i = 0; i < N; i += 1) {
//#pragma HLS pipeline
//		sample_real[i] = temp_real[i];
//		sample_imag[i] = temp_imag[i];
//	}
//}

void dft(IN_TYPE sample_real[N], IN_TYPE sample_imag[N]) {
	int i, j;
	TEMP_TYPE w;
	TEMP_TYPE c, s;

	// Temporary arrays to hold the intermediate frequency domain results
	TEMP_TYPE temp_real[N];
	TEMP_TYPE temp_imag[N];

	// Calculate each frequency domain sample iteratively
	L1:for (i = 0; i < N; i += 1) {
		temp_real[i] = 0;
		temp_imag[i] = 0;

		TEMP_TYPE tmp_real[N], tmp_img[N];
#pragma HLS ARRAY_PARTITION variable=tmp_real complete
#pragma HLS ARRAY_PARTITION variable=tmp_img complete

		// Calculate the jth frequency sample sequentially
		L2:for (j = 0; j < N; j += 1) {
#pragma HLS pipeline
			c = cos_table[i * j % N];
			s = sin_table[i * j % N];

			// Multiply the current phasor with the appropriate input sample and keep
			// running sum
			TEMP_TYPE rj = sample_real[j];
			TEMP_TYPE imj = sample_imag[j];
			tmp_real[j] = (rj * c - imj * s);
			tmp_img[j] = (rj * s + imj * c);
		}
		TEMP_TYPE t1=0, t2=0;
		L3:for (j = 0; j < N; j += 1) {
//#pragma HLS pipeline
#pragma HLS unroll
			 t1 += tmp_real[j];
			 t2 += tmp_img[j];
		}
		temp_real[i] = t1;
		temp_imag[i] = t2;
	}

	// Perform an inplace DFT, i.e., copy result into the input arrays
	for (i = 0; i < N; i += 1) {
#pragma HLS pipeline
		sample_real[i] = temp_real[i];
		sample_imag[i] = temp_imag[i];
	}
}
