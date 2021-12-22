#include <stdio.h>
#include "3pxnet_util.h"

/*
 * input - pckDtype array of n*btwdt/pckWdith elements
 * output - int array of n elements
 * n - number of elements in output
 */
/*
int pckDtype_to_int(pckDtype* input, int* output, int n, int pckWidth, int btwdt) {
	for (int i = 0; i < n; i++) {
		output[i] = 0;
		int pckVal = 0;
		int placeVal = 1;
		for (int j = 0; j < btwdt; j++) {
			int pckVal = input[i/(pckWidth/btwdt)] & (0b1 << (pckWidth - (i % (pckWidth/btwdt))*btwdt + j));
			output[i] += (pckVal == 0b0) ? placeVal*-1 : placeVal;
			placeVal *= 2;
		}
	}
	return 0;
}
*/

/*
 * input - pckDtype array of n*btwdt/pckWdith elements
 * output - int array of n elements
 * n - number of elements in output
 */
int pckDtype_to_int(pckDtype* input, int* output, int n, int pckWidth, int btwdt) {
	for (int i = 0; i < n; i++) {
		output[i] = 0;
		int placeVal = 1;
		for (int j = btwdt-1; j >= 0; j--) {
			int pckVal = input[(i/pckWidth)*btwdt+j] & (0b1 << (pckWidth -1 - (i % pckWidth)));
			output[i] += (pckVal == 0b0) ? placeVal*-1 : placeVal;
			placeVal *= 2;
		}
	}
	return 0;
}

int int_to_pckDtype(int* input, pckDtype* output, int n, int pckWidth, int btwdt) {
	return 0;
}

int print_int_array(int* input, int n) {
	for (int i = 0; i < n; i++)
		printf("%d, ", input[i]);
	printf("\n");
	return 0;
}
