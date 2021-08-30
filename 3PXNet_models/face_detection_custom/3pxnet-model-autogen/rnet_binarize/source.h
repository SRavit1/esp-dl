#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include "datatypes.h"
#include "utils.h"
#include "xnor_base.h"
#include "xnor_fc.h"
#include "3pxnet_fc.h"
#include "3pxnet_cn.h"
#include "xnor_fc.h"
#include "bwn_dense_cn.h"
#include "conv1_weight.h" 
#include "conv2_weight.h" 
#include "conv3_weight.h" 
#include "30.h" 
#include "31.h" 
#include "32.h" 
#include "image.h"
static uint8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define C1KXY 3
#define C1XY   24
#define C1Z   3
#define C1KZ 32
#define C1PD 0
#define C1PL 2
static int8_t l1wght[] = _conv1_weight ;
#define C2KXY 3
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt]; 
#define C2PD 0
#define C2PL 3
static pckDtype l2wght[] = _conv2_weight ;
#define C3KXY 2
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 64
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt]; 
#define C3PD 0
#define C3PL 1 
static pckDtype l3wght[] = _conv3_weight ;
#define F4I  256
#define F4NPI  0
#define F4O  128
static pckDtype l4wght[] = _30 ;
static pckDtype l4act_bin[F4I/pckWdt]; 
#define F5I  128
#define F5NPI  0
#define F5O  2
static pckDtype l5wght[] = _31 ;
static pckDtype l5act_bin[F4O/pckWdt]; 
#define F6I  128
#define F6NPI  0
#define F6O  4
static pckDtype l6wght[] = _32 ;
static pckDtype l6act_bin[F5O/pckWdt]; 
static float output[10]; 
/*int main(){ 
	int correct = 0; 
	for(int img = 0; img < 100; img++) {
		uint8_t *curr_im = l1_act + img*24*24*3*sizeof(uint8_t);
		CnBnBwn(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL)
		CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL);
		CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL);
		FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, NULL, NULL);
		FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, NULL, NULL);
		float max = -INFINITY; 
		int maxIdx = 0; 
		for (int i = 0; i <10; i++) { 
			 printf("%f, ", output[i]);
			 if (output[i] > max) { 
				 max = output[i]; 
				maxIdx = i;
			 }
		}
		printf("\n");printf("Image %d: label: %d, actual: %d\n",img, maxIdx, labels[img]); 
		if (maxIdx == labels[img]) correct += 1; 
	}
	printf("Accuracy: %f%%\n", 100.0*(float)correct/100); 
	return (EXIT_SUCCESS); 
}*/