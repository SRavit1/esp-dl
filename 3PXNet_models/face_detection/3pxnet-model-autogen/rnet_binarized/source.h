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
#include "69.h" 
#include "72.h" 
#include "75.h" 
#include "77.h" 
#include "78.h" 
#include "79.h" 
#include "bn6_running_mean.h" 
#include "bn6_running_var.h" 
#include "bn6_bias.h" 
#include "bn6_weight.h" 
#include "bn1.h" 
#include "bn2.h" 
#include "image.h"
static int8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define C1KXY 3
#define C1XY   24
#define C1Z   3
#define C1KZ 32
#define C1PD 0
#define C1PL 2
static int8_t l1wght[] = _69 ;
#define C2KXY 3
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt]; 
#define C2PD 0
#define C2PL 2
static pckDtype l2wght[] = _72 ;
#define C3KXY 3
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 64
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt]; 
#define C3PD 0
#define C3PL 2
static pckDtype l3wght[] = _75 ;
#define F4I  64
#define F4NPI  0
#define F4O  128
static pckDtype l4wght[] = _77 ;
static pckDtype l4act_bin[F4I/pckWdt]; 
#define F5I  128
#define F5NPI  0
#define F5O  2
static pckDtype l5wght[] = _78 ;
static pckDtype l5act_bin[F4O/pckWdt]; 
#define F6I  128
#define F6NPI  0
#define F6O  4
static pckDtype l6wght[] = _79 ;
static pckDtype l6act_bin[F5O/pckWdt]; 
static float output[10]; 
static pckDtype bn1thr[] = bn1_thresh ; 
static pckDtype bn1sign[] = bn1_sign ; 
static pckDtype bn1offset[] = bn1_offset ; 
static pckDtype bn2thr[] = bn2_thresh ; 
static pckDtype bn2sign[] = bn2_sign ; 
static pckDtype bn2offset[] = bn2_offset ; 
static bnDtype bn3mean[] = _bn6_running_mean ; 
static bnDtype bn3var[] = _bn6_running_var ; 
static bnDtype bn3gamma[] = _bn6_weight ; 
static bnDtype bn3beta[] = _bn6_bias ; 
/*int main(){ 
	int correct = 0; 
	for(int img = 0; img < 1; img++) {
		uint8_t *curr_im = l1_act + img*24*24*3*sizeof(uint8_t);
		CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
		CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
		CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
		FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset);
		FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn2thr, bn2sign, bn2offset);
		int res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn3mean, bn3var, bn3gamma, bn3beta);
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
	printf("Accuracy: %f%%\n", 100.0*(float)correct/1); 
	return (EXIT_SUCCESS); 
}*/
