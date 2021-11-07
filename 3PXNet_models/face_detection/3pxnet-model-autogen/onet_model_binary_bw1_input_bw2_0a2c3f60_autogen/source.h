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
#include "85.h" 
#include "88.h" 
#include "91.h" 
#include "94.h" 
#include "96.h" 
#include "97.h" 
#include "98.h" 
#include "99.h" 
#include "bn8_running_mean.h" 
#include "bn8_running_var.h" 
#include "bn8_bias.h" 
#include "bn8_weight.h" 
#include "bn1.h" 
#include "bn2.h" 
#include "bn3.h" 
#include "image.h"
static int8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define C1KXY 3
#define C1XY   48
#define C1Z   3
#define C1KZ 32
#define C1PD 0
#define C1PL 3
static int8_t l1wght[] = _85 ;
#define C2KXY 3
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[2*C2XY*C2XY*C2Z/pckWdt]; 
#define C2PD 0
#define C2PL 3
static pckDtype l2wght[] = _88 ;
#define C3KXY 3
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 32
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt]; 
#define C3PD 0
#define C3PL 1 
static pckDtype l3wght[] = _91 ;
#define C4KXY 2
#define C4XY ((2*C3PD+C3XY-C3KXY+1)/C3PL) 
#define C4Z 32
#define C4KZ 64
static pckDtype l4act_bin[C4XY*C4XY*C4Z/pckWdt]; 
#define C4PD 0
#define C4PL 1 
static pckDtype l4wght[] = _94 ;
#define F5I  64
#define F5NPI  0
#define F5O  128
static pckDtype l5wght[] = _96 ;
static pckDtype l5act_bin[F5I/pckWdt]; 
#define F6I  128
#define F6NPI  0
#define F6O  2
static pckDtype l6wght[] = _97 ;
static pckDtype l6act_bin[F5O/pckWdt]; 
#define F7I  128
#define F7NPI  0
#define F7O  4
static pckDtype l7wght[] = _98 ;
static pckDtype l7act_bin[F6O/pckWdt]; 
#define F8I  128
#define F8NPI  0
#define F8O  10
static pckDtype l8wght[] = _99 ;
static pckDtype l8act_bin[F7O/pckWdt]; 
static float output[10]; 
static pckDtype bn1thr[] = bn1_thresh ; 
static pckDtype bn1sign[] = bn1_sign ; 
static pckDtype bn1offset[] = bn1_offset ; 
static pckDtype bn2thr[] = bn2_thresh ; 
static pckDtype bn2sign[] = bn2_sign ; 
static pckDtype bn2offset[] = bn2_offset ; 
static pckDtype bn3thr[] = bn3_thresh ; 
static pckDtype bn3sign[] = bn3_sign ; 
static pckDtype bn3offset[] = bn3_offset ; 
static bnDtype bn4mean[] = _bn8_running_mean ; 
static bnDtype bn4var[] = _bn8_running_var ; 
static bnDtype bn4gamma[] = _bn8_weight ; 
static bnDtype bn4beta[] = _bn8_bias ; 
/*
int main(){ 
	int correct = 0; 
	for(int img = 0; img < 1; img++) {
		uint8_t *curr_im = l1_act + img*48*48*3*sizeof(uint8_t);
		CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
		CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
		CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
		CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL);
		FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn1thr, bn1sign, bn1offset);
		int res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn2mean, bn2var, bn2gamma, bn2beta);
		int res = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn2mean, bn2var, bn2gamma, bn2beta);
		int res = FcXnorNoBinWrap(l8act_bin, l8wght, F8I, F8O, output, bn2mean, bn2var, bn2gamma, bn2beta);
		float max = -INFINITY; 
		int maxIdx = 0; 
		for (int i = 0; i <10; i++) { 
			 printf("%f, ", output[i]);
			 if (output[i] > max) { 
				 max = output[i]; 
				maxIdx = i;
			 }
		}
		printf("\n");
		printf("Image %d: label: %d, actual: %d\n",img, maxIdx, labels[img]); 
		if (maxIdx == labels[img]) correct += 1; 
	}
	printf("Accuracy: %f%%\n", 100.0*(float)correct/1); 
	return (EXIT_SUCCESS); 
}
*/