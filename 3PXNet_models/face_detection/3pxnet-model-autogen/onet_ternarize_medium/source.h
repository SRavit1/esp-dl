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
#include "xnor_cn.h"
#include "bwn_dense_cn.h"
#include "100.h" 
#include "103.h" 
#include "106.h" 
#include "108.h" 
#include "109.h" 
#include "110.h" 
#include "111.h" 
#include "batchnorm8_running_mean.h" 
#include "batchnorm8_running_var.h" 
#include "batchnorm8_bias.h" 
#include "batchnorm8_weight.h" 
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
#define C1PL 2
static int8_t l1wght[] = _100 ;
#define C2KXY 3
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt]; 
#define C2PD 0
#define C2PL 2
#define C2NPI 143
static uint8_t l2ind[] = _103_indices ;
static pckDtype l2wght[] = _103 ;
#define C3KXY 3
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 32
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt]; 
#define C3PD 0
#define C3PL 1 
#define C3NPI 143
static uint8_t l3ind[] = _106_indices ;
static pckDtype l3wght[] = _106 ;
#define F4I  2048
#define F4NPI  1023
#define F4O  128
static pckDtype l4wght[] = _108 ;
static uint8_t l4ind[] = _108_indices ;
static pckDtype l4act_bin[F4I/pckWdt]; 
#define F5I  128
#define F5NPI  0
#define F5O  2
static pckDtype l5wght[] = _109 ;
static pckDtype l5act_bin[F4O/pckWdt]; 
#define F6I  128
#define F6NPI  0
#define F6O  4
static pckDtype l6wght[] = _110 ;
static pckDtype l6act_bin[F5O/pckWdt]; 
#define F7I  128
#define F7NPI  0
#define F7O  10
static pckDtype l7wght[] = _111 ;
static pckDtype l7act_bin[F6O/pckWdt]; 
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
static bnDtype bn4mean[] = _batchnorm8_running_mean ; 
static bnDtype bn4var[] = _batchnorm8_running_var ; 
static bnDtype bn4gamma[] = _batchnorm8_weight ; 
static bnDtype bn4beta[] = _batchnorm8_bias ; 
/*int main(){ 
	int correct = 0; 
	for(int img = 0; img < 100; img++) {
		uint8_t *curr_im = l1_act + img*48*48*3*sizeof(uint8_t);
		CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
		Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
		Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
		Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset);
		int res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, output, bn2mean, bn2var, bn2gamma, bn2beta);
		int res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn2mean, bn2var, bn2gamma, bn2beta);
		int res = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn2mean, bn2var, bn2gamma, bn2beta);
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
	printf("Accuracy: %f%%\n", 100.0*(float)correct/100); 
	return (EXIT_SUCCESS); 
}*/
