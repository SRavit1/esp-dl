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
#include "61.h" 
#include "conv2_weight.h" 
#include "conv3_weight.h" 
#include "conv4_weight.h" 
#include "conv5_weight.h" 
#include "batchnorm5_running_mean.h" 
#include "batchnorm5_running_var.h" 
#include "batchnorm5_bias.h" 
#include "batchnorm5_weight.h" 
#include "bn1.h" 
#include "bn2.h" 
#include "bn3.h" 
#include "image.h"
static int8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define C1KXY 3
#define C1XY   12
#define C1Z   3
#define C1KZ 32
#define C1PD 0
#define C1PL 2
static int8_t l1wght[] = _61 ;
#define C2KXY 3
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt]; 
#define C2PD 0
#define C2PL 1 
#define C2NPI 259
static uint8_t l2ind[] = _conv2_weight_indices ;
static pckDtype l2wght[] = _conv2_weight ;
#define C3KXY 3
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 32
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt]; 
#define C3PD 0
#define C3PL 1 
#define C3NPI 259
static uint8_t l3ind[] = _conv3_weight_indices ;
static pckDtype l3wght[] = _conv3_weight ;
#define C4KXY 1
#define C4XY ((2*C3PD+C3XY-C3KXY+1)/C3PL) 
#define C4Z 32
#define C4KZ 2
static pckDtype l4act_bin[C4XY*C4XY*C4Z/pckWdt]; 
#define C4PD 0
#define C4PL 1 
static pckDtype l4wght[] = _conv4_weight ;
#define C5KXY 1
#define C5XY ((2*C4PD+C4XY-C4KXY+1)/C4PL) 
#define C5Z 2
#define C5KZ 4
static pckDtype l5act_bin[C5XY*C5XY*C5Z/pckWdt]; 
#define C5PD 0
#define C5PL 1 
static pckDtype l5wght[] = _conv5_weight ;
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
static bnDtype bn4mean[] = _batchnorm5_running_mean ; 
static bnDtype bn4var[] = _batchnorm5_running_var ; 
static bnDtype bn4gamma[] = _batchnorm5_weight ; 
static bnDtype bn4beta[] = _batchnorm5_bias ; 
/*int main(){ 
	int correct = 0; 
	for(int img = 0; img < 100; img++) {
		uint8_t *curr_im = l1_act + img*12*12*3*sizeof(uint8_t);
		CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
		Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn1thr, bn1sign, bn1offset);
		Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn2thr, bn2sign, bn2offset);
		int res = CnXnorNoBinWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, output, C4PD, C4PL, bn3mean, bn3var, bn3gamma, bn3beta);
		int res = CnXnorNoBinWrap(l5act_bin, l5wght, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, output, C5PD, C5PL, bn3mean, bn3var, bn3gamma, bn3beta);
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
