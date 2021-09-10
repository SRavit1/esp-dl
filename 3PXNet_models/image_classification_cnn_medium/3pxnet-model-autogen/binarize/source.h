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
#include "conv4_weight.h" 
#include "conv5_weight.h" 
#include "conv6_weight.h" 
#include "conv7_weight.h" 
#include "bnfc1_running_mean.h" 
#include "bnfc1_running_var.h" 
#include "bnfc1_bias.h" 
#include "bnfc1_weight.h" 
#include "bn1.h" 
#include "bn2.h" 
#include "bn3.h" 
#include "bn4.h" 
#include "bn5.h" 
#include "bn6.h" 
#include "image.h"
//static uint8_t l1_act[] = IMAGES ; 
//static uint8_t   labels[] = LABELS; 
#define C1KXY 3
#define C1XY   32
#define C1Z   3
#define C1KZ 128
#define C1PD 1
#define C1PL 1 
static int8_t l1wght[] = _conv1_weight ;
#define C2KXY 3
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 128
#define C2KZ 128
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt]; 
#define C2PD 1
#define C2PL 2
static pckDtype l2wght[] = _conv2_weight ;
#define C3KXY 3
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 128
#define C3KZ 256
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt]; 
#define C3PD 1
#define C3PL 1 
static pckDtype l3wght[] = _conv3_weight ;
#define C4KXY 3
#define C4XY ((2*C3PD+C3XY-C3KXY+1)/C3PL) 
#define C4Z 256
#define C4KZ 256
static pckDtype l4act_bin[C4XY*C4XY*C4Z/pckWdt]; 
#define C4PD 1
#define C4PL 2
static pckDtype l4wght[] = _conv4_weight ;
#define C5KXY 3
#define C5XY ((2*C4PD+C4XY-C4KXY+1)/C4PL) 
#define C5Z 256
#define C5KZ 512
static pckDtype l5act_bin[C5XY*C5XY*C5Z/pckWdt]; 
#define C5PD 1
#define C5PL 1 
static pckDtype l5wght[] = _conv5_weight ;
#define C6KXY 3
#define C6XY ((2*C5PD+C5XY-C5KXY+1)/C5PL) 
#define C6Z 512
#define C6KZ 512
static pckDtype l6act_bin[C6XY*C6XY*C6Z/pckWdt]; 
#define C6PD 1
#define C6PL 2
static pckDtype l6wght[] = _conv6_weight ;
#define C7KXY 4
#define C7XY ((2*C6PD+C6XY-C6KXY+1)/C6PL) 
#define C7Z 512
#define C7KZ 10
static pckDtype l7act_bin[C7XY*C7XY*C7Z/pckWdt]; 
#define C7PD 0
#define C7PL 1 
static pckDtype l7wght[] = _conv7_weight ;
static float output[10]; 
static bnDtype bn1thr[] = bn1_thresh ; 
static pckDtype bn1sign[] = bn1_sign ; 
static bnDtype bn2thr[] = bn2_thresh ; 
static pckDtype bn2sign[] = bn2_sign ; 
static bnDtype bn3thr[] = bn3_thresh ; 
static pckDtype bn3sign[] = bn3_sign ; 
static bnDtype bn4thr[] = bn4_thresh ; 
static pckDtype bn4sign[] = bn4_sign ; 
static bnDtype bn5thr[] = bn5_thresh ; 
static pckDtype bn5sign[] = bn5_sign ; 
static bnDtype bn6thr[] = bn6_thresh ; 
static pckDtype bn6sign[] = bn6_sign ; 
static bnDtype bn7mean[] = _bnfc1_running_mean ; 
static bnDtype bn7var[] = _bnfc1_running_var ; 
static bnDtype bn7gamma[] = _bnfc1_weight ; 
static bnDtype bn7beta[] = _bnfc1_bias ; 
/*int main(){ 
	int correct = 0; 
	for(int img = 0; img < 100; img++) {
		uint8_t *curr_im = l1_act + img*32*32*3*sizeof(uint8_t);
		CnBnBwn(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr, bn1sign);
		CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn2thr, bn2sign);
		CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn3thr, bn3sign);
		CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, bn4thr, bn4sign);
		CnXnorWrap(l5act_bin, l5wght, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, l6act_bin, C5PD, C5PL, bn5thr, bn5sign);
		CnXnorWrap(l6act_bin, l6wght, C6Z, C6XY, C6XY, C6Z, C6KXY, C6KXY, C6KZ, l7act_bin, C6PD, C6PL, bn6thr, bn6sign);
		int res = CnXnorNoBinWrap(l7act_bin, l7wght, C7Z, C7XY, C7XY, C7Z, C7KXY, C7KXY, C7KZ, output, C7PD, C7PL, bn7mean, bn7var, bn7gamma, bn7beta);
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
}
*/