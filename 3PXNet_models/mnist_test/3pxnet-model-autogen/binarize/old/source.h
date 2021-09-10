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
#include "25.h" 
#include "26.h" 
#include "bn2_running_mean.h" 
#include "bn2_running_var.h" 
#include "bn2_bias.h" 
#include "bn2_weight.h" 
#include "bn1.h" 
#include "image.h"
static uint8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define F1I  768
#define F1NPI  0
#define F1O  128
static pckDtype l1wght[] = _25 ;
static pckDtype l1act_bin[F1I/pckWdt]; 
#define F2I  128
#define F2NPI  0
#define F2O  10
static pckDtype l2wght[] = _26 ;
static pckDtype l2act_bin[F1O/pckWdt]; 
static float output[10]; 
static bnDtype bn1thr[] = bn1_thresh ; 
static pckDtype bn1sign[] = bn1_sign ; 
static bnDtype bn2mean[] = _bn2_running_mean ; 
static bnDtype bn2var[] = _bn2_running_var ; 
static bnDtype bn2gamma[] = _bn2_weight ; 
static bnDtype bn2beta[] = _bn2_bias ; 
/*int main(){ 
	int correct = 0; 
	for(int img = 0; img < 100; img++) {
		uint8_t *curr_im = l1_act + img*784*sizeof(uint8_t);
		packBinThrsArr(curr_im, l1act_bin, F1I, 1);
		FcXnorWrap(l1act_bin, l1wght, F1I, F1O, l2act_bin, bn1thr, bn1sign);
		int res = FcXnorNoBinWrap(l2act_bin, l2wght, F2I, F2O, output, bn2mean, bn2var, bn2gamma, bn2beta);
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