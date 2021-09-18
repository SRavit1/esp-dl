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
#include "74.h" 
#include "77.h" 
#include "80.h" 
#include "86.h" 
#include "83.h" 
#include "image.h"
static int8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define C1KXY 3
#define C1XY   12
#define C1Z   3
#define C1KZ 32
#define C1PD 0
#define C1PL 2
static int8_t l1wght[] = _74 ;
#define C2KXY 3
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt]; 
#define C2PD 0
#define C2PL 1 
#define C2NPI 143
static uint8_t l2ind[] = _77_indices ;
static pckDtype l2wght[] = _77 ;
#define C3KXY 3
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 32
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt]; 
#define C3PD 0
#define C3PL 1 
#define C3NPI 143
static uint8_t l3ind[] = _80_indices ;
static pckDtype l3wght[] = _80 ;
#define C4KXY 1
#define C4XY ((2*C3PD+C3XY-C3KXY+1)/C3PL) 
#define C4Z 32
#define C4KZ 4
static pckDtype l4act_bin[C4XY*C4XY*C4Z/pckWdt]; 
#define C4PD 0
#define C4PL 1 
#define C4NPI 8
static uint8_t l4ind[] = _86_indices ;
static pckDtype l4wght[] = _86 ;
#define C5KXY 1
#define C5XY ((2*C4PD+C4XY-C4KXY+1)/C4PL) 
#define C5Z 4
#define C5KZ 2
static pckDtype l5act_bin[C5XY*C5XY*C5Z/pckWdt]; 
#define C5PD 0
#define C5PL 1 
#define C5NPI 0
static uint8_t l5ind[] = _83_indices ;
static pckDtype l5wght[] = _83 ;
static float output[10]; 
/*int main(){ 
	int correct = 0; 
	for(int img = 0; img < 1; img++) {
		uint8_t *curr_im = l1_act + img*12*12*3*sizeof(uint8_t);
		CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
		Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
		Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
		Cn3pxnWrap(l4act_bin, l4wght, l4ind, C4NPI, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL);
		int res = Cn3pxnNoBinWrap(l5act_bin, l5wght, l5ind, C5NPI, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, output, C5PD, C5PL, bn1mean, bn1var, bn1gamma, bn1beta);
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
