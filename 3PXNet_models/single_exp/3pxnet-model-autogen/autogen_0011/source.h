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
#include "esp_log.h"
static const char *TAG = "app_main";
#include "conv0_weight.h" 
#include "conv1_weight.h" 
#include "conv2_weight.h" 
#include "image.h"
static int8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define C1KXY 3
#define C1XY   8
#define C1Z   1
#define C1KZ 32
#define C1PD 1
#define C1PL 1 
static int8_t l1wght[] = _conv0_weight ;
#define C2KXY 3
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt]; 
#define C2PD 1
#define C2PL 2
#define C2NPI 32
static uint8_t l2ind[] = _conv1_weight_indices ;
static pckDtype l2wght[] = _conv1_weight ;
#define C3KXY 3
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 1
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt]; 
#define C3PD 1
#define C3PL 1 
static pckDtype l3wght[] = _conv2_weight ;
static float output[10]; 
int main(){ 
	int correct = 0; 
	for(int img = 0; img < 1; img++) {
		//uint8_t *curr_im = l1_act + img*8*8*1*sizeof(uint8_t);
		//CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
		int64_t start_time = esp_timer_get_time();
		#ifdef INF_LIB_PUBLIC
		Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL);
		#endif
		#ifdef INF_LIB_PRIVATE
		Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL, NULL, 1, 1);
		#endif
		int64_t end_time = esp_timer_get_time();
		ESP_LOGI(TAG, "forward pass took %lld microseconds", (end_time - start_time));
		//int res = CnXnorNoBinWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, output, C3PD, C3PL, bn1mean, bn1var, bn1gamma, bn1beta);
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