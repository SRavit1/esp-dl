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
#include "110.h" 
#include "111.h" 
#include "112.h" 
#include "113.h" 
#include "114.h" 
#include "115.h" 
#include "116.h" 
#include "117.h" 
#include "118.h" 
#include "119.h" 
#include "batchnorm10_running_mean.h" 
#include "batchnorm10_running_var.h" 
#include "batchnorm10_bias.h" 
#include "batchnorm10_weight.h" 
#include "bn1.h" 
#include "bn2.h" 
#include "bn3.h" 
#include "bn4.h" 
#include "bn5.h" 
#include "bn6.h" 
#include "bn7.h" 
#include "bn8.h" 
#include "bn9.h" 
#include "image.h"

#include "esp_log.h"
static const char *TAG = "app_main";

static int8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define F1I  640
#define F1NPI  0
#define F1O  128
static pckDtype l1wght[] = _110 ;
static pckDtype l1act_bin[F1I/pckWdt]; 
#define F2I  128
#define F2NPI  0
#define F2O  128
static pckDtype l2wght[] = _111 ;
static pckDtype l2act_bin[F1O/pckWdt]; 
#define F3I  128
#define F3NPI  0
#define F3O  128
static pckDtype l3wght[] = _112 ;
static pckDtype l3act_bin[F2O/pckWdt]; 
#define F4I  128
#define F4NPI  0
#define F4O  128
static pckDtype l4wght[] = _113 ;
static pckDtype l4act_bin[F3O/pckWdt]; 
#define F5I  128
#define F5NPI  0
#define F5O  8
static pckDtype l5wght[] = _114 ;
static pckDtype l5act_bin[F4O/pckWdt]; 
#define F6I  8
#define F6NPI  0
#define F6O  128
static pckDtype l6wght[] = _115 ;
static pckDtype l6act_bin[F5O/pckWdt]; 
#define F7I  128
#define F7NPI  0
#define F7O  128
static pckDtype l7wght[] = _116 ;
static pckDtype l7act_bin[F6O/pckWdt]; 
#define F8I  128
#define F8NPI  0
#define F8O  128
static pckDtype l8wght[] = _117 ;
static pckDtype l8act_bin[F7O/pckWdt]; 
#define F9I  128
#define F9NPI  0
#define F9O  128
static pckDtype l9wght[] = _118 ;
static pckDtype l9act_bin[F8O/pckWdt]; 
#define F10I  128
#define F10NPI  0
#define F10O  640
static pckDtype l10wght[] = _119 ;
static pckDtype l10act_bin[F9O/pckWdt]; 
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
static pckDtype bn4thr[] = bn4_thresh ; 
static pckDtype bn4sign[] = bn4_sign ; 
static pckDtype bn4offset[] = bn4_offset ; 
static pckDtype bn5thr[] = bn5_thresh ; 
static pckDtype bn5sign[] = bn5_sign ; 
static pckDtype bn5offset[] = bn5_offset ; 
static pckDtype bn6thr[] = bn6_thresh ; 
static pckDtype bn6sign[] = bn6_sign ; 
static pckDtype bn6offset[] = bn6_offset ; 
static pckDtype bn7thr[] = bn7_thresh ; 
static pckDtype bn7sign[] = bn7_sign ; 
static pckDtype bn7offset[] = bn7_offset ; 
static pckDtype bn8thr[] = bn8_thresh ; 
static pckDtype bn8sign[] = bn8_sign ; 
static pckDtype bn8offset[] = bn8_offset ; 
static pckDtype bn9thr[] = bn9_thresh ; 
static pckDtype bn9sign[] = bn9_sign ; 
static pckDtype bn9offset[] = bn9_offset ; 
static bnDtype bn10mean[] = _batchnorm10_running_mean ; 
static bnDtype bn10var[] = _batchnorm10_running_var ; 
static bnDtype bn10gamma[] = _batchnorm10_weight ; 
static bnDtype bn10beta[] = _batchnorm10_bias ; 
int main(){ 
	int correct = 0; 
	for(int img = 0; img < 1; img++) {
		int res;

		uint8_t *curr_im = (uint8_t *) (l1_act + img*784*sizeof(uint8_t));
		//int8_t *curr_im_int8 = (uint8_t *)curr_im;
		packBinThrsArr(curr_im, l1act_bin, F1I, 1);
		int64_t start_time = esp_timer_get_time();
		res = FcXnorWrap(l1act_bin, l1wght, F1I, F1O, l2act_bin, bn1thr, bn1sign, bn1offset, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
		int64_t fc1_time = esp_timer_get_time();
		res = FcXnorWrap(l2act_bin, l2wght, F2I, F2O, l3act_bin, bn2thr, bn2sign, bn2offset, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
		int64_t fc2_time = esp_timer_get_time();
		/*
		res = FcXnorWrap(l3act_bin, l3wght, F3I, F3O, l4act_bin, bn3thr, bn3sign, bn3offset, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
		int64_t fc3_time = esp_timer_get_time();
		res = FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn4thr, bn4sign, bn4offset, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
		int64_t fc4_time = esp_timer_get_time();
		res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn5thr, bn5sign, bn5offset, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc5 res is 1");
		int64_t fc5_time = esp_timer_get_time();
		res = FcXnorWrap(l6act_bin, l6wght, F6I, F6O, l7act_bin, bn6thr, bn6sign, bn6offset, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc6 res is 1");
		int64_t fc6_time = esp_timer_get_time();
		res = FcXnorWrap(l7act_bin, l7wght, F7I, F7O, l8act_bin, bn7thr, bn7sign, bn7offset, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc7 res is 1");
		int64_t fc7_time = esp_timer_get_time();
		res = FcXnorWrap(l8act_bin, l8wght, F8I, F8O, l9act_bin, bn8thr, bn8sign, bn8offset, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc8 res is 1");
		int64_t fc8_time = esp_timer_get_time();
		res = FcXnorWrap(l9act_bin, l9wght, F9I, F9O, l10act_bin, bn9thr, bn9sign, bn9offset, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc9 res is 1");
		int64_t fc9_time = esp_timer_get_time();
		res = FcXnorNoBinWrap(l10act_bin, l10wght, F10I, F10O, output, bn10mean, bn10var, bn10gamma, bn10beta, 1, 1);
		if (res) ESP_LOGI(TAG, "ERROR: fc10 res is 1");
		int64_t fc10_time = esp_timer_get_time();
		*/

		//ESP_LOGI(TAG, "forward pass took %lld microseconds", (fc10_time - start_time));
		ESP_LOGI(TAG, "fc1 took %lld microseconds", (fc1_time - start_time));
		ESP_LOGI(TAG, "fc2 took %lld microseconds", (fc2_time - fc1_time));
		/*
		ESP_LOGI(TAG, "fc3 took %lld microseconds", (fc3_time - fc2_time));
		ESP_LOGI(TAG, "fc4 took %lld microseconds", (fc4_time - fc3_time));
		ESP_LOGI(TAG, "fc5 took %lld microseconds", (fc5_time - fc4_time));
		ESP_LOGI(TAG, "fc6 took %lld microseconds", (fc6_time - fc5_time));
		ESP_LOGI(TAG, "fc7 took %lld microseconds", (fc7_time - fc6_time));
		ESP_LOGI(TAG, "fc8 took %lld microseconds", (fc8_time - fc7_time));
		ESP_LOGI(TAG, "fc9 took %lld microseconds", (fc9_time - fc8_time));
		ESP_LOGI(TAG, "fc10 took %lld microseconds", (fc10_time - fc9_time));
		*/

		/*
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
		*/
	}
	//printf("Accuracy: %f%%\n", 100.0*(float)correct/1); 
	return (EXIT_SUCCESS); 
}