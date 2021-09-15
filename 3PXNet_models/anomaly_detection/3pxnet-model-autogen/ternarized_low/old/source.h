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
#include "150.h" 
#include "151.h" 
#include "152.h" 
#include "153.h" 
#include "154.h" 
#include "155.h" 
#include "156.h" 
#include "157.h" 
#include "158.h" 
#include "159.h" 
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

static uint8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define F1I  640
#define F1NPI  575
#define F1O  128
static pckDtype l1wght[] = _150 ;
static uint8_t l1ind[] = _150_indices ;
static pckDtype l1act_bin[F1I/pckWdt]; 
#define F2I  128
#define F2NPI  115
#define F2O  128
static pckDtype l2wght[] = _151 ;
static uint8_t l2ind[] = _151_indices ;
static pckDtype l2act_bin[F1O/pckWdt]; 
#define F3I  128
#define F3NPI  115
#define F3O  128
static pckDtype l3wght[] = _152 ;
static uint8_t l3ind[] = _152_indices ;
static pckDtype l3act_bin[F2O/pckWdt]; 
#define F4I  128
#define F4NPI  115
#define F4O  128
static pckDtype l4wght[] = _153 ;
static uint8_t l4ind[] = _153_indices ;
static pckDtype l4act_bin[F3O/pckWdt]; 
#define F5I  128
#define F5NPI  112
#define F5O  8
static pckDtype l5wght[] = _154 ;
static uint8_t l5ind[] = _154_indices ;
static pckDtype l5act_bin[F4O/pckWdt]; 
#define F6I  8
#define F6NPI  0
#define F6O  128
static pckDtype l6wght[] = _155 ;
static uint8_t l6ind[] = _155_indices ;
static pckDtype l6act_bin[F5O/pckWdt]; 
#define F7I  128
#define F7NPI  115
#define F7O  128
static pckDtype l7wght[] = _156 ;
static uint8_t l7ind[] = _156_indices ;
static pckDtype l7act_bin[F6O/pckWdt]; 
#define F8I  128
#define F8NPI  115
#define F8O  128
static pckDtype l8wght[] = _157 ;
static uint8_t l8ind[] = _157_indices ;
static pckDtype l8act_bin[F7O/pckWdt]; 
#define F9I  128
#define F9NPI  115
#define F9O  128
static pckDtype l9wght[] = _158 ;
static uint8_t l9ind[] = _158_indices ;
static pckDtype l9act_bin[F8O/pckWdt]; 
#define F10I  128
#define F10NPI  115
#define F10O  640
static pckDtype l10wght[] = _159 ;
static uint8_t l10ind[] = _159_indices ;
static pckDtype l10act_bin[F9O/pckWdt]; 
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
static bnDtype bn7thr[] = bn7_thresh ; 
static pckDtype bn7sign[] = bn7_sign ; 
static bnDtype bn8thr[] = bn8_thresh ; 
static pckDtype bn8sign[] = bn8_sign ; 
static bnDtype bn9thr[] = bn9_thresh ; 
static pckDtype bn9sign[] = bn9_sign ; 
static bnDtype bn10mean[] = _batchnorm10_running_mean ; 
static bnDtype bn10var[] = _batchnorm10_running_var ; 
static bnDtype bn10gamma[] = _batchnorm10_weight ; 
static bnDtype bn10beta[] = _batchnorm10_bias ; 
int main(){ 
	int correct = 0; 
	for(int img = 0; img < 1; img++) {
		uint8_t *curr_im = l1_act + img*784*sizeof(uint8_t);
		packBinThrsArr(curr_im, l1act_bin, F1I, 1);
		/*
		int64_t start_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnWrap(l1act_bin, l1wght, l1ind, F1NPI, F1O, l2act_bin, bn1thr, bn1sign);
		int64_t fc1_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnWrap(l2act_bin, l2wght, l2ind, F2NPI, F2O, l3act_bin, bn2thr, bn2sign);
		int64_t fc2_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnWrap(l3act_bin, l3wght, l3ind, F3NPI, F3O, l4act_bin, bn3thr, bn3sign);
		int64_t fc3_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn4thr, bn4sign);
		*/
		int64_t fc4_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnWrap(l5act_bin, l5wght, l5ind, F5NPI, F5O, l6act_bin, bn5thr, bn5sign);
		int64_t fc5_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnWrap(l6act_bin, l6wght, l6ind, F6NPI, F6O, l7act_bin, bn6thr, bn6sign);
		int64_t fc6_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnWrap(l7act_bin, l7wght, l7ind, F7NPI, F7O, l8act_bin, bn7thr, bn7sign);
		int64_t fc7_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnWrap(l8act_bin, l8wght, l8ind, F8NPI, F8O, l9act_bin, bn8thr, bn8sign);
		int64_t fc8_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnWrap(l9act_bin, l9wght, l9ind, F9NPI, F9O, l10act_bin, bn9thr, bn9sign);
		int64_t fc9_time = esp_timer_get_time();
		for (int i = 0; i < 1000; i++)
		Fc3pxnNoBinWrap(l10act_bin, l10wght, l10ind, F10NPI, F10O, output, bn10mean, bn10var, bn10gamma, bn10beta);
		int64_t fc10_time = esp_timer_get_time();

		//ESP_LOGI(TAG, "forward pass took %lld microseconds", (fc10_time - start_time));
		/*
		ESP_LOGI(TAG, "fc1 took %lld microseconds", (fc1_time - start_time));
		ESP_LOGI(TAG, "fc2 took %lld microseconds", (fc2_time - fc1_time));
		ESP_LOGI(TAG, "fc3 took %lld microseconds", (fc3_time - fc2_time));
		ESP_LOGI(TAG, "fc4 took %lld microseconds", (fc4_time - fc3_time));
		*/
		ESP_LOGI(TAG, "fc5 took %lld microseconds", (fc5_time - fc4_time));
		ESP_LOGI(TAG, "fc6 took %lld microseconds", (fc6_time - fc5_time));
		ESP_LOGI(TAG, "fc7 took %lld microseconds", (fc7_time - fc6_time));
		ESP_LOGI(TAG, "fc8 took %lld microseconds", (fc8_time - fc7_time));
		ESP_LOGI(TAG, "fc9 took %lld microseconds", (fc9_time - fc8_time));
		ESP_LOGI(TAG, "fc10 took %lld microseconds", (fc10_time - fc9_time));
		
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