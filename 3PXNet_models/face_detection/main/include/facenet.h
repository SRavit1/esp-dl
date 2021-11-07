#pragma once
#ifndef FACENET_GUARD
#define FACENET_GUARD

#include "dl_lib_matrix3d.h"
#include "mtmn.h"
#include "fd_forward.h"
#include "esp_log.h"
#include "printUtils.h"

#define _3PXNET_IMPL
#define ONET
#define BINARIZE_BW1_INPUTBW2

#if defined (ESP_IMPL)
#if defined(FULL_PREC)
#include "facenet_full_prec.h"
#elif defined(QUANTIZED)
#include "facenet_full_prec_qu.h"
#endif
#endif

static const char *TAG = "app_process";

#if defined(_3PXNET_IMPL)

//These should be in source.h, but redefining just in case
#include "xnor_base.h"
#include "xnor_fc.h"
#include "3pxnet_fc.h"
#include "3pxnet_cn.h"
#include "xnor_cn.h"

#if defined(BINARIZE)
#ifdef PNET
#include "pnet_binarized/source.h"
#endif
#ifdef RNET
#include "rnet_binarized/source.h"
#endif
#ifdef ONET
#include "onet_binarized/source.h"
#endif
#endif

#if defined(TERNARIZE_LOW)
#ifdef PNET
#include "pnet_ternarized_low/source.h"
#endif
#ifdef RNET
#include "rnet_ternarized_low/source.h"
#endif
#ifdef ONET
#include "onet_ternarized_low/source.h"
#endif
#endif

#if defined(TERNARIZE_MEDIUM)
#ifdef PNET
#include "pnet_ternarized_medium/source.h"
#endif
#ifdef RNET
#include "rnet_ternarized_medium/source.h"
#endif
#ifdef ONET
#include "onet_ternarized_medium/source.h"
#endif
#endif

#if defined(TERNARIZE_HIGH)
#ifdef PNET
#include "pnet_ternarized_high/source.h"
#endif
#ifdef RNET
#include "rnet_ternarized_high/source.h"
#endif
#ifdef ONET
#include "onet_ternarized_high/source.h"
#endif
#endif

#if defined(BINARIZE_BW1_INPUTBW2)
#ifdef PNET
#include "pnet_model_binary_bw1_input_bw2_46bf2839_autogen/source.h"
#endif
#ifdef RNET
#include "rnet_model_binary_bw1_input_bw2_b17db2b0_autogen/source.h"
#endif
#ifdef ONET
#include "onet_model_binary_bw1_input_bw2_0a2c3f60_autogen/source.h"
#endif
#endif

#if defined(BINARIZE_BW1_INPUTBW4)
#ifdef PNET
#include "pnet_model_binary_bw1_input_bw4_a879e7fb_autogen/source.h"
#endif
#ifdef RNET
#include "rnet_model_binary_bw1_input_bw4_031e30bf_autogen/source.h"
#endif
#ifdef ONET
#include "onet_model_binary_bw1_input_bw4_55351e1a_autogen/source.h"
#endif
#endif

#if defined(BINARIZE_BW1_INPUTBW6)
#ifdef PNET
#include "pnet_model_binary_bw1_input_bw6_e9f6532c_autogen/source.h"
#endif
#ifdef RNET
#include "rnet_model_binary_bw1_input_bw6_3c2f38a4_autogen/source.h"
#endif
#ifdef ONET
#include "onet_model_binary_bw1_input_bw6_ab918c15_autogen/source.h"
#endif
#endif

#if defined(BINARIZE_BW2_INPUTBW2)
#ifdef PNET
#include "pnet_model_binary_bw2_input_bw2_54fb74db_autogen/source.h"
#endif
#ifdef RNET
#include "rnet_model_binary_bw2_input_bw2_c3091def_autogen/source.h"
#endif
#ifdef ONET
#include "onet_model_binary_bw2_input_bw2_6695afda_autogen/source.h"
#endif
#endif

#if defined(BINARIZE_BW2_INPUTBW4)
#ifdef PNET
#include "pnet_model_binary_bw2_input_bw4_5c9e98e8_autogen/source.h"
#endif
#ifdef RNET
#include "rnet_model_binary_bw2_input_bw4_c580b651_autogen/source.h"
#endif
#ifdef ONET
#include "onet_model_binary_bw2_input_bw4_941626bc_autogen/source.h"
#endif
#endif

#if defined(BINARIZE_BW2_INPUTBW6)
#ifdef PNET
#include "pnet_model_binary_bw2_input_bw6_2388faf8_autogen/source.h"
#endif
#ifdef RNET
#include "rnet_model_binary_bw2_input_bw6_d9bd48e5_autogen/source.h"
#endif
#ifdef ONET
#include "onet_model_binary_bw2_input_bw6_b31bf2aa_autogen/source.h"
#endif
#endif

void pnet_lite_f_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset){
        int res;

#ifdef PNET
        uint8_t *curr_im = in->item;
        //this conversion is incorrect, but sufficient for inference time measurements
        int8_t *curr_im_int8 = (int8_t*) curr_im;
        //This is very incorrect, since it is just casting pointer types. TODO: Fix it.
        pckDtype* curr_im_packed = (pckDtype*)curr_im;
#ifdef BINARIZE
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL,1);
	int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL, NULL, 1, 1);
	if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL, NULL, 1, 1);
	if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, category->item, C4PD, C4PL, NULL, NULL, NULL, NULL, 1, 1);
	if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l4act_bin, l5wght, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, offset->item, C5PD, C5PL, NULL, NULL, NULL, NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv5 res is 1");
        int64_t time_conv5 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_LOW
        int64_t time_start = esp_timer_get_time();
	CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
	int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL, NULL, 1, 1);
	if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL, NULL, 1, 1);
	if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Cn3pxnNoBinWrap(l4act_bin, l4wght, l4ind, C4NPI, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, category->item, C4PD, C4PL, NULL, NULL,NULL, NULL, 1 ,1);
	if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = Cn3pxnNoBinWrap(l5act_bin, l5wght, l5ind, C5NPI, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, offset->item, C5PD, C5PL, NULL, NULL, NULL, NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv5 res is 1");
        int64_t time_conv5 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_MEDIUM
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL, NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL, NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Cn3pxnNoBinWrap(l4act_bin, l4wght, l4ind, C4NPI, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, category->item, C4PD, C4PL, NULL, NULL,NULL, NULL, 1 ,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = Cn3pxnNoBinWrap(l5act_bin, l5wght, l5ind, C5NPI, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, offset->item, C5PD, C5PL, NULL, NULL, NULL, NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv5 res is 1");
        int64_t time_conv5 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_HIGH
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL, NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL, NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Cn3pxnNoBinWrap(l4act_bin, l4wght, l4ind, C4NPI, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, category->item, C4PD, C4PL, NULL, NULL,NULL, NULL, 1 ,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = Cn3pxnNoBinWrap(l5act_bin, l5wght, l5ind, C5NPI, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, offset->item, C5PD, C5PL, NULL, NULL, NULL, NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv5 res is 1");
        int64_t time_conv5 = esp_timer_get_time();
#endif
#ifdef BINARIZE_BW1_INPUTBW2
        int64_t time_start = esp_timer_get_time();
        //CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
        CnBnMulti(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset, 2, 1);
        int64_t time_conv1 = esp_timer_get_time();
        /*CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
        CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
        int res0 = CnXnorNoBinWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, output, C4PD, C4PL, bn1mean, bn1var, bn1gamma, bn1beta);
        int res1 = CnXnorNoBinWrap(l5act_bin, l5wght, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, output, C5PD, C5PL, bn1mean, bn1var, bn1gamma, bn1beta);*/
#endif
#ifdef BINARIZE_BW1_INPUTBW4
#endif
#ifdef BINARIZE_BW1_INPUTBW6
#endif
#ifdef BINARIZE_BW2_INPUTBW2
#endif
#ifdef BINARIZE_BW2_INPUTBW4
#endif
#ifdef BINARIZE_BW2_INPUTBW6
#endif
        ESP_LOGI(TAG, "category output");
        for (int i = 0; i < 2; i++) {
            printf("%f, ", category->item[i]);
        }
        ESP_LOGI(TAG, "offset output");
        for (int i = 0; i < 4; i++) {
            printf("%f, ", offset->item[i]);
        }

        //ESP_LOGI(TAG, "pnet forward pass finished in %lld mu_s.", (time_conv5 - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv1 - time_start));
        /*ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv2 - time_conv1));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv3 - time_conv2));
        ESP_LOGI(TAG, "conv_4 time: %lld mu_s.", (time_conv4 - time_conv3));
        ESP_LOGI(TAG, "conv_5 time: %lld mu_s.", (time_conv5 - time_conv4));*/
#endif
}

void rnet_lite_f_with_score_verify_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset) {
        int res;

#ifdef RNET
        uint8_t *curr_im = in->item;
        //this conversion is incorrect, but sufficient for inference time measurements
        int8_t *curr_im_int8 = (int8_t*) curr_im;
        //This is very incorrect, since it is just casting pointer types. TODO: Fix it.
        pckDtype* curr_im_packed = (pckDtype*)curr_im;
#ifdef BINARIZE
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn2thr, bn2sign, bn2offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn3mean, bn3var, bn3gamma, bn3beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_LOW
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn3mean, bn3var, bn3gamma, bn3beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, offset->item, bn3mean, bn3var, bn3gamma, bn3beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_MEDIUM
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn3mean, bn3var, bn3gamma, bn3beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, offset->item, bn3mean, bn3var, bn3gamma, bn3beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_HIGH
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn3mean, bn3var, bn3gamma, bn3beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, offset->item, bn3mean, bn3var, bn3gamma, bn3beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
#endif
#ifdef BINARIZE_BW1_INPUTBW2
        int64_t time_start = esp_timer_get_time();
        //CnBwnWrap(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
        CnBnMulti(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset, 1, 1);
        int64_t time_conv1 = esp_timer_get_time();
        /*CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
        CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
        FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset);
        int res0 = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, output, bn2mean, bn2var, bn2gamma, bn2beta);
        int res1 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn2mean, bn2var, bn2gamma, bn2beta);*/
#endif
#ifdef BINARIZE_BW1_INPUTBW4
int64_t time_start = esp_timer_get_time();
        //CnBwnWrap(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
        CnBnMulti(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset, 4, 1);
        int64_t time_conv1 = esp_timer_get_time();
        /*CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
        CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
        FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset);
        int res0 = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, output, bn2mean, bn2var, bn2gamma, bn2beta);
        int res1 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn2mean, bn2var, bn2gamma, bn2beta);*/
#endif
#ifdef BINARIZE_BW1_INPUTBW6
int64_t time_start = esp_timer_get_time();
        //CnBwnWrap(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
        CnBnMulti(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset, 6, 1);
        int64_t time_conv1 = esp_timer_get_time();
        /*CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
        CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
        FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset);
        int res0 = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, output, bn2mean, bn2var, bn2gamma, bn2beta);
        int res1 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn2mean, bn2var, bn2gamma, bn2beta);*/
#endif
#ifdef BINARIZE_BW2_INPUTBW2
int64_t time_start = esp_timer_get_time();
        //CnBwnWrap(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
        CnBnMulti(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset, 2, 2);
        int64_t time_conv1 = esp_timer_get_time();
        /*CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
        CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
        FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset);
        int res0 = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, output, bn2mean, bn2var, bn2gamma, bn2beta);
        int res1 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn2mean, bn2var, bn2gamma, bn2beta);*/
#endif
#ifdef BINARIZE_BW2_INPUTBW4
int64_t time_start = esp_timer_get_time();
        //CnBwnWrap(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
        CnBnMulti(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset, 4, 2);
        int64_t time_conv1 = esp_timer_get_time();
        /*CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
        CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
        FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset);
        int res0 = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, output, bn2mean, bn2var, bn2gamma, bn2beta);
        int res1 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn2mean, bn2var, bn2gamma, bn2beta);*/
#endif
#ifdef BINARIZE_BW2_INPUTBW6
int64_t time_start = esp_timer_get_time();
        //CnBwnWrap(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL);
        CnBnMulti(curr_im_packed, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset, 6, 2);
        int64_t time_conv1 = esp_timer_get_time();
        /*CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL);
        CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL);
        FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset);
        int res0 = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, output, bn2mean, bn2var, bn2gamma, bn2beta);
        int res1 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn2mean, bn2var, bn2gamma, bn2beta);*/
#endif
        //ESP_LOGI(TAG, "rnet forward pass finished in %lld mu_s.", (time_fc3 - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv1 - time_start));
        /*ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv2 - time_conv1));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv3 - time_conv2));
        ESP_LOGI(TAG, "fc_1 time: %lld mu_s.", (time_fc1 - time_conv3));
        ESP_LOGI(TAG, "fc_2 time: %lld mu_s.", (time_fc2 - time_fc1));
        ESP_LOGI(TAG, "fc_3 time: %lld mu_s.", (time_fc3 - time_fc2));*/
#endif
}

void onet_lite_f_with_score_verify_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset, dl_matrix3d_t *landmark) {
        int res;
#ifdef ONET
        uint8_t *curr_im = in->item;
        //this conversion is incorrect, but sufficient for inference time measurements
	int8_t *curr_im_int8 = (int8_t*) curr_im;
        //This is very incorrect, since it is just casting pointer types. TODO: Fix it.
        pckDtype* curr_im_packed = (pckDtype*)curr_im;
        
#ifdef BINARIZE
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn1thr, bn1sign, bn1offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorWrap(l6act_bin, l6wght, F6I, F6O, l7act_bin, bn2thr, bn2sign, bn2offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorWrap(l7act_bin, l7wght, F7I, F7O, l8act_bin, bn3thr, bn3sign, bn3offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l8act_bin, l8wght, F8I, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
        int64_t time_fc4 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_LOW
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = Fc3pxnWrap(l5act_bin, l5wght, l5ind, F5NPI, F5O, l6act_bin, bn1thr, bn1sign, bn1offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = Fc3pxnWrap(l6act_bin, l6wght, l6ind, F6NPI, F6O, l7act_bin, bn2thr, bn2sign, bn2offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = Fc3pxnWrap(l7act_bin, l7wght, l7ind, F7NPI, F7O, l8act_bin, bn3thr, bn3sign, bn3offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        res = Fc3pxnNoBinWrap(l8act_bin, l8wght, l8ind, F8NPI, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
        int64_t time_fc4 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_MEDIUM
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = Fc3pxnWrap(l5act_bin, l5wght, l5ind, F5NPI, F5O, l6act_bin, bn1thr, bn1sign, bn1offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = Fc3pxnWrap(l6act_bin, l6wght, l6ind, F6NPI, F6O, l7act_bin, bn2thr, bn2sign, bn2offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = Fc3pxnWrap(l7act_bin, l7wght, l7ind, F7NPI, F7O, l8act_bin, bn3thr, bn3sign, bn3offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        res = Fc3pxnNoBinWrap(l8act_bin, l8wght, l8ind, F8NPI, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
        int64_t time_fc4 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_HIGH
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = Fc3pxnWrap(l5act_bin, l5wght, l5ind, F5NPI, F5O, l6act_bin, bn1thr, bn1sign, bn1offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = Fc3pxnWrap(l6act_bin, l6wght, l6ind, F6NPI, F6O, l7act_bin, bn2thr, bn2sign, bn2offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = Fc3pxnWrap(l7act_bin, l7wght, l7ind, F7NPI, F7O, l8act_bin, bn3thr, bn3sign, bn3offset, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        res = Fc3pxnNoBinWrap(l8act_bin, l8wght, l8ind, F8NPI, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta, 1, 1);
        if (res) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
        int64_t time_fc4 = esp_timer_get_time();
#endif
#ifdef BINARIZE_BW1_INPUTBW2
        int64_t time_start = esp_timer_get_time();
        CnBnMulti(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset,2,1);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn1thr, bn1sign, bn1offset,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        int res0 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
        if (res0) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        int res1 = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
        if (res1) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        int res2 = FcXnorNoBinWrap(l8act_bin, l8wght, F8I, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
        int64_t time_fc4 = esp_timer_get_time();
        if (res2) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
#endif
#ifdef BINARIZE_BW1_INPUTBW4
        int64_t time_start = esp_timer_get_time();
        CnBnMulti(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset,4,1);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn1thr, bn1sign, bn1offset,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        int res0 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
        if (res0) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        int res1 = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
        if (res1) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        int res2 = FcXnorNoBinWrap(l8act_bin, l8wght, F8I, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
        int64_t time_fc4 = esp_timer_get_time();
        if (res2) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
#endif
#ifdef BINARIZE_BW1_INPUTBW6
        int64_t time_start = esp_timer_get_time();
        CnBnMulti(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset,6,1);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn1thr, bn1sign, bn1offset,1,1);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        int res0 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
        if (res0) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        int res1 = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
        if (res1) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        int res2 = FcXnorNoBinWrap(l8act_bin, l8wght, F8I, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
        int64_t time_fc4 = esp_timer_get_time();
        if (res2) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
#endif
#ifdef BINARIZE_BW2_INPUTBW2
        int64_t time_start = esp_timer_get_time();
        CnBnMulti(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset,2,2);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn1thr, bn1sign, bn1offset,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        int res0 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn4mean, bn4var, bn4gamma, bn4beta,2,2);
        if (res0) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        int res1 = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn4mean, bn4var, bn4gamma, bn4beta,2,2);
        if (res1) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        int res2 = FcXnorNoBinWrap(l8act_bin, l8wght, F8I, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta,2,2);
        int64_t time_fc4 = esp_timer_get_time();
        if (res2) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
#endif
#ifdef BINARIZE_BW2_INPUTBW4
        int64_t time_start = esp_timer_get_time();
        CnBnMulti(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset,4,2);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn1thr, bn1sign, bn1offset,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        int res0 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn4mean, bn4var, bn4gamma, bn4beta,2,2);
        if (res0) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        int res1 = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn4mean, bn4var, bn4gamma, bn4beta,2,2);
        if (res1) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        int res2 = FcXnorNoBinWrap(l8act_bin, l8wght, F8I, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta,2,2);
        int64_t time_fc4 = esp_timer_get_time();
        if (res2) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
#endif
#ifdef BINARIZE_BW2_INPUTBW6
        int64_t time_start = esp_timer_get_time();
        CnBnMulti(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset,6,2);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn1thr, bn1sign, bn1offset,2,2);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        int res0 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn4mean, bn4var, bn4gamma, bn4beta,2,2);
        if (res0) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        int res1 = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn4mean, bn4var, bn4gamma, bn4beta,2,2);
        if (res1) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
        int res2 = FcXnorNoBinWrap(l8act_bin, l8wght, F8I, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta,2,2);
        int64_t time_fc4 = esp_timer_get_time();
        if (res2) ESP_LOGI(TAG, "ERROR: fc4 res is 1");
#endif
        ESP_LOGI(TAG, "onet forward pass finished in %lld mu_s.", (time_fc4 - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv1 - time_start));
        ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv2 - time_conv1));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv3 - time_conv2));
        ESP_LOGI(TAG, "conv_4 time: %lld mu_s.", (time_conv4 - time_conv3));
        ESP_LOGI(TAG, "fc_1 time: %lld mu_s.", (time_fc1 - time_conv4));
        ESP_LOGI(TAG, "fc_2 time: %lld mu_s.", (time_fc2 - time_fc1));
        ESP_LOGI(TAG, "fc_3 time: %lld mu_s.", (time_fc3 - time_fc2));
        ESP_LOGI(TAG, "fc_4 time: %lld mu_s.", (time_fc4 - time_fc3));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv1 - time_start));
#endif
}

#endif

#endif