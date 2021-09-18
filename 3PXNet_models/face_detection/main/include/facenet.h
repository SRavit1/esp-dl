#pragma once
#ifndef FACENET_GUARD
#define FACENET_GUARD

#include "dl_lib_matrix3d.h"
#include "mtmn.h"
#include "fd_forward.h"
#include "esp_log.h"
#include "printUtils.h"

#define _3PXNET_IMPL
#define PNET
#define BINARIZE

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

void pnet_lite_f_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset){
        int res;

#ifdef PNET
        uint8_t *curr_im = in->item;
        //this conversion is incorrect, but sufficient for inference time measurements
        int8_t *curr_im_int8 = (int8_t*) curr_im;
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
        ESP_LOGI(TAG, "pnet forward pass finished in %lld mu_s.", (time_conv5 - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv1 - time_start));
        ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv2 - time_conv1));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv3 - time_conv2));
        ESP_LOGI(TAG, "conv_4 time: %lld mu_s.", (time_conv4 - time_conv3));
        ESP_LOGI(TAG, "conv_5 time: %lld mu_s.", (time_conv5 - time_conv4));
#endif
}

void rnet_lite_f_with_score_verify_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset) {
        int res;

#ifdef RNET
        uint8_t *curr_im = in->item;
        //this conversion is incorrect, but sufficient for inference time measurements
        int8_t *curr_im_int8 = (int8_t*) curr_im;
#ifdef BINARIZE
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
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
        ESP_LOGI(TAG, "rnet forward pass finished in %lld mu_s.", (time_fc3 - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv1 - time_start));
        ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv2 - time_conv1));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv3 - time_conv2));
        ESP_LOGI(TAG, "fc_1 time: %lld mu_s.", (time_fc1 - time_conv3));
        ESP_LOGI(TAG, "fc_2 time: %lld mu_s.", (time_fc2 - time_fc1));
        ESP_LOGI(TAG, "fc_3 time: %lld mu_s.", (time_fc3 - time_fc2));
#endif
}

void onet_lite_f_with_score_verify_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset, dl_matrix3d_t *landmark) {
        int res;
#ifdef ONET
        uint8_t *curr_im = in->item;
        //this conversion is incorrect, but sufficient for inference time measurements
	int8_t *curr_im_int8 = (int8_t*) curr_im;
#ifdef BINARIZE
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
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
        CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
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
        CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
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
        CnBwnWrap(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 1);
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
        ESP_LOGI(TAG, "onet forward pass finished in %lld mu_s.", (time_fc4 - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv1 - time_start));
        ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv2 - time_conv1));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv3 - time_conv2));
        ESP_LOGI(TAG, "conv_4 time: %lld mu_s.", (time_conv4 - time_conv3));
        ESP_LOGI(TAG, "fc_1 time: %lld mu_s.", (time_fc1 - time_conv4));
        ESP_LOGI(TAG, "fc_2 time: %lld mu_s.", (time_fc2 - time_fc1));
        ESP_LOGI(TAG, "fc_3 time: %lld mu_s.", (time_fc3 - time_fc2));
        ESP_LOGI(TAG, "fc_4 time: %lld mu_s.", (time_fc4 - time_fc3));
#endif
}

#elif defined (ESP_IMPL)
#if defined(FULL_PREC)
void pnet_lite_f_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset) {
        ESP_LOGI(TAG, "Custom pnet_lite_f called!");

        /*
        int64_t time_start = esp_timer_get_time();
        dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &pnet_conv2d_kernel1, &pnet_conv2d_bias1, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_1);
        int64_t time_conv_1 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);
        int64_t time_pool_1 = esp_timer_get_time();;
        dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &pnet_conv2d_kernel2, &pnet_conv2d_bias2, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_2);
        int64_t time_conv_2 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_conv_2, &pnet_conv2d_kernel3, &pnet_conv2d_bias3, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_3);
        int64_t time_conv_3 = esp_timer_get_time();
        *category = *(dl_matrix3dff_conv_common(out_conv_3, &pnet_conv2d_kernel4, &pnet_conv2d_bias4, 1, 1, PADDING_VALID));
        dl_matrix3d_softmax(category); //TODO: How to indicate that this should be done over axis 3?
        int64_t time_category = esp_timer_get_time();
        *offset = *(dl_matrix3dff_conv_common(out_conv_3, &pnet_conv2d_kernel5, &pnet_conv2d_bias5, 1, 1, PADDING_VALID));
        int64_t time_finish = esp_timer_get_time();
        */

        int64_t time_start = esp_timer_get_time();
        dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &pnet_conv2d_kernel1, 0, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_1);
        int64_t time_conv_1 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);
        int64_t time_pool_1 = esp_timer_get_time();;
        dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &pnet_conv2d_kernel2, 0, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_2);
        int64_t time_conv_2 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_conv_2, &pnet_conv2d_kernel3, 0, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_3);
        int64_t time_conv_3 = esp_timer_get_time();
        *category = *(dl_matrix3dff_conv_common(out_conv_3, &pnet_conv2d_kernel4, 0, 1, 1, PADDING_VALID));
        dl_matrix3d_softmax(category); //TODO: How to indicate that this should be done over axis 3?
        int64_t time_category = esp_timer_get_time();
        *offset = *(dl_matrix3dff_conv_common(out_conv_3, &pnet_conv2d_kernel5, 0, 1, 1, PADDING_VALID));
        int64_t time_finish = esp_timer_get_time();

        //TODO: Call free as soon as tensors are no longer live
        dl_matrix3d_free(out_conv_1);
        //TODO: The following statement causes problems
        //dl_matrix3d_free(out_pool_1);
        dl_matrix3d_free(out_conv_2);
        dl_matrix3d_free(out_conv_3);

        ESP_LOGI(TAG, "pnet forward pass finished in %lld mu_s.", (time_finish - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
        ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
        ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_conv_2));
        ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_conv_3));
        ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_finish - time_category));
}

void rnet_lite_f_with_score_verify_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset) {
        ESP_LOGI(TAG, "Custom rnet_lite_f_with_score_verify called!");

        /*
        int64_t time_start = esp_timer_get_time();
        dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &rnet_conv2d_kernel1, &rnet_conv2d_bias1, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_1);
        int64_t time_conv_1 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);
        int64_t time_pool_1 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &rnet_conv2d_kernel2, &rnet_conv2d_bias2, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_2);
        int64_t time_conv_2 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_2 = dl_matrix3d_pooling(out_conv_2, 2, 2, 2, 2, PADDING_VALID, DL_POOLING_MAX);
        int64_t time_pool_2 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_pool_2, &rnet_conv2d_kernel3, &rnet_conv2d_bias3, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_3);
        int64_t time_conv_3 = esp_timer_get_time();
        //flatten out_conv_3 for matrix multiplication
        out_conv_3->c = out_conv_3->h*out_conv_3->w*out_conv_3->c;
        out_conv_3->h = 1;
        out_conv_3->w = 1;
        dl_matrix3d_t *out_dense_1 = dl_matrix3d_alloc(1, 1, 1, rnet_dense_kernel1.h);
        dl_matrix3dff_fc_with_bias(out_dense_1, out_conv_3, &rnet_dense_kernel1, &rnet_dense_bias1);
        dl_matrix3d_relu(out_dense_1);
        int64_t time_dense_1 = esp_timer_get_time();
        *category = *(dl_matrix3d_alloc(1, 1, 1, rnet_dense_kernel2.h));
        dl_matrix3dff_fc_with_bias(category, out_dense_1, &rnet_dense_kernel2, &rnet_dense_bias2);
        dl_matrix3d_softmax(category);
        int64_t time_category = esp_timer_get_time();
        *offset = *(dl_matrix3d_alloc(1, 1, 1, rnet_dense_kernel3.h));
        dl_matrix3dff_fc_with_bias(offset, out_dense_1, &rnet_dense_kernel3, &rnet_dense_bias3);
        int64_t time_finish = esp_timer_get_time();
        */

        int64_t time_start = esp_timer_get_time();
        dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &rnet_conv2d_kernel1, 0, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_1);
        int64_t time_conv_1 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);
        int64_t time_pool_1 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &rnet_conv2d_kernel2, 0, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_2);
        int64_t time_conv_2 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_2 = dl_matrix3d_pooling(out_conv_2, 3, 3, 3, 3, PADDING_VALID, DL_POOLING_MAX);
        int64_t time_pool_2 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_pool_2, &rnet_conv2d_kernel3, 0, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_3);
        int64_t time_conv_3 = esp_timer_get_time();
        //flatten out_conv_3 for matrix multiplication
        out_conv_3->c = out_conv_3->h*out_conv_3->w*out_conv_3->c;
        out_conv_3->h = 1;
        out_conv_3->w = 1;
        dl_matrix3d_t *out_dense_1 = dl_matrix3d_alloc(1, 1, 1, rnet_dense_kernel1.h);
        dl_matrix3dff_fc(out_dense_1, out_conv_3, &rnet_dense_kernel1);
        dl_matrix3d_relu(out_dense_1);
        int64_t time_dense_1 = esp_timer_get_time();
        *category = *(dl_matrix3d_alloc(1, 1, 1, rnet_dense_kernel2.h));
        dl_matrix3dff_fc(category, out_dense_1, &rnet_dense_kernel2);
        dl_matrix3d_softmax(category);
        int64_t time_category = esp_timer_get_time();
        *offset = *(dl_matrix3d_alloc(1, 1, 1, rnet_dense_kernel3.h));
        dl_matrix3dff_fc(offset, out_dense_1, &rnet_dense_kernel3);
        int64_t time_finish = esp_timer_get_time();

        //TODO: Call free as soon as tensors are no longer live
        dl_matrix3d_free(out_conv_1);
        //dl_matrix3d_free(out_pool_1);
        dl_matrix3d_free(out_conv_2);
        //dl_matrix3d_free(out_pool_2);
        dl_matrix3d_free(out_conv_3);
        dl_matrix3d_free(out_dense_1);

        ESP_LOGI(TAG, "rnet forward pass finished in %lld mu_s.", (time_finish - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
        ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
        ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
        ESP_LOGI(TAG, "pool_2 time: %lld mu_s.", (time_pool_2 - time_conv_2));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_pool_2));
        ESP_LOGI(TAG, "dense_1 time: %lld mu_s.", (time_dense_1 - time_conv_3));
        ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_dense_1));
        ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_finish - time_category));
}

void onet_lite_f_with_score_verify_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset, dl_matrix3d_t *landmark) {
        ESP_LOGI(TAG, "Custom onet_lite_f_with_score_verify called!");

        /*
        int64_t time_start = esp_timer_get_time();
        dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &onet_conv2d_kernel1, &onet_conv2d_bias1, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_1);
        int64_t time_conv_1 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 3, 3, 2, 2, PADDING_SAME, DL_POOLING_MAX);
        int64_t time_pool_1 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &onet_conv2d_kernel2, &onet_conv2d_bias2, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_2);
        int64_t time_conv_2 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_2 = dl_matrix3d_pooling(out_conv_2, 3, 3, 2, 2, PADDING_VALID, DL_POOLING_MAX);
        int64_t time_pool_2 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_pool_2, &onet_conv2d_kernel3, &onet_conv2d_bias3, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_3);
        int64_t time_conv_3 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_4 = dl_matrix3d_pooling(out_conv_3, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);
        int64_t time_pool_3 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_4 = dl_matrix3dff_conv_common(out_conv_4, &onet_conv2d_kernel4, &onet_conv2d_bias4, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_4);
        int64_t time_conv_4 = esp_timer_get_time();
        //flatten out_conv_4 for matrix multiplication
        out_conv_4->c = out_conv_4->h*out_conv_4->w*out_conv_4->c;
        out_conv_4->h = 1;
        out_conv_4->w = 1;
        dl_matrix3d_t *out_dense_1 = dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel1.h);
        dl_matrix3dff_fc_with_bias(out_dense_1, out_conv_4, &onet_dense_kernel1, &onet_dense_bias1);
        dl_matrix3d_relu(out_dense_1);
        int64_t time_dense_1 = esp_timer_get_time();
        *category = *(dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel2.h));
        dl_matrix3dff_fc_with_bias(category, out_dense_1, &onet_dense_kernel2, &onet_dense_bias2);
        dl_matrix3d_softmax(category);
        int64_t time_category = esp_timer_get_time();
        *offset = *(dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel3.h));
        dl_matrix3dff_fc_with_bias(offset, out_dense_1, &onet_dense_kernel3, &onet_dense_bias3);
        int64_t time_offset = esp_timer_get_time();
        *landmark = *(dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel4.h));
        dl_matrix3dff_fc_with_bias(landmark, out_dense_1, &onet_dense_kernel4, &onet_dense_bias4);
        int64_t time_finish = esp_timer_get_time();
        */

        int64_t time_start = esp_timer_get_time();
        dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &onet_conv2d_kernel1, 0, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_1);
        int64_t time_conv_1 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 3, 3, 2, 2, PADDING_SAME, DL_POOLING_MAX);
        int64_t time_pool_1 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &onet_conv2d_kernel2, 0, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_2);
        int64_t time_conv_2 = esp_timer_get_time();
        dl_matrix3d_t *out_pool_2 = dl_matrix3d_pooling(out_conv_2, 3, 3, 2, 2, PADDING_VALID, DL_POOLING_MAX);
        int64_t time_pool_2 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_pool_2, &onet_conv2d_kernel3, 0, 1, 1, PADDING_VALID);
        dl_matrix3d_relu(out_conv_3);
        int64_t time_conv_3 = esp_timer_get_time();
        dl_matrix3d_t *out_conv_4 = dl_matrix3dff_conv_common(out_conv_3, &onet_conv2d_kernel4, 0, 1, 1, PADDING_VALID);
        int64_t time_conv4 = esp_timer_get_time();
        //flatten out_conv_4 for matrix multiplication
        out_conv_4->c = out_conv_4->h*out_conv_4->w*out_conv_4->c;
        out_conv_4->h = 1;
        out_conv_4->w = 1;
        dl_matrix3d_t *out_dense_1 = dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel1.h);
        dl_matrix3dff_fc(out_dense_1, out_conv_4, &onet_dense_kernel1);
        dl_matrix3d_relu(out_dense_1);
        int64_t time_dense_1 = esp_timer_get_time();
        *category = *(dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel2.h));
        dl_matrix3dff_fc(category, out_dense_1, &onet_dense_kernel2);
        dl_matrix3d_softmax(category);
        int64_t time_category = esp_timer_get_time();
        *offset = *(dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel3.h));
        dl_matrix3dff_fc(offset, out_dense_1, &onet_dense_kernel3);
        int64_t time_offset = esp_timer_get_time();
        *landmark = *(dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel4.h));
        dl_matrix3dff_fc(landmark, out_dense_1, &onet_dense_kernel4);
        int64_t time_finish = esp_timer_get_time();

        //TODO: Call free as soon as tensors are no longer live
        dl_matrix3d_free(out_conv_1);
        //dl_matrix3d_free(out_pool_1);
        dl_matrix3d_free(out_conv_2);
        //dl_matrix3d_free(out_pool_2);
        dl_matrix3d_free(out_conv_3);
        dl_matrix3d_free(out_dense_1);

        ESP_LOGI(TAG, "onet forward pass finished in %lld mu_s.", (time_finish - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
        ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
        ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
        ESP_LOGI(TAG, "pool_2 time: %lld mu_s.", (time_pool_2 - time_conv_2));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_pool_2));
        ESP_LOGI(TAG, "conv_4 time: %lld mu_s.", (time_conv4 - time_conv_3));
        ESP_LOGI(TAG, "dense_1 time: %lld mu_s.", (time_dense_1 - time_conv4));
        ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_dense_1));
        ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_offset - time_category));
        ESP_LOGI(TAG, "landmark time: %lld mu_s.", (time_finish - time_offset));
}
#endif
#if defined(QUANTIZED)
const int EXP_TODO = 0;

int pnet_lite_q_esp(dl_matrix3du_t *in, dl_matrix3d_t *category_f, dl_matrix3d_t *offset_f, dl_conv_mode mode) {
    /*
    int64_t time_start = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_1 = dl_matrix3duq_conv_common(in, &pnet_conv2d_kernel1_q, &pnet_conv2d_bias1_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_1);
    int64_t time_conv_1 = esp_timer_get_time();
    dl_matrix3dq_t *out_pool_1 = dl_matrix3dq_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);
    int64_t time_pool_1 = esp_timer_get_time();;
    dl_matrix3dq_t *out_conv_2 = dl_matrix3dqq_conv_common(out_pool_1, &pnet_conv2d_kernel2_q, &pnet_conv2d_bias2_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_2);
    int64_t time_conv_2 = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_3 = dl_matrix3dqq_conv_common(out_conv_2, &pnet_conv2d_kernel3_q, &pnet_conv2d_bias3_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_3);
    int64_t time_conv_3 = esp_timer_get_time();
    dl_matrix3dq_t *category = dl_matrix3dqq_conv_common(out_conv_3, &pnet_conv2d_kernel4_q, &pnet_conv2d_bias4_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    *category_f = *(dl_matrix3d_from_matrixq(category));
    dl_matrix3d_softmax(category_f); //TODO: How to indicate that this should be done over axis 3?
    int64_t time_category = esp_timer_get_time();
    dl_matrix3dq_t *offset = dl_matrix3dqq_conv_common(out_conv_3, &pnet_conv2d_kernel5_q, &pnet_conv2d_bias5_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    *offset_f = *(dl_matrix3d_from_matrixq(offset));
    int64_t time_finish = esp_timer_get_time();
    */

    int64_t time_start = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_1 = dl_matrix3duq_conv_common(in, &pnet_conv2d_kernel1, &pnet_conv2d_bias1, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_1);
    int64_t time_conv_1 = esp_timer_get_time();
    dl_matrix3dq_t *out_pool_1 = dl_matrix3dq_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);
    int64_t time_pool_1 = esp_timer_get_time();;
    dl_matrix3dq_t *out_conv_2 = dl_matrix3dqq_conv_common(out_pool_1, &pnet_conv2d_kernel2, &pnet_conv2d_bias2, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_2);
    int64_t time_conv_2 = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_3 = dl_matrix3dqq_conv_common(out_conv_2, &pnet_conv2d_kernel3, &pnet_conv2d_bias3, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_3);
    int64_t time_conv_3 = esp_timer_get_time();
    dl_matrix3dq_t *category = dl_matrix3dqq_conv_common(out_conv_3, &pnet_conv2d_kernel4, &pnet_conv2d_bias4, 1, 1, PADDING_VALID, EXP_TODO, mode);
    *category_f = *(dl_matrix3d_from_matrixq(category));
    dl_matrix3d_softmax(category_f); //TODO: How to indicate that this should be done over axis 3?
    int64_t time_category = esp_timer_get_time();
    dl_matrix3dq_t *offset = dl_matrix3dqq_conv_common(out_conv_3, &pnet_conv2d_kernel5, &pnet_conv2d_bias5, 1, 1, PADDING_VALID, EXP_TODO, mode);
    *offset_f = *(dl_matrix3d_from_matrixq(offset));
    int64_t time_finish = esp_timer_get_time();
    
    ESP_LOGI(TAG, "pnet forward pass finished in %lld mu_s.", (time_finish - time_start));
    ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
    ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
    ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
    ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_conv_2));
    ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_conv_3));
    ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_finish - time_category));

    //TODO: Call free as soon as tensors are no longer live
    dl_matrix3dq_free(out_conv_1);
    //TODO: The following statement causes problems
    //dl_matrix3dq_free(out_pool_1);
    dl_matrix3dq_free(out_conv_2);
    dl_matrix3dq_free(out_conv_3);
    dl_matrix3dq_free(offset);
    dl_matrix3dq_free(category);

    return 1;
}

int rnet_lite_q_with_score_verify_esp(dl_matrix3du_t *in, dl_matrix3d_t *category_f, dl_matrix3d_t *offset_f, dl_conv_mode mode, float threshold) {
    int64_t time_start = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_1 = dl_matrix3duq_conv_common(in, &rnet_conv2d_kernel1, &rnet_conv2d_bias1, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_1);
    int64_t time_conv_1 = esp_timer_get_time();
    dl_matrix3dq_t *out_pool_1 = dl_matrix3dq_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);
    int64_t time_pool_1 = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_2 = dl_matrix3dqq_conv_common(out_pool_1, &rnet_conv2d_kernel2, &rnet_conv2d_bias2, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_2);
    int64_t time_conv_2 = esp_timer_get_time();
    dl_matrix3dq_t *out_pool_2 = dl_matrix3dq_pooling(out_conv_2, 3, 3, 3, 3, PADDING_VALID, DL_POOLING_MAX);
    int64_t time_pool_2 = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_3 = dl_matrix3dqq_conv_common(out_pool_2, &rnet_conv2d_kernel3, &rnet_conv2d_bias3, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_3);
    int64_t time_conv_3 = esp_timer_get_time();
    //flatten out_conv_3 for matrix multiplication
    out_conv_3->c = out_conv_3->h*out_conv_3->w*out_conv_3->c;
    out_conv_3->h = 1;
    out_conv_3->w = 1;
    dl_matrix3dq_t *out_dense_1 = dl_matrix3dq_alloc(1, 1, 1, rnet_dense_kernel1.h, EXP_TODO);
    dl_matrix3dqq_fc(out_dense_1, out_conv_3, &rnet_dense_kernel1, mode, "out_dense_1");
    dl_matrix3dq_relu(out_dense_1);
    int64_t time_dense_1 = esp_timer_get_time();
    dl_matrix3dq_t *category = dl_matrix3dq_alloc(1, 1, 1, rnet_dense_kernel2.h, EXP_TODO);
    dl_matrix3dqq_fc(category, out_dense_1, &rnet_dense_kernel2, mode, "category");
    *category_f = *(dl_matrix3d_from_matrixq(category));
    dl_matrix3d_softmax(category_f);
    int64_t time_category = esp_timer_get_time();
    dl_matrix3dq_t *offset = dl_matrix3dq_alloc(1, 1, 1, rnet_dense_kernel3.h, EXP_TODO);
    dl_matrix3dqq_fc(offset, out_dense_1, &rnet_dense_kernel3, mode, "offset");
    *offset_f = *(dl_matrix3d_from_matrixq(offset));
    int64_t time_finish = esp_timer_get_time();
    
    ESP_LOGI(TAG, "rnet forward pass finished in %lld mu_s.", (time_finish - time_start));
    ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
    ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
    ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
    ESP_LOGI(TAG, "pool_2 time: %lld mu_s.", (time_pool_2 - time_conv_2));
    ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_pool_2));
    ESP_LOGI(TAG, "dense_1 time: %lld mu_s.", (time_dense_1 - time_conv_3));
    ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_dense_1));
    ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_finish - time_category));

    if (category->item[0] < threshold) {
        dl_matrix3d_free(category_f);
        dl_matrix3d_free(offset_f);
        return 1;
    }

    //TODO: Call free as soon as tensors are no longer live
    dl_matrix3dq_free(out_conv_1);
    //dl_matrix3dq_free(out_pool_1);
    dl_matrix3dq_free(out_conv_2);
    //dl_matrix3dq_free(out_pool_2);
    dl_matrix3dq_free(out_conv_3);
    dl_matrix3dq_free(out_dense_1);
    dl_matrix3dq_free(offset);
    dl_matrix3dq_free(category);

    return 1;
}

int onet_lite_q_with_score_verify_esp(dl_matrix3du_t *in, dl_matrix3d_t *category_f, dl_matrix3d_t *offset_f, dl_matrix3d_t *landmark_f, dl_conv_mode mode, float threshold) {
    int64_t time_start = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_1 = dl_matrix3duq_conv_common(in, &onet_conv2d_kernel1, &onet_conv2d_bias1, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_1);
    int64_t time_conv_1 = esp_timer_get_time();
    dl_matrix3dq_t *out_pool_1 = dl_matrix3dq_pooling(out_conv_1, 3, 3, 2, 2, PADDING_SAME, DL_POOLING_MAX);
    int64_t time_pool_1 = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_2 = dl_matrix3dqq_conv_common(out_pool_1, &onet_conv2d_kernel2, &onet_conv2d_bias2, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_2);
    int64_t time_conv_2 = esp_timer_get_time();
    dl_matrix3dq_t *out_pool_2 = dl_matrix3dq_pooling(out_conv_2, 3, 3, 2, 2, PADDING_VALID, DL_POOLING_MAX);
    int64_t time_pool_2 = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_3 = dl_matrix3dqq_conv_common(out_pool_2, &onet_conv2d_kernel3, &onet_conv2d_bias3, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_3);
    int64_t time_conv_3 = esp_timer_get_time();
    dl_matrix3dq_t *out_conv_4 = dl_matrix3dqq_conv_common(out_conv_3, &onet_conv2d_kernel4, &onet_conv2d_bias4, 1, 1, PADDING_VALID, EXP_TODO, mode);
    int64_t time_conv_4 = esp_timer_get_time();
    //flatten out_conv_4 for matrix multiplication
    out_conv_4->c = out_conv_4->h*out_conv_4->w*out_conv_4->c;
    out_conv_4->h = 1;
    out_conv_4->w = 1;
    dl_matrix3dq_t *out_dense_1 = dl_matrix3dq_alloc(1, 1, 1, onet_dense_kernel1.h, EXP_TODO);
    dl_matrix3dqq_fc(out_dense_1, out_conv_4, &onet_dense_kernel1, mode, "out_dense_1");
    dl_matrix3dq_relu(out_dense_1);
    int64_t time_dense_1 = esp_timer_get_time();
    dl_matrix3dq_t *category = dl_matrix3dq_alloc(1, 1, 1, onet_dense_kernel2.h, EXP_TODO);
    dl_matrix3dqq_fc(category, out_dense_1, &onet_dense_kernel2, mode, "category");
    *category_f = *(dl_matrix3d_from_matrixq(category));
    dl_matrix3d_softmax(category_f);
    int64_t time_category = esp_timer_get_time();
    dl_matrix3dq_t *offset = dl_matrix3dq_alloc(1, 1, 1, onet_dense_kernel3.h, EXP_TODO);
    dl_matrix3dqq_fc(offset, out_dense_1, &onet_dense_kernel3, mode, "offset");
    *offset_f = *(dl_matrix3d_from_matrixq(offset));
    int64_t time_offset = esp_timer_get_time();
    dl_matrix3dq_t *landmark = dl_matrix3dq_alloc(1, 1, 1, onet_dense_kernel4.h, EXP_TODO);
    dl_matrix3dqq_fc(landmark, out_dense_1, &onet_dense_kernel4, mode, "landmark");
    *landmark_f = *(dl_matrix3d_from_matrixq(landmark));
    int64_t time_finish = esp_timer_get_time();

    ESP_LOGI(TAG, "onet forward pass finished in %lld mu_s.", (time_finish - time_start));
    ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
    ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
    ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
    ESP_LOGI(TAG, "pool_2 time: %lld mu_s.", (time_pool_2 - time_conv_2));
    ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_pool_2));
    ESP_LOGI(TAG, "conv_4 time: %lld mu_s.", (time_conv_4 - time_conv_3));
    ESP_LOGI(TAG, "dense_1 time: %lld mu_s.", (time_dense_1 - time_conv_4));
    ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_dense_1));
    ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_offset - time_category));
    ESP_LOGI(TAG, "landmark time: %lld mu_s.", (time_finish - time_offset));

    if (category->item[0] < threshold) {
        dl_matrix3d_free(category_f);
        dl_matrix3d_free(offset_f);
        dl_matrix3d_free(landmark_f);
        return 1;
    }

    //TODO: Call free as soon as tensors are no longer live
    dl_matrix3dq_free(out_conv_1);
    //dl_matrix3dq_free(out_pool_1);
    dl_matrix3dq_free(out_conv_2);
    //dl_matrix3dq_free(out_pool_2);
    dl_matrix3dq_free(out_conv_3);
    dl_matrix3dq_free(out_dense_1);
    dl_matrix3dq_free(offset);
    dl_matrix3dq_free(category);
    dl_matrix3dq_free(landmark);

    return 1;
}
#endif
#endif

#endif
