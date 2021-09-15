#pragma once
#ifndef FACENET_GUARD
#define FACENET_GUARD

#include "dl_lib_matrix3d.h"
#include "mtmn.h"
#include "fd_forward.h"
#include "esp_log.h"
#include "printUtils.h"

#define ESP_IMPL
#define ONET
#define FULL_PREC

#if defined (ESP_IMPL)
#if defined(FULL_PREC)
#include "facenet_full_prec.h"
#endif
#endif

static const char *TAG = "app_process";

#if defined(_3PXNET_IMPL)
#if defined(BINARIZE)
#ifdef PNET
#include "pnet_binarize/source.h"
#endif
#ifdef RNET
#include "rnet_binarize/source.h"
#endif
#ifdef ONET
#include "onet_binarize/source.h"
#endif
#endif

#if defined(TERNARIZE_LOW)
#ifdef PNET
#include "pnet_ternarize_low/source.h"
#endif
#ifdef RNET
#include "rnet_ternarize_low/source.h"
#endif
#ifdef ONET
#include "onet_ternarize_low/source.h"
#endif
#endif

#if defined(TERNARIZE_MEDIUM)
#ifdef PNET
#include "pnet_ternarize_medium/source.h"
#endif
#ifdef RNET
#include "rnet_ternarize_medium/source.h"
#endif
#ifdef ONET
#include "onet_ternarize_medium/source.h"
#endif
#endif

#if defined(TERNARIZE_HIGH)
#ifdef PNET
#include "pnet_ternarize_high/source.h"
#endif
#ifdef RNET
#include "rnet_ternarize_high/source.h"
#endif
#ifdef ONET
#include "onet_ternarize_high/source.h"
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
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL,0);
	int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn1thr, bn1sign, bn1offset, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn2thr, bn2sign, bn2offset, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, category->item, C4PD, C4PL, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l4act_bin, l5wght, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, offset->item, C5PD, C5PL, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv5 res is 1");
        int64_t time_conv5 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_LOW
        int64_t time_start = esp_timer_get_time();
	CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
	int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn1thr, bn1sign, bn1offset, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn2thr, bn2sign, bn2offset, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, category->item, C4PD, C4PL, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l5act_bin, l5wght, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, offset->item, C5PD, C5PL, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv5 res is 1");
        int64_t time_conv5 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_MEDIUM
        int64_t time_start = esp_timer_get_time();
	CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
	int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn1thr, bn1sign, bn1offset, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn2thr, bn2sign, bn2offset, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, category->item, C4PD, C4PL, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l5act_bin, l5wght, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, offset->item, C5PD, C5PL, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv5 res is 1");
        int64_t time_conv5 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_HIGH
        int64_t time_start = esp_timer_get_time();
	CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
	int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn1thr, bn1sign, bn1offset, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn2thr, bn2sign, bn2offset, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, category->item, C4PD, C4PL, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
	if (res) ESP_LOGI(TAG, "ERROR: conv4 res is 1");
        int64_t time_conv4 = esp_timer_get_time();
        res = CnXnorNoBinWrap(l5act_bin, l5wght, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, offset->item, C5PD, C5PL, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
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
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn3mean, bn3var, bn3gamma, bn3beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l6wght, F6I, F6O, offset->item, bn3mean, bn3var, bn3gamma, bn3beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_LOW
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn3mean, bn3var, bn3gamma, bn3beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, offset->item, bn3mean, bn3var, bn3gamma, bn3beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_MEDIUM
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn3mean, bn3var, bn3gamma, bn3beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, offset->item, bn3mean, bn3var, bn3gamma, bn3beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc3 res is 1");
        int64_t time_fc3 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_HIGH
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv2 res is 1");
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: conv3 res is 1");
        int64_t time_conv3 = esp_timer_get_time();
        res = Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc1 res is 1");
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn3mean, bn3var, bn3gamma, bn3beta, 0, 0);
        if (res) ESP_LOGI(TAG, "ERROR: fc2 res is 1");
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, offset->item, bn3mean, bn3var, bn3gamma, bn3beta, 0, 0);
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
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
        int64_t time_conv1 = esp_timer_get_time();
        res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 0, 0);
        int64_t time_conv2 = esp_timer_get_time();
        res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 0 ,0);
        int64_t time_conv3 = esp_timer_get_time();
        res = FcXnorWrap(l4act_bin, l4wght, F4I, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 0, 0);
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, offset->item, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc3 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, landmark->item, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc4 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_LOW
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 0, 0);
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 0, 0);
        int64_t time_conv3 = esp_timer_get_time();
        res = Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 0, 0);
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, offset->item, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc3 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, landmark->item, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc4 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_MEDIUM
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 0, 0);
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 0, 0);
        int64_t time_conv3 = esp_timer_get_time();
        res = Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 0, 0);
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, category->item, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, offset->item, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc3 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, landmark->item, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc4 = esp_timer_get_time();
#endif
#ifdef TERNARIZE_HIGH
        int64_t time_start = esp_timer_get_time();
        CnBwnWrap(curr_im_int8, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL, 0);
        int64_t time_conv1 = esp_timer_get_time();
        res = Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL, 0, 0);
        int64_t time_conv2 = esp_timer_get_time();
        res = Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL, 0, 0);
        int64_t time_conv3 = esp_timer_get_time();
        res = Fc3pxnWrap(l4act_bin, l4wght, l4ind, F4NPI, F4O, l5act_bin, bn1thr, bn1sign, bn1offset, 0, 0);
        int64_t time_fc1 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l5act_bin, l5wght, F5I, F5O, output, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc2 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc3 = esp_timer_get_time();
        res = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn4mean, bn4var, bn4gamma, bn4beta, 0, 0);
        int64_t time_fc4 = esp_timer_get_time();
#endif
        ESP_LOGI(TAG, "onet forward pass finished in %lld mu_s.", (time_fc4 - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv1 - time_start));
        ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv2 - time_conv1));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv3 - time_conv2));
        ESP_LOGI(TAG, "fc_1 time: %lld mu_s.", (time_fc1 - time_conv3));
        ESP_LOGI(TAG, "fc_2 time: %lld mu_s.", (time_fc2 - time_fc1));
        ESP_LOGI(TAG, "fc_3 time: %lld mu_s.", (time_fc3 - time_fc2));
        ESP_LOGI(TAG, "fc_4 time: %lld mu_s.", (time_fc4 - time_fc3));
#endif
}

#elif defined (ESP_IMPL)
#if defined(FULL_PREC)
void pnet_lite_f_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset) {
        ESP_LOGI(TAG, "Custom pnet_lite_f called!");

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

        //TODO: Call free as soon as tensors are no longer live
        dl_matrix3d_free(out_conv_1);
        //dl_matrix3d_free(out_pool_1);
        dl_matrix3d_free(out_conv_2);
        //dl_matrix3d_free(out_pool_2);
        dl_matrix3d_free(out_conv_3);
        dl_matrix3d_free(out_dense_1);

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
}

void onet_lite_f_with_score_verify_esp(dl_matrix3du_t *in, dl_matrix3d_t *category, dl_matrix3d_t *offset, dl_matrix3d_t *landmark) {
        ESP_LOGI(TAG, "Custom onet_lite_f_with_score_verify called!");

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

        dl_matrix3d_t *out_pool_3 = dl_matrix3d_pooling(out_conv_3, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);

        int64_t time_pool_3 = esp_timer_get_time();

        dl_matrix3d_t *out_conv_4 = dl_matrix3dff_conv_common(out_pool_3, &onet_conv2d_kernel4, &onet_conv2d_bias4, 1, 1, PADDING_VALID);
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

        //TODO: Call free as soon as tensors are no longer live
        dl_matrix3d_free(out_conv_1);
        //dl_matrix3d_free(out_pool_1);
        dl_matrix3d_free(out_conv_2);
        //dl_matrix3d_free(out_pool_2);
        dl_matrix3d_free(out_conv_3);
        dl_matrix3d_free(out_dense_1);

        int64_t time_finish = esp_timer_get_time();
        ESP_LOGI(TAG, "onet forward pass finished in %lld mu_s.", (time_finish - time_start));
        ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
        ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
        ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
        ESP_LOGI(TAG, "pool_2 time: %lld mu_s.", (time_pool_2 - time_conv_2));
        ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_pool_2));
        ESP_LOGI(TAG, "pool_3 time: %lld mu_s.", (time_pool_3 - time_conv_3));
        ESP_LOGI(TAG, "conv_4 time: %lld mu_s.", (time_conv_4 - time_pool_3));
        ESP_LOGI(TAG, "dense_1 time: %lld mu_s.", (time_dense_1 - time_conv_4));
        ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_dense_1));
        ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_offset - time_category));
        ESP_LOGI(TAG, "landmark time: %lld mu_s.", (time_finish - time_offset));
}
#endif
#if defined(QUANTIZED)
void pnet_lite_q_esp() {

}

void rnet_lite_q_with_score_verify_esp() {

}

void onet_lite_q_with_score_verify_esp() {

}
#endif
#endif

#endif