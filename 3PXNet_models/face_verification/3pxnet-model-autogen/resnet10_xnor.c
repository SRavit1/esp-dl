/*
* MIT License
* 
* Copyright (c) 2019 UCLA NanoCAD Laboratory 
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

/*!
 * \file      resnet18_xnor.c
 * \brief     XNOR implementation of ResNet18 
 * \author    Ravit Sharma 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

//////////////////////////////
// General Headers
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>

#include "conv1.h"
#include "conv2_1.h"
#include "conv2_2.h"
#include "conv3_1.h"
#include "conv3_2.h"
#include "conv4_1.h"
#include "conv4_2.h"
#include "conv4_d.h"
#include "conv5_1.h"
#include "conv5_2.h"
#include "fc.h"

// Datatypes
#include "datatypes.h"
// NN functions
#include "utils.h"
#include "xnor_base.h"
#include "xnor_fc.h"
#include "bwn_dense_cn.h"
#include "xnor_cn.h"

#include "special.h"

#define act_bw 1
#define weight_bw 1

//static int16_t conv1_act_unpacked[C1XY*C1XY*C1Z] = input_1;

int16_t *conv1_act_unpacked;
int16_t *conv1_wgt_unpacked;
bnDtype *conv1_mean;
bnDtype *conv1_var;
bnDtype *conv1_gamma;
bnDtype *conv1_beta;

bnDtype *conv2_1_act_unpacked;
pckDtype *conv2_1_act;
pckDtype *conv2_2_act;
bnDtype *conv2_3_act_unpacked;
pckDtype *conv2_3_act;
pckDtype *conv2_1_wgt;
bnDtype *conv2_1_thresh;
pckDtype *conv2_1_sign;
pckDtype *conv2_2_wgt;
bnDtype *conv2_2_mean;
bnDtype *conv2_2_var;
bnDtype *conv2_2_gamma;
bnDtype *conv2_2_beta;

bnDtype *conv3_1_act_unpacked;
pckDtype *conv3_1_act;
pckDtype *conv3_2_act;
bnDtype *conv3_3_act_unpacked;
pckDtype *conv3_3_act;
pckDtype *conv3_1_wgt;
bnDtype *conv3_1_thresh;
pckDtype *conv3_1_sign;
pckDtype *conv3_2_wgt;
bnDtype *conv3_2_mean;
bnDtype *conv3_2_var;
bnDtype *conv3_2_gamma;
bnDtype *conv3_2_beta;

bnDtype *conv4_1_act_unpacked;
pckDtype *conv4_1_act;
pckDtype *conv4_2_act;
bnDtype *conv4_3_act_unpacked;
pckDtype *conv4_3_act;
pckDtype *conv4_1_wgt;
bnDtype *conv4_1_thresh;
pckDtype *conv4_1_sign;
pckDtype *conv4_2_wgt;
bnDtype *conv4_2_mean;
bnDtype *conv4_2_var;
bnDtype *conv4_2_gamma;
bnDtype *conv4_2_beta;
bnDtype *conv4_d_act_unpacked;
pckDtype *conv4_d_wgt;
bnDtype *conv4_d_mean;
bnDtype *conv4_d_var;
bnDtype *conv4_d_gamma;
bnDtype *conv4_d_beta;

bnDtype *conv5_1_act_unpacked;
pckDtype *conv5_1_act;
pckDtype *conv5_2_act;
bnDtype *conv5_3_act_unpacked;
pckDtype *conv5_3_act;
pckDtype *conv5_1_wgt;
bnDtype *conv5_1_thresh;
pckDtype *conv5_1_sign;
pckDtype *conv5_2_wgt;
bnDtype *conv5_2_mean;
bnDtype *conv5_2_var;
bnDtype *conv5_2_gamma;
bnDtype *conv5_2_beta;

bnDtype *fc_in;
pckDtype *fc_wgt;
bnDtype *fc_out;

static void* buffer1;
static void* buffer2;
static void* buffer3;
static void* weight_buffer;

//pingpong buffer implementation
void forward_new() {
   int64_t s1, s2;
   s1 = esp_timer_get_time();

   //copy conv1_act_unpacked to buffer1
   //printf("Memcpy arguments %p %p %d", buffer1, conv1_act_unpacked, C1XY*C1XY*C1Z);
   //fflush(stdout);
   //memcpy(buffer1, conv1_act_unpacked, C1XY*C1XY*C1Z);

   refCn_bwn(buffer1, conv1_wgt_unpacked, C1Z, C1XY, C1XY, C1KXY, C1KXY, C1KZ, buffer2, C1PL, conv1_mean, conv1_var, conv1_gamma, conv1_beta);
   
   s2 = esp_timer_get_time();
   printf("first layer (mus): %f\n", ((float)(s2-s1)));
   fflush(stdout);
   s1 = s2;
   //printFloatArray(buffer2, 100);

   struct identity_block_conf block2_conf = {.C_1_act_unpacked= buffer2, .C_1_act= buffer3, .C_2_act= buffer1, .C_3_act_unpacked= buffer3, .C_1_wgt= conv2_1_wgt, .C_2_wgt= conv2_2_wgt, .C_2_mean= conv2_2_mean, .C_2_var= conv2_2_var, .C_2_gamma= conv2_2_gamma, .C_2_beta= conv2_2_beta, .C_1_thresh= conv2_1_thresh, .C_1_sign= conv2_1_sign, .C_1XY= C2_1XY, .C_1Z= C2_1Z, .C_1KXY= C2_1KXY, .C_1KZ= C2_1KZ, .C_1PD= C2_1PD, .C_1PL= C2_1PL, .C_2XY= C2_2XY, .C_2Z= C2_2Z, .C_2KXY= C2_2KXY, .C_2KZ= C2_2KZ, .C_2PD= C2_2PD, .C_2PL= C2_2PL, .C_2OXY= C2_2OXY};
   identity_block(block2_conf);

   s2 = esp_timer_get_time();
   printf("second block (mus): %f\n", ((float)(s2-s1)));
   fflush(stdout);
   s1 = s2;

   //printFloatArray(buffer3, 100);

   struct identity_block_conf block3_conf = {.C_1_act_unpacked= buffer3, .C_1_act= buffer1, .C_2_act= buffer2, .C_3_act_unpacked= buffer1, .C_1_wgt= conv3_1_wgt, .C_2_wgt= conv3_2_wgt, .C_2_mean= conv3_2_mean, .C_2_var= conv3_2_var, .C_2_gamma= conv3_2_gamma, .C_2_beta= conv3_2_beta, .C_1_thresh= conv3_1_thresh, .C_1_sign= conv3_1_sign, .C_1XY= C3_1XY, .C_1Z= C3_1Z, .C_1KXY= C3_1KXY, .C_1KZ= C3_1KZ, .C_1PD= C3_1PD, .C_1PL= C3_1PL, .C_2XY= C3_2XY, .C_2Z= C3_2Z, .C_2KXY= C3_2KXY, .C_2KZ= C3_2KZ, .C_2PD= C3_2PD, .C_2PL= C3_2PL, .C_2OXY= C3_2OXY};
   identity_block(block3_conf);

   s2 = esp_timer_get_time();
   printf("third block (mus): %f\n", ((float)(s2-s1)));
   fflush(stdout);
   s1 = s2;

   //printFloatArray(buffer1, 100);

   struct convolutional_block_conf block4_conf = {.C_1_act_unpacked= buffer1, .C_1_act= buffer2, .C_2_act= buffer1, .C_3_act_unpacked= buffer3, .C_1_wgt= conv4_1_wgt, .C_2_wgt= conv4_2_wgt, .C_2_mean= conv4_2_mean, .C_2_var= conv4_2_var, .C_2_gamma= conv4_2_gamma, .C_2_beta= conv4_2_beta, .C_1_thresh= conv4_1_thresh, .C_1_sign= conv4_1_sign, .C_1XY= C4_1XY, .C_1Z= C4_1Z, .C_1KXY= C4_1KXY, .C_1KZ= C4_1KZ, .C_1PD= C4_1PD, .C_1PL= C4_1PL, .C_2XY= C4_2XY, .C_2Z= C4_2Z, .C_2KXY= C4_2KXY, .C_2KZ= C4_2KZ, .C_2PD= C4_2PD, .C_2PL= C4_2PL, .C_2OXY= C4_2OXY, .C_d_act_unpacked=buffer1, .C_d_wgt=conv4_d_wgt, .C_d_mean=conv4_d_mean, .C_d_var=conv4_d_var, .C_d_gamma=conv4_d_gamma, .C_d_beta=conv4_d_beta, .C_dKXY=C4_dKXY,  .C_dKZ=C4_dKZ,  .C_dPD=C4_dPD,  .C_dPL=C4_dPL,  .C_dOXY=C4_dOXY};
   convolutional_block(block4_conf);

   s2 = esp_timer_get_time();
   printf("fourth block (mus): %f\n", ((float)(s2-s1)));
   fflush(stdout);
   s1 = s2;

   //printFloatArray(buffer3, 100);

   struct identity_block_conf block5_conf = {.C_1_act_unpacked= buffer3, .C_1_act= buffer1, .C_2_act= buffer2, .C_3_act_unpacked= buffer1, .C_1_wgt= conv5_1_wgt, .C_2_wgt= conv5_2_wgt, .C_2_mean= conv5_2_mean, .C_2_var= conv5_2_var, .C_2_gamma= conv5_2_gamma, .C_2_beta= conv5_2_beta, .C_1_thresh= conv5_1_thresh, .C_1_sign= conv5_1_sign, .C_1XY= C5_1XY, .C_1Z= C5_1Z, .C_1KXY= C5_1KXY, .C_1KZ= C5_1KZ, .C_1PD= C5_1PD, .C_1PL= C5_1PL, .C_2XY= C5_2XY, .C_2Z= C5_2Z, .C_2KXY= C5_2KXY, .C_2KZ= C5_2KZ, .C_2PD= C5_2PD, .C_2PL= C5_2PL, .C_2OXY= C5_2OXY};
   identity_block(block5_conf);

   s2 = esp_timer_get_time();
   printf("fifth layer (mus): %f\n", ((float)(s2-s1)));
   fflush(stdout);
   s1 = s2;

   //printFloatArray(buffer1, 100);
   //pack(conv5_3_act_unpacked, conv5_3_act, C5_2OXY*C5_2OXY*C5_2Z);
   //printPackedArray(conv5_3_act, C5_2OXY*C5_2OXY*C5_2Z/pckWdt);

   averagePool1_1_int8(buffer3, buffer1, C5_1KZ, C5_2OXY, C5_2OXY);
   
   bwn_fc(buffer1, fc_wgt, buffer2, F1I, F1O);
   //printFloatArray(fc_out, F1O);
   normalize(buffer2, F1O);

   s2 = esp_timer_get_time();
   printf("last layer (mus): %f\n", ((float)(s2-s1)));
   fflush(stdout);
   s1 = s2;

   //printFloatArray(buffer2, 10);
}

void forward() {
   forward_new();
}

int main(void) {
   forward();
   
   return (EXIT_SUCCESS);
}
