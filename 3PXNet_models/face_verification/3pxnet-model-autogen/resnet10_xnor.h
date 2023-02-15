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

#define act_bw 1
#define weight_bw 1