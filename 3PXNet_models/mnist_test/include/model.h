#include "cnn.h"
#include "dl_lib_matrix3d.h"
#include "source.h"

#define _3PXNET_LIB_IMPL

dl_matrix3d_t *mnist_model (dl_matrix3d_t *image) {
    dl_matrix3d_t *result = 0;
#if defined(ESP_LIB_IMPL)
    dl_matrix3d_t *o1_1 = dl_matrix3dff_conv_common(image, &conv2d_kernel, &conv2d_bias, 1, 1, PADDING_SAME);
    dl_matrix3d_t *o1_2 = dl_matrix3d_pooling(o1_1, 2, 2, 2, 2, PADDING_VALID, DL_POOLING_MAX);
    dl_matrix3d_relu(o1_2);
    dl_matrix3d_t *o2_1 = dl_matrix3dff_conv_common(o1_2, &conv2d_1_kernel, &conv2d_1_bias, 1, 1, PADDING_SAME);
    dl_matrix3d_t *o2_2 = dl_matrix3d_pooling(o2_1, 2, 2, 2, 2, PADDING_VALID, DL_POOLING_MAX);
    dl_matrix3d_relu(o2_2);
    dl_matrix3d_t *o3_1 = dl_matrix3dff_conv_common(o2_2, &conv2d_2_kernel, &conv2d_2_bias, 1, 1, PADDING_SAME);
    dl_matrix3d_t *o3_2 = dl_matrix3d_pooling(o3_1, 2, 2, 2, 2, PADDING_VALID, DL_POOLING_MAX);
    dl_matrix3d_relu(o3_2);
    dl_matrix3d_t *o4_1 = dl_matrix3d_alloc(1, 1, 1, dense_kernel.h);
    dl_matrix3dff_fc_with_bias(o4_1, o3_2, &dense_kernel, &dense_bias);
    dl_matrix3d_relu(o4_1);
    dl_matrix3d_t *o4_2 = dl_matrix3d_alloc(1, 1, 1, dense_1_kernel.h);
    dl_matrix3dff_fc_with_bias(o4_2, o4_1, &dense_1_kernel, &dense_1_bias);

    result = o4_2;
#elif defined(_3PXNET_LIB_IMPL)
    uint8_t curr_im[784];
    for (int i = 0; i < 784; i++) {
        curr_im[i] = (uint8_t) (image->item[i]*255);
    }

    //output defined in source.h
	packBinThrsArr(curr_im, l1act_bin, F1I, 1);
	FcXnorWrap(l1act_bin, l1wght, F1I, F1O, l2act_bin, bn1thr, bn1sign);
	int res = FcXnorNoBinWrap(l2act_bin, l2wght, F2I, F2O, output, bn2mean, bn2var, bn2gamma, bn2beta);

    result = (dl_matrix3d_t *)dl_lib_calloc(1, sizeof(dl_matrix3d_t), 0);
    result->n = 1;
    result->h = 1;
    result->w = 1;
    result->c = 10;
    result->stride = 10;
    result->item = output;

#else
    //ERROR: At least one of the above macros should be defined
#endif
    return result;
}