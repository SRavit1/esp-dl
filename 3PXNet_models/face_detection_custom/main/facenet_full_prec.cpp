#include "app_facenet.h"

#ifdef FACENET_FULL_PREC

#include "facenet_full_prec.h"
#include "dl_lib_matrix3d.h"
#include "mtmn.h"
#include "fd_forward.h"

//Docstrings and headers taken from mtmn.h
/**
 * @brief Forward the pnet process, coarse detection. Calculate in float.
 *
 * @param in        Image matrix, rgb888 format, size is 320x240
 * @return          Scores for every pixel, and box offset with respect.
 */
mtmn_net_t *pnet_lite_f(dl_matrix3du_t *in) {
	dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &pnet_conv2d_kernel1, &pnet_conv2d_bias1, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_1);
    dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);

    dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &pnet_conv2d_kernel2, &pnet_conv2d_bias2, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_2);

    dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_conv_2, &pnet_conv2d_kernel3, &pnet_conv2d_bias3, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_3);

    dl_matrix3d_t *category = dl_matrix3dff_conv_common(out_conv_3, &pnet_conv2d_kernel4, &pnet_conv2d_bias4, 1, 1, PADDING_VALID);
    dl_matrix3d_softmax(category); //TODO: How to indicate that this should be done over axis 3?
    dl_matrix3d_t *offset = dl_matrix3dff_conv_common(out_conv_3, &pnet_conv2d_kernel5, &pnet_conv2d_bias5, 1, 1, PADDING_VALID);

    //TODO: Call free as soon as tensors are no longer live
    dl_matrix3d_free(out_conv_1);
    dl_matrix3d_free(out_pool_1);
    dl_matrix3d_free(out_conv_2);
    dl_matrix3d_free(out_conv_3);

    mtmn_net_t *result = (mtmn_net_t*) dl_lib_calloc(1, sizeof *result, 0);
    result->category = category;
    result->offset = offset;
    result->landmark = nullptr;

	return result;
}

/**
 * @brief Forward the rnet process, fine determine the boxes from pnet. Calculate in float.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, and box offset with respect.
 */
mtmn_net_t *rnet_lite_f_with_score_verify(dl_matrix3du_t *in, float threshold) { return nullptr; }

/**
 * @brief Forward the onet process, fine determine the boxes from rnet. Calculate in float.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, box offset, and landmark with respect.
 */
mtmn_net_t *onet_lite_f_with_score_verify(dl_matrix3du_t *in, float threshold) { return nullptr; }

/**
 * @brief Forward the pnet process, coarse detection. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format, size is 320x240
 * @return          Scores for every pixel, and box offset with respect.
 */
mtmn_net_t *pnet_lite_q(dl_matrix3du_t *in, dl_conv_mode mode) { return nullptr; }

/**
 * @brief Forward the rnet process, fine determine the boxes from pnet. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, and box offset with respect.
 */
mtmn_net_t *rnet_lite_q_with_score_verify(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) { return nullptr; }

/**
 * @brief Forward the onet process, fine determine the boxes from rnet. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, box offset, and landmark with respect.
 */
mtmn_net_t *onet_lite_q_with_score_verify(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) { return nullptr; }

/**
 * @brief Forward the pnet process, coarse detection. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format, size is 320x240
 * @return          Scores for every pixel, and box offset with respect.
 */
mtmn_net_t *pnet_heavy_q(dl_matrix3du_t *in, dl_conv_mode mode) { return nullptr; }

/**
 * @brief Forward the rnet process, fine determine the boxes from pnet. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, and box offset with respect.
 */
mtmn_net_t *rnet_heavy_q_with_score_verify(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) { return nullptr; }

/**
 * @brief Forward the onet process, fine determine the boxes from rnet. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, box offset, and landmark with respect.
 */
mtmn_net_t *onet_heavy_q_with_score_verify(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) { return nullptr; }

#endif