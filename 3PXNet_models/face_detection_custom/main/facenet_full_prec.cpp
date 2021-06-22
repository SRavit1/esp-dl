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
	//Test: Running input through a single convolutional layer with dummy weights
	//In this test, nullptr is returned, so the output is not connected to the output of the conv operator
	//Weights taken from mnist example, defined in include/facenet_full_prec.h

	dl_matrix3d_t *o1_1 = dl_matrix3dff_conv_common(in, &conv2d_kernel, &conv2d_bias, 1, 1, PADDING_SAME);
	return nullptr;
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