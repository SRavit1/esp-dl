#include "model_qu.h"
#include "fd_forward.h"
#include "esp_log.h"

#include "dl_lib_matrix3d.h" //TEMPORARY- for debugging conv2_out

#include "printUtils.h"

static const char *TAG = "app_process";
const int EXP_TODO = -12;
dl_conv_mode mode = DL_C_IMPL;

dl_matrix3dq_t* copyMatrixQu(dl_matrix3dq_t* src) {
	if(!src) return NULL;
	ESP_LOGI(TAG, "Attempting to allocate quantized matrix of size (nwhc), (%d, %d, %d, %d)", src->n, src->w, src->h, src->c);
	
	dl_matrix3dq_t* dest = dl_matrix3dq_alloc(src->n, src->w, src->h, src->c, src->exponent);
	for (int i = 0; i < dest->n*dest->w*dest->h*dest->c; i++)
		dest->item[i] = src->item[i];
	return dest;
}

dl_matrix3d_t* model_forward(dl_matrix3du_t *input) {
	int64_t time_start = esp_timer_get_time();

	dl_matrix3dq_t *conv1_out = dl_matrix3duq_conv_common(input, &conv1_filter, &conv1_bias, 1, 1, PADDING_SAME, EXP_TODO, mode);

	int64_t time_conv1 = esp_timer_get_time();

	//dl_matrix3dq_batch_normalize(x, &batchnorm1_scale, &batchnorm1_offset);
	dl_matrix3dq_relu(conv1_out);
	int64_t time_relu1 = esp_timer_get_time();

	//Unfortunately, dl_matrix3dqq_conv_common seems to have a bug that causes it to return an incorrect result
	//dl_matrix3dq_t *conv2_out = dl_matrix3dqq_conv_common(conv1_out, &conv2_filter, &conv2_bias, 1, 1, PADDING_SAME, EXP_TODO, mode);	

	dl_matrix3d_t *conv1_out_f = dl_matrix3d_from_matrixq(conv1_out);
	dl_matrix3d_t *conv2_filter_f = dl_matrix3d_from_matrixq(&conv2_filter);
	dl_matrix3d_t *conv2_bias_f = dl_matrix3d_from_matrixq(&conv2_bias);
	dl_matrix3d_t *conv2_out_f = dl_matrix3dff_conv_common(conv1_out_f, conv2_filter_f, conv2_bias_f, 1, 1, PADDING_SAME);
	dl_matrix3dq_t *conv2_out = dl_matrixq_from_matrix3d(conv2_out_f);
	dl_matrix3d_free(conv1_out_f);
	dl_matrix3d_free(conv2_filter_f);
	dl_matrix3d_free(conv2_bias_f);
	dl_matrix3d_free(conv2_out_f);


	int64_t time_conv2 = esp_timer_get_time();

	//dl_matrix3dq_batch_normalize(y, &batchnorm2_scale, &batchnorm2_offset);
	dl_matrix3dq_relu(conv2_out);

	int64_t time_relu2 = esp_timer_get_time();

	//See conv2_out comment
	//dl_matrix3dq_t *conv3_out = dl_matrix3dqq_conv_common(conv2_out, &conv3_filter, &conv3_bias, 1, 1, PADDING_SAME, EXP_TODO, mode);
	conv2_out_f = dl_matrix3d_from_matrixq(conv2_out);
	dl_matrix3d_t *conv3_filter_f = dl_matrix3d_from_matrixq(&conv3_filter);
	dl_matrix3d_t *conv3_bias_f = dl_matrix3d_from_matrixq(&conv3_bias);
	dl_matrix3d_t *conv3_out_f = dl_matrix3dff_conv_common(conv2_out_f, conv3_filter_f, conv3_bias_f, 1, 1, PADDING_SAME);
	dl_matrix3dq_t *conv3_out = dl_matrixq_from_matrix3d(conv3_out_f);
	dl_matrix3d_free(conv2_out_f);
	dl_matrix3d_free(conv3_filter_f);
	dl_matrix3d_free(conv3_bias_f);
	dl_matrix3d_free(conv3_out_f);

	int64_t time_conv3 = esp_timer_get_time();

	//dl_matrix3dq_batch_normalize(y, &batchnorm3_scale, &batchnorm3_offset);

	dl_matrix3dq_t *add1_out = dl_matrix3dq_add(conv1_out, conv3_out, EXP_TODO);

	int64_t time_add1 = esp_timer_get_time();

	dl_matrix3dq_relu(add1_out);
	dl_matrix3dq_t* add1_out_copy = copyMatrixQu(add1_out);

	int64_t time_relu3 = esp_timer_get_time();

	//See conv2_out comment
	//dl_matrix3dq_t *conv4_out = dl_matrix3dqq_conv_common(add1_out, &conv4_filter, &conv4_bias, 2, 2, PADDING_SAME, EXP_TODO, mode);
	dl_matrix3d_t *add1_out_f = dl_matrix3d_from_matrixq(add1_out);
	dl_matrix3d_t *conv4_filter_f = dl_matrix3d_from_matrixq(&conv4_filter);
	dl_matrix3d_t *conv4_bias_f = dl_matrix3d_from_matrixq(&conv4_bias);
	dl_matrix3d_t *conv4_out_f = dl_matrix3dff_conv_common(add1_out_f, conv4_filter_f, conv4_bias_f, 2, 2, PADDING_SAME);
	dl_matrix3dq_t *conv4_out = dl_matrixq_from_matrix3d(conv4_out_f);
	dl_matrix3d_free(add1_out_f);
	dl_matrix3d_free(conv4_filter_f);
	dl_matrix3d_free(conv4_bias_f);
	dl_matrix3d_free(conv4_out_f);

	int64_t time_conv4 = esp_timer_get_time();

	//dl_matrix3dq_batch_normalize(y, &batchnorm4_scale, &batchnorm4_offset);
	dl_matrix3dq_relu(conv4_out);

	int64_t time_relu4 = esp_timer_get_time();

	//See conv2_out comment
	//dl_matrix3dq_t *conv5_out = dl_matrix3dqq_conv_common(conv4_out, &conv5_filter, &conv5_bias, 1, 1, PADDING_SAME, EXP_TODO, mode);
	conv4_out_f = dl_matrix3d_from_matrixq(conv4_out);
	dl_matrix3d_t *conv5_filter_f = dl_matrix3d_from_matrixq(&conv5_filter);
	dl_matrix3d_t *conv5_bias_f = dl_matrix3d_from_matrixq(&conv5_bias);
	dl_matrix3d_t *conv5_out_f = dl_matrix3dff_conv_common(conv4_out_f, conv5_filter_f, conv5_bias_f, 1, 1, PADDING_SAME);
	dl_matrix3dq_t *conv5_out = dl_matrixq_from_matrix3d(conv5_out_f);
	dl_matrix3d_free(conv4_out_f);
	dl_matrix3d_free(conv5_filter_f);
	dl_matrix3d_free(conv5_bias_f);
	dl_matrix3d_free(conv5_out_f);

	int64_t time_conv5 = esp_timer_get_time();

	//dl_matrix3dq_batch_normalize(y, &batchnorm5_scale, &batchnorm5_offset);

	//x is supposed to be relu3 output; y is supposed to be conv5 output

	//See conv2_out comment
	//dl_matrix3dq_t *conv6_out = dl_matrix3dqq_conv_common(add1_out_copy, &conv6_filter, &conv6_bias, 2, 2, PADDING_VALID, EXP_TODO, mode);
	dl_matrix3d_t *add1_out_copy_f = dl_matrix3d_from_matrixq(add1_out_copy);
	dl_matrix3d_t *conv6_filter_f = dl_matrix3d_from_matrixq(&conv6_filter);
	dl_matrix3d_t *conv6_bias_f = dl_matrix3d_from_matrixq(&conv6_bias);
	dl_matrix3d_t *conv6_out_f = dl_matrix3dff_conv_common(add1_out_copy_f, conv6_filter_f, conv6_bias_f, 2, 2, PADDING_VALID);
	dl_matrix3dq_t *conv6_out = dl_matrixq_from_matrix3d(conv6_out_f);
	dl_matrix3d_free(add1_out_copy_f);
	dl_matrix3d_free(conv6_filter_f);
	dl_matrix3d_free(conv6_bias_f);
	dl_matrix3d_free(conv6_out_f);

	int64_t time_conv6 = esp_timer_get_time();

	//x is supposed to be conv6 output; y is supposed to be conv5 output

	dl_matrix3dq_t *add2_out = dl_matrix3dq_add(conv5_out, conv6_out, EXP_TODO);

	int64_t time_add2 = esp_timer_get_time();

	dl_matrix3dq_relu(add2_out);
	dl_matrix3dq_t *add2_out_copy = copyMatrixQu(add2_out);

	int64_t time_relu5 = esp_timer_get_time();

	//See conv2_out comment
	//dl_matrix3dq_t *conv7_out = dl_matrix3dqq_conv_common(add2_out, &conv7_filter, &conv7_bias, 2, 2, PADDING_SAME, EXP_TODO, mode);
	dl_matrix3d_t *add2_out_f = dl_matrix3d_from_matrixq(add2_out);
	dl_matrix3d_t *conv7_filter_f = dl_matrix3d_from_matrixq(&conv7_filter);
	dl_matrix3d_t *conv7_bias_f = dl_matrix3d_from_matrixq(&conv7_bias);
	dl_matrix3d_t *conv7_out_f = dl_matrix3dff_conv_common(add2_out_f, conv7_filter_f, conv7_bias_f, 2, 2, PADDING_SAME);
	dl_matrix3dq_t *conv7_out = dl_matrixq_from_matrix3d(conv7_out_f);
	dl_matrix3d_free(add2_out_f);
	dl_matrix3d_free(conv7_filter_f);
	dl_matrix3d_free(conv7_bias_f);
	dl_matrix3d_free(conv7_out_f);

	int64_t time_conv7 = esp_timer_get_time();

	//dl_matrix3dq_batch_normalize(y, &batchnorm6_scale, &batchnorm6_offset);
	dl_matrix3dq_relu(conv7_out);

	int64_t time_relu6 = esp_timer_get_time();

	//See conv2_out comment
	//dl_matrix3dq_t *conv8_out = dl_matrix3dqq_conv_common(conv7_out, &conv8_filter, &conv8_bias, 1, 1, PADDING_SAME, EXP_TODO, mode);
	conv7_out_f = dl_matrix3d_from_matrixq(conv7_out);
	dl_matrix3d_t *conv8_filter_f = dl_matrix3d_from_matrixq(&conv8_filter);
	dl_matrix3d_t *conv8_bias_f = dl_matrix3d_from_matrixq(&conv8_bias);
	dl_matrix3d_t *conv8_out_f = dl_matrix3dff_conv_common(conv7_out_f, conv8_filter_f, conv8_bias_f, 1, 1, PADDING_SAME);
	dl_matrix3dq_t *conv8_out = dl_matrixq_from_matrix3d(conv8_out_f);
	dl_matrix3d_free(conv7_out_f);
	dl_matrix3d_free(conv8_filter_f);
	dl_matrix3d_free(conv8_bias_f);
	dl_matrix3d_free(conv8_out_f);

	int64_t time_conv8 = esp_timer_get_time();

	//dl_matrix3dq_batch_normalize(y, &batchnorm7_scale, &batchnorm7_offset);

	//See conv2_out comment
	//dl_matrix3dq_t *conv9_out = dl_matrix3dqq_conv_common(add2_out_copy, &conv9_filter, &conv9_bias, 2, 2, PADDING_VALID, EXP_TODO, mode);
	dl_matrix3d_t *add2_out_copy_f = dl_matrix3d_from_matrixq(add2_out_copy);
	dl_matrix3d_t *conv9_filter_f = dl_matrix3d_from_matrixq(&conv9_filter);
	dl_matrix3d_t *conv9_bias_f = dl_matrix3d_from_matrixq(&conv9_bias);
	dl_matrix3d_t *conv9_out_f = dl_matrix3dff_conv_common(add2_out_copy_f, conv9_filter_f, conv9_bias_f, 2, 2, PADDING_VALID);
	dl_matrix3dq_t *conv9_out = dl_matrixq_from_matrix3d(conv9_out_f);
	dl_matrix3d_free(add2_out_copy_f);
	dl_matrix3d_free(conv9_filter_f);
	dl_matrix3d_free(conv9_bias_f);
	dl_matrix3d_free(conv9_out_f);

	int64_t time_conv9 = esp_timer_get_time();

	dl_matrix3dq_t *add3_out = dl_matrix3dq_add(conv8_out, conv9_out, EXP_TODO);

	int64_t time_add3 = esp_timer_get_time();

	dl_matrix3dq_relu(add3_out);

	int64_t time_relu7 = esp_timer_get_time();

	dl_matrix3dq_free(conv1_out);
	dl_matrix3dq_free(conv2_out);
	dl_matrix3dq_free(conv3_out);
	dl_matrix3dq_free(conv4_out);
	dl_matrix3dq_free(conv5_out);
	dl_matrix3dq_free(conv6_out);
	dl_matrix3dq_free(conv7_out);


	dl_matrix3dq_t *pool1_out = dl_matrix3dq_pooling(add3_out, 8, 8, 8, 8, PADDING_VALID, DL_POOLING_AVG);

	dl_matrix3dq_free(conv8_out);
	dl_matrix3dq_free(conv9_out);
	dl_matrix3dq_free(add1_out);
	dl_matrix3dq_free(add1_out_copy);
	dl_matrix3dq_free(add2_out);
	dl_matrix3dq_free(add2_out_copy);
	dl_matrix3dq_free(add3_out);

	int64_t time_pool1 = esp_timer_get_time();

	pool1_out->c *= pool1_out->w * pool1_out->h;
	pool1_out->w = 1;
	pool1_out->h = 1;
	
	//Like with conv_2 and beyond, dl_matrix3dqq_fc_with_bias seems buggy
	//dl_matrix3dq_t *output = dl_matrix3dq_alloc(1, 1, 1, 10, EXP_TODO);
	//dl_matrix3dqq_fc_with_bias(output, pool1_out, &fc1_filter, &fc1_bias, mode, "fc");
	dl_matrix3d_t *output_f = dl_matrix3d_alloc(1, 1, 1, 10);
	dl_matrix3d_t *pool1_out_f = dl_matrix3d_from_matrixq(pool1_out);
	dl_matrix3d_t *fc1_filter_f = dl_matrix3d_from_matrixq(&fc1_filter);
	dl_matrix3d_t *fc1_bias_f = dl_matrix3d_from_matrixq(&fc1_bias);
	dl_matrix3dff_fc_with_bias(output_f, pool1_out_f, fc1_filter_f, fc1_bias_f);
	dl_matrix3dq_t *output = dl_matrixq_from_matrix3d(output_f);
	dl_matrix3d_free(pool1_out_f);
	dl_matrix3d_free(fc1_filter_f);
	dl_matrix3d_free(fc1_bias_f);
	dl_matrix3d_free(output_f);

	dl_matrix3dq_free(pool1_out);

	int64_t time_fc1 = esp_timer_get_time();

    ESP_LOGI(TAG, "forward pass finished in %lld mu_s.", (time_fc1 - time_start));

    ESP_LOGI(TAG, "conv1 time: %lld mu_s.", (time_conv1 - time_start));
    ESP_LOGI(TAG, "relu1 time: %lld mu_s.", (time_relu1 - time_conv1));
    ESP_LOGI(TAG, "conv2 time: %lld mu_s.", (time_conv2 - time_relu1));
    ESP_LOGI(TAG, "relu2 time: %lld mu_s.", (time_relu2 - time_conv2));
    ESP_LOGI(TAG, "conv3 time: %lld mu_s.", (time_conv3 - time_relu2));
    ESP_LOGI(TAG, "add1 time: %lld mu_s.", (time_add1 - time_conv3));
    ESP_LOGI(TAG, "relu3 time: %lld mu_s.", (time_relu3 - time_add1));
    ESP_LOGI(TAG, "conv4 time: %lld mu_s.", (time_conv4 - time_relu3));
    ESP_LOGI(TAG, "relu4 time: %lld mu_s.", (time_relu4 - time_conv4));
    ESP_LOGI(TAG, "conv5 time: %lld mu_s.", (time_conv5 - time_relu4));
    ESP_LOGI(TAG, "conv6 time: %lld mu_s.", (time_conv6 - time_conv5));
    ESP_LOGI(TAG, "add2 time: %lld mu_s.", (time_add2 - time_conv6));
    ESP_LOGI(TAG, "relu5 time: %lld mu_s.", (time_relu5 - time_add2));
    ESP_LOGI(TAG, "conv7 time: %lld mu_s.", (time_conv7 - time_relu5));
    ESP_LOGI(TAG, "relu6 time: %lld mu_s.", (time_relu6 - time_conv7));
    ESP_LOGI(TAG, "conv8 time: %lld mu_s.", (time_conv8 - time_relu6));
    ESP_LOGI(TAG, "conv9 time: %lld mu_s.", (time_conv9 - time_conv8));
    ESP_LOGI(TAG, "add3 time: %lld mu_s.", (time_add3 - time_conv9));
    ESP_LOGI(TAG, "relu7 time: %lld mu_s.", (time_relu7 - time_add3));
    ESP_LOGI(TAG, "pool1 time: %lld mu_s.", (time_pool1 - time_relu7));
    ESP_LOGI(TAG, "fc1 time: %lld mu_s.", (time_fc1 - time_pool1));

    dl_matrix3d_t* output_final = dl_matrix3d_from_matrixq(output);
	return output_final;
}