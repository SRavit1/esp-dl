#include "model.h"
#include "dl_lib_matrix3d.h"
#include "fd_forward.h"
#include "esp_log.h"

#include "printUtils.h"

static const char *TAG = "app_process";

dl_matrix3d_t* model_forward(dl_matrix3du_t *input) {
	dl_matrix3d_t *x, *y;
	dl_matrix3d_t *x_temp, *y_temp;

	int64_t time_start = esp_timer_get_time();

	x = dl_matrix3duf_conv_common(input, &conv1_filter, &conv1_bias, 1, 1, PADDING_SAME);

	int64_t time_conv1 = esp_timer_get_time();

	//dl_matrix3d_batch_normalize(x, &batchnorm1_scale, &batchnorm1_offset);
	dl_matrix3d_relu(x);

	int64_t time_relu1 = esp_timer_get_time();

	y = dl_matrix3dff_conv_common(x, &conv2_filter, &conv2_bias, 1, 1, PADDING_SAME);

	int64_t time_conv2 = esp_timer_get_time();

	//dl_matrix3d_batch_normalize(y, &batchnorm2_scale, &batchnorm2_offset);
	dl_matrix3d_relu(y);

	int64_t time_relu2 = esp_timer_get_time();

	y_temp = dl_matrix3dff_conv_common(y, &conv3_filter, &conv3_bias, 1, 1, PADDING_SAME);

	int64_t time_conv3 = esp_timer_get_time();

	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm3_scale, &batchnorm3_offset);

	x_temp = dl_matrix3d_add(x, y);

	int64_t time_add1 = esp_timer_get_time();

	dl_matrix3d_free(x);
	x = x_temp;
	dl_matrix3d_relu(x);

	int64_t time_relu3 = esp_timer_get_time();

	y_temp = dl_matrix3dff_conv_common(x, &conv4_filter, &conv4_bias, 2, 2, PADDING_SAME);

	int64_t time_conv4 = esp_timer_get_time();

	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm4_scale, &batchnorm4_offset);
	dl_matrix3d_relu(y);

	int64_t time_relu4 = esp_timer_get_time();

	y_temp = dl_matrix3dff_conv_common(y, &conv5_filter, &conv5_bias, 1, 1, PADDING_SAME);

	int64_t time_conv5 = esp_timer_get_time();

	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm5_scale, &batchnorm5_offset);

	x = dl_matrix3dff_conv_common(x, &conv6_filter, &conv6_bias, 2, 2, PADDING_VALID);

	int64_t time_conv6 = esp_timer_get_time();

	x_temp = dl_matrix3d_add(x, y);

	int64_t time_add2 = esp_timer_get_time();

	dl_matrix3d_free(x);
	x = x_temp;
	dl_matrix3d_relu(x);

	int64_t time_relu5 = esp_timer_get_time();

	y_temp = dl_matrix3dff_conv_common(x, &conv7_filter, &conv7_bias, 2, 2, PADDING_SAME);

	int64_t time_conv7 = esp_timer_get_time();

	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm6_scale, &batchnorm6_offset);
	dl_matrix3d_relu(y);

	int64_t time_relu6 = esp_timer_get_time();

	y_temp = dl_matrix3dff_conv_common(y, &conv8_filter, &conv8_bias, 1, 1, PADDING_SAME);

	int64_t time_conv8 = esp_timer_get_time();

	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm7_scale, &batchnorm7_offset);

	x_temp = dl_matrix3dff_conv_common(x, &conv9_filter, &conv9_bias, 2, 2, PADDING_VALID);

	int64_t time_conv9 = esp_timer_get_time();

	dl_matrix3d_free(x);
	x = x_temp;
	x_temp = dl_matrix3d_add(x, y);

	int64_t time_add3 = esp_timer_get_time();

	dl_matrix3d_free(x);
	x = x_temp;
	dl_matrix3d_relu(x);

	int64_t time_relu7 = esp_timer_get_time();

	x_temp = dl_matrix3d_pooling(x, 8, 8, 8, 8, PADDING_VALID, DL_POOLING_AVG);

	int64_t time_pool1 = esp_timer_get_time();

	dl_matrix3d_free(x);
	x = x_temp;
	y = x;

	y->c *= y->w * y->h;
	y->w = 1;
	y->h = 1;

	x = dl_matrix3d_alloc(1, 1, 1, 10);
	
	dl_matrix3dff_fc_with_bias(x, y, &fc1_filter, &fc1_bias);
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

	return x;
}