#include "model.h"
#include "esp_log.h"

static const char *TAG = "app_main";

int main() {
	dl_matrix3d_t *input = dl_matrix3d_alloc(1, 1, 1, 576);
	dl_matrix3d_t *fc1_out = dl_matrix3d_alloc(1, 1, 1, 128);
	dl_matrix3d_t *fc2_out = dl_matrix3d_alloc(1, 1, 1, 128);
	dl_matrix3d_t *fc3_out = dl_matrix3d_alloc(1, 1, 1, 128);
	dl_matrix3d_t *fc4_out = dl_matrix3d_alloc(1, 1, 1, 128);
	dl_matrix3d_t *fc5_out = dl_matrix3d_alloc(1, 1, 1, 8);
	dl_matrix3d_t *fc6_out = dl_matrix3d_alloc(1, 1, 1, 128);
	dl_matrix3d_t *fc7_out = dl_matrix3d_alloc(1, 1, 1, 128);
	dl_matrix3d_t *fc8_out = dl_matrix3d_alloc(1, 1, 1, 128);
	dl_matrix3d_t *fc9_out = dl_matrix3d_alloc(1, 1, 1, 128);
	dl_matrix3d_t *fc10_out = dl_matrix3d_alloc(1, 1, 1, 576);
	
	int64_t start_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc1_out, input, &fc1_filter);
	dl_matrix3d_relu(fc1_out);
	int64_t fc1_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc2_out, fc1_out, &fc2_filter);
	dl_matrix3d_relu(fc2_out);
	int64_t fc2_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc3_out, fc2_out, &fc3_filter);
	dl_matrix3d_relu(fc3_out);
	int64_t fc3_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc4_out, fc3_out, &fc4_filter);
	dl_matrix3d_relu(fc4_out);
	int64_t fc4_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc5_out, fc4_out, &fc5_filter);
	dl_matrix3d_relu(fc5_out);
	int64_t fc5_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc6_out, fc5_out, &fc6_filter);
	dl_matrix3d_relu(fc6_out);
	int64_t fc6_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc7_out, fc6_out, &fc7_filter);
	dl_matrix3d_relu(fc7_out);
	int64_t fc7_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc8_out, fc7_out, &fc8_filter);
	dl_matrix3d_relu(fc8_out);
	int64_t fc8_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc9_out, fc8_out, &fc9_filter);
	dl_matrix3d_relu(fc9_out);
	int64_t fc9_time = esp_timer_get_time();

	dl_matrix3dff_fc(fc10_out, fc9_out, &fc10_filter);
	dl_matrix3d_relu(fc10_out);
	int64_t fc10_time = esp_timer_get_time();

	ESP_LOGI(TAG, "forward pass took %lld microseconds", (fc10_time - start_time));
	ESP_LOGI(TAG, "fc1 took %lld microseconds", (fc1_time - start_time));
	ESP_LOGI(TAG, "fc2 took %lld microseconds", (fc2_time - fc1_time));
	ESP_LOGI(TAG, "fc3 took %lld microseconds", (fc3_time - fc2_time));
	ESP_LOGI(TAG, "fc4 took %lld microseconds", (fc4_time - fc3_time));
	ESP_LOGI(TAG, "fc5 took %lld microseconds", (fc5_time - fc4_time));
	ESP_LOGI(TAG, "fc6 took %lld microseconds", (fc6_time - fc5_time));
	ESP_LOGI(TAG, "fc7 took %lld microseconds", (fc7_time - fc6_time));
	ESP_LOGI(TAG, "fc8 took %lld microseconds", (fc8_time - fc7_time));
	ESP_LOGI(TAG, "fc9 took %lld microseconds", (fc9_time - fc8_time));
	ESP_LOGI(TAG, "fc10 took %lld microseconds", (fc10_time - fc9_time));

	dl_matrix3d_free(input);
	dl_matrix3d_free(fc1_out);
	dl_matrix3d_free(fc2_out);
	dl_matrix3d_free(fc3_out);
	dl_matrix3d_free(fc4_out);
	dl_matrix3d_free(fc5_out);
	dl_matrix3d_free(fc6_out);
	dl_matrix3d_free(fc7_out);
	dl_matrix3d_free(fc8_out);
	dl_matrix3d_free(fc9_out);
	dl_matrix3d_free(fc10_out);
}