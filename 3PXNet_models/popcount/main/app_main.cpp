/* ESPRESSIF MIT License
 * 
 * Copyright (c) 2018 <ESPRESSIF SYSTEMS (SHANGHAI) PTE LTD>
 * 
 * Permission is hereby granted for use on all ESPRESSIF SYSTEMS products, in which case,
 * it is free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "sdkconfig.h"
#include "dl_lib_matrix3d.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "utils.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

const char *TAG = "app_main";

void run_experiments(void* arg) {
    //float multiplication measurement
    const int n1 = 100;
    const int n2 = 100;
    const int n3 = 10000;
    const int n4 = 10000;
    const int n5 = 10000;
    do {
        //EXPERIMENT 1
        float *item1 = (float *)dl_lib_calloc(n1, sizeof(float), 0);
        for (int i = 0; i < n1; i++)
            item1[i] = 256;
        int64_t exp1_start = esp_timer_get_time();
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                int val = item1[i] * item1[j];
            }
        }
        int64_t exp1_finish = esp_timer_get_time();

        ESP_LOGI(TAG, "exp1: measure average duration of floating point memory access and multiplication");
        ESP_LOGI(TAG, "exp1 average duration (microseconds): %f", ((float )(exp1_finish-exp1_start))/(n1*n1));

        //EXPERIMENT 2
        int32_t *item2 = (int32_t *)dl_lib_calloc(n2, sizeof(int32_t), 0);
        for (int i = 0; i < n2; i++)
            item2[i] = 256;
        int64_t exp2_start = esp_timer_get_time();
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                int val = ~(item2[i] ^ item2[j]);
            }
        }
        int64_t exp2_finish = esp_timer_get_time();

        ESP_LOGI(TAG, "exp2: measure average duration of int32 memory access and xnor");
        ESP_LOGI(TAG, "exp2 average duration (microseconds): %f", ((float )(exp2_finish-exp2_start))/(n2*n2*32));

        //EXPERIMENT 3
        int32_t *item3 = (int32_t *)dl_lib_calloc(n3, sizeof(int32_t), 0);
        for (int i = 0; i < n3; i++)
            item3[i] = 256;
        int64_t exp3_start = esp_timer_get_time();
        for (int i = 0; i < n3; i++) {
            int val = popcount(item3[i]);
        }
        int64_t exp3_finish = esp_timer_get_time();

        ESP_LOGI(TAG, "exp3: measure average duration of int32 memory access and software popcount");
        ESP_LOGI(TAG, "exp3 average duration (microseconds): %f", ((float )(exp3_finish-exp3_start))/(n3*32));


        //EXPERIMENT 4
        int32_t *item4 = (int32_t *)dl_lib_calloc(n4, sizeof(int32_t), 0);
        for (int i = 0; i < n4; i++)
            item4[i] = 256;
        int64_t exp4_start = esp_timer_get_time();
        for (int i = 0; i < n4; i++) {
            int val = __builtin_popcountl(item4[i]);
        }
        int64_t exp4_finish = esp_timer_get_time();

        ESP_LOGI(TAG, "exp4: measure average duration of int32 memory access and builtin popcount");
        ESP_LOGI(TAG, "exp4 average duration (microseconds): %f", ((float )(exp4_finish-exp4_start))/(n4*32));

        //EXPERIMENT 5
        int32_t *item5 = (int32_t *)dl_lib_calloc(n5, sizeof(int32_t), 0);
        for (int i = 0; i < n5; i++)
            item4[i] = 256;
        int64_t exp5_start = esp_timer_get_time();
        for (int i = 0; i < n5; i++) {
            int val = popcount_naive(item5[i]);
        }
        int64_t exp5_finish = esp_timer_get_time();

        ESP_LOGI(TAG, "exp5: measure average duration of int32 memory access and naive popcount");
        ESP_LOGI(TAG, "exp5 average duration (microseconds): %f", ((float )(exp5_finish-exp5_start))/(n5*32));

        dl_lib_free(item1);
        dl_lib_free(item2);
        dl_lib_free(item3);
        dl_lib_free(item4);
        dl_lib_free(item5);
    } while(1);
}

extern "C" void app_main()
{
    xTaskCreatePinnedToCore(run_experiments, "process", 4 * 1024, NULL, 5, NULL, 1);
}
