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

#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "printUtils.h"
#include "app_model.h"
#include "image_util.h"

extern dl_matrix3d_t* model_forward(dl_matrix3du_t *input);

static const char *TAG = "app_process";

void task_process (void *arg)
{
    dl_matrix3du_t *image_matrix = NULL;
    camera_fb_t *fb = NULL;

    do
    {
        int64_t start_time = esp_timer_get_time();
        /* 2. Get one image with camera */
        fb = esp_camera_fb_get();
        if (!fb)
        {
            ESP_LOGE(TAG, "Camera capture failed");
            continue;
        }
        int64_t fb_get_time = esp_timer_get_time();
        ESP_LOGI(TAG, "Get one frame in %lld ms.", (fb_get_time - start_time) / 1000);

        /* 3. Allocate image matrix to store RGB data */
        ESP_LOGI(TAG, "Attempting to allocate matrix of size %zu x %zu.", fb->width, fb->height);
        size_t width = fb->width;
        size_t height = fb->height;
        image_matrix = dl_matrix3du_alloc(1, width, height, 3);
        ESP_LOGI(TAG, "Successfully allocated matrix.");

        /* 4. Transform image to RGB */
        uint32_t res = fmt2rgb888(fb->buf, fb->len, fb->format, image_matrix->item);
        if (true != res)
        {
            ESP_LOGE(TAG, "fmt2rgb888 failed, fb: %d", fb->len);
            dl_matrix3du_free(image_matrix);
            continue;
        }

        esp_camera_fb_return(fb);

        dl_matrix3du_t *image_matrix_resized = dl_matrix3du_alloc(1, 32, 32, 3);
        for (int i = 0; i < 1*32*32*3; i++)
            image_matrix_resized->item[i] = 1;
        //image_resize_linear(image_matrix_resized->item, image_matrix->item, 32, 32, 3, 320, 240);

        /* 5. Do image classification */
        dl_matrix3d_t* result = model_forward(image_matrix_resized);
        printMatrix(result);

        // Use image_matrix

        ESP_LOGI(TAG, "Test log message!");

        dl_matrix3du_free(image_matrix);
        dl_matrix3du_free(image_matrix_resized);
        dl_matrix3d_free(result);

    } while(1);
}

void app_model_main()
{
    xTaskCreatePinnedToCore(task_process, "process", 4 * 1024, NULL, 5, NULL, 1);
}
