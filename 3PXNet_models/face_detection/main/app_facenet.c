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
#include "app_facenet.h"
#include "sdkconfig.h"
#include "facenet_full_prec.h" // "facenet_full_prec_qu.h"

static const char *TAG = "app_process";

mtmn_config_t init_config()
{
    mtmn_config_t mtmn_config = {0};
    mtmn_config.type = FAST;
    mtmn_config.min_face = 80;
    mtmn_config.pyramid = 0.707;
    mtmn_config.pyramid_times = 4;
    mtmn_config.p_threshold.score = 0.6;
    mtmn_config.p_threshold.nms = 0.7;
    mtmn_config.p_threshold.candidate_number = 20;
    mtmn_config.r_threshold.score = 0.7;
    mtmn_config.r_threshold.nms = 0.7;
    mtmn_config.r_threshold.candidate_number = 10;
    mtmn_config.o_threshold.score = 0.7;
    mtmn_config.o_threshold.nms = 0.7;
    mtmn_config.o_threshold.candidate_number = 1;

    return mtmn_config;
}

void task_process (void *arg)
{/*{{{*/
    size_t frame_num = 0;
    dl_matrix3du_t *image_matrix = NULL;
    camera_fb_t *fb = NULL;
    
    /* 1. Load configuration for detection */
    mtmn_config_t mtmn_config = init_config();

    dl_conv_mode conv_mode = DL_XTENSA_IMPL; //DL_C_IMPL
    do
    {        
        dl_matrix3du_t *pnet_image = dl_matrix3du_alloc(1, 12, 12, 3);
        for (int i = 0; i < 1*12*12*3; i++)
            pnet_image->item[i] = 1;
        //pnet_lite_f_custom(pnet_image);
        //pnet_lite_q_custom(pnet_image, conv_mode);

        dl_matrix3du_t *rnet_image = dl_matrix3du_alloc(1, 24, 24, 3);
        for (int i = 0; i < 1*24*24*3; i++)
            rnet_image->item[i] = 1;
        //rnet_lite_f_with_score_verify_custom(rnet_image, 1);
        //rnet_lite_q_with_score_verify_custom(rnet_image, 1, conv_mode);

        dl_matrix3du_t *onet_image = dl_matrix3du_alloc(1, 48, 48, 3);
        for (int i = 0; i < 1*48*48*3; i++)
            onet_image->item[i] = 1;
        onet_lite_f_with_score_verify_custom(onet_image, 1);
        //onet_lite_q_with_score_verify_custom(onet_image, 1, conv_mode);

        /*int64_t start_time = esp_timer_get_time();
        // 2. Get one image with camera * /
        fb = esp_camera_fb_get();
        if (!fb)
        {
            ESP_LOGE(TAG, "Camera capture failed");
            continue;
        }
        int64_t fb_get_time = esp_timer_get_time();
        ESP_LOGI(TAG, "Get one frame in %lld ms.", (fb_get_time - start_time) / 1000);

        // 3. Allocate image matrix to store RGB data * /
        image_matrix = dl_matrix3du_alloc(1, fb->width, fb->height, 3);

        // 4. Transform image to RGB * /
        uint32_t res = fmt2rgb888(fb->buf, fb->len, fb->format, image_matrix->item);
        if (true != res)
        {
            ESP_LOGE(TAG, "fmt2rgb888 failed, fb: %d", fb->len);
            dl_matrix3du_free(image_matrix);
            continue;
        }

        esp_camera_fb_return(fb);

        // 5. Do face detection * /

        box_array_t *net_boxes = face_detect(image_matrix, &mtmn_config);
        ESP_LOGI(TAG, "Detection time consumption: %lldms", (esp_timer_get_time() - fb_get_time) / 1000);

        if (net_boxes)
        {
            frame_num++;
            ESP_LOGI(TAG, "DETECTED: %d\n", frame_num);
            dl_lib_free(net_boxes->score);
            dl_lib_free(net_boxes->box);
            dl_lib_free(net_boxes->landmark);
            dl_lib_free(net_boxes);
        }

        dl_matrix3du_free(image_matrix);
        */

    } while(1);
}/*}}}*/

void app_facenet_main()
{
    xTaskCreatePinnedToCore(task_process, "process", 4 * 1024, NULL, 5, NULL, 1);
}
