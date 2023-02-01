#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "protocol_examples_common.h"

#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include "lwip/netdb.h"
#include "lwip/dns.h"

#include "resnet10_xnor.c"
#include "get_weights_http.h"

void test(void *arg)
{
    while(1)
    {
        main();

        vTaskDelay(100);
    }
}

static void get_weights(void *pvParameters) {
    while(1) {
        printf("Largest free block: %zu. Free size: %zu. Minimum free size: %zu.\n", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM), heap_caps_get_free_size(MALLOC_CAP_SPIRAM), heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));

        //conv1_act_unpacked = http_get_task(WEB_SERVER_, "conv_act_1.npy", WEB_PORT_, "/int", "int");
        conv1_act_unpacked = heap_caps_malloc(C1XY*C1XY*C1Z*4, MALLOC_CAP_SPIRAM);
        conv1_wgt_unpacked = http_get_task(WEB_SERVER_, "conv_weight_1.npy", WEB_PORT_, "/int", "int");
        conv1_mean = http_get_task(WEB_SERVER_, "mu_1.npy", WEB_PORT_, "/float", "float");
        conv1_var = http_get_task(WEB_SERVER_, "sigma_1.npy", WEB_PORT_, "/float", "float");
        conv1_gamma = http_get_task(WEB_SERVER_, "gamma_1.npy", WEB_PORT_, "/float", "float");
        conv1_beta = http_get_task(WEB_SERVER_, "beta_1.npy", WEB_PORT_, "/float", "float");

        conv2_1_act_unpacked = heap_caps_malloc(C2_1XY*C2_1XY*C2_1Z*4, MALLOC_CAP_SPIRAM); //malloc(4*C2_1XY*C2_1XY*C2_1Z);
        conv2_1_act = heap_caps_malloc(C2_1XY*C2_1XY*C2_1Z/8, MALLOC_CAP_SPIRAM);
        conv2_2_act = heap_caps_malloc(C2_2XY*C2_2XY*C2_2Z/8, MALLOC_CAP_SPIRAM);
        printf("Largest free block: %zu. Free size: %zu. Minimum free size: %zu.\n", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM), heap_caps_get_free_size(MALLOC_CAP_SPIRAM), heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
        printf("Trying to allocate: %d bytes\n", C2_2OXY*C2_2OXY*C2_2KZ*4);
        conv2_3_act_unpacked = heap_caps_malloc(C2_2OXY*C2_2OXY*C2_2KZ*4, MALLOC_CAP_SPIRAM);
        printf("conv2_3_act_unpacked pointer value %p", conv2_3_act_unpacked);

        conv2_3_act = heap_caps_malloc(C2_2OXY*C2_2OXY*C2_2KZ/8, MALLOC_CAP_SPIRAM);
        //conv2_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_2_1.npy", WEB_PORT_, "/pack", "pack");
        conv2_1_wgt = heap_caps_malloc(C2_1KXY*C2_1KXY*C2_1Z*C2_1KZ/8, MALLOC_CAP_SPIRAM);
        conv2_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_2_1.npy", WEB_PORT_, "/float", "float");
        conv2_1_sign = http_get_task(WEB_SERVER_, "conv_sign_2_1.npy", WEB_PORT_, "/pack", "pack");
        //conv2_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_2_1.npy", WEB_PORT_, "/pack", "pack");
        conv2_2_wgt = heap_caps_malloc(C2_2KXY*C2_2KXY*C2_2Z*C2_2KZ/8, MALLOC_CAP_SPIRAM);
        conv2_2_mean = http_get_task(WEB_SERVER_, "mean_2_2.npy", WEB_PORT_, "/float", "float");
        conv2_2_var = http_get_task(WEB_SERVER_, "sigma_2_2.npy", WEB_PORT_, "/float", "float");
        conv2_2_gamma = http_get_task(WEB_SERVER_, "gamma_2_2.npy", WEB_PORT_, "/float", "float");
        conv2_2_beta = http_get_task(WEB_SERVER_, "beta_2_2.npy", WEB_PORT_, "/float", "float");

        conv3_1_act_unpacked = heap_caps_malloc(C3_1XY*C3_1XY*C3_1Z*4, MALLOC_CAP_SPIRAM); //malloc(4*C3_1XY*C3_1XY*C3_1Z);
        conv3_1_act = heap_caps_malloc(C3_1XY*C3_1XY*C3_1Z/8, MALLOC_CAP_SPIRAM);
        conv3_2_act = heap_caps_malloc(C3_2XY*C3_2XY*C3_2Z/8, MALLOC_CAP_SPIRAM);
        conv3_3_act_unpacked = heap_caps_malloc(C3_2XY*C3_2XY*C3_2Z*4, MALLOC_CAP_SPIRAM);
        conv3_3_act = heap_caps_malloc(C3_2XY*C3_2XY*C3_2Z/8, MALLOC_CAP_SPIRAM);
        //conv3_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_3_1.npy", WEB_PORT_, "/pack", "pack");
        conv3_1_wgt = heap_caps_malloc(C3_1KXY*C3_1KXY*C3_1Z*C3_1KZ/8, MALLOC_CAP_SPIRAM);
        conv3_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_3_1.npy", WEB_PORT_, "/float", "float");
        conv3_1_sign = http_get_task(WEB_SERVER_, "conv_sign_3_1.npy", WEB_PORT_, "/pack", "pack");
        //conv3_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_3_1.npy", WEB_PORT_, "/pack", "pack");
        conv3_2_wgt = heap_caps_malloc(C3_2KXY*C3_2KXY*C3_2Z*C3_2KZ/8, MALLOC_CAP_SPIRAM);
        conv3_2_mean = http_get_task(WEB_SERVER_, "mean_3_2.npy", WEB_PORT_, "/float", "float");
        conv3_2_var = http_get_task(WEB_SERVER_, "sigma_3_2.npy", WEB_PORT_, "/float", "float");
        conv3_2_gamma = http_get_task(WEB_SERVER_, "gamma_3_2.npy", WEB_PORT_, "/float", "float");
        conv3_2_beta = http_get_task(WEB_SERVER_, "beta_3_2.npy", WEB_PORT_, "/float", "float");

        conv4_1_act_unpacked = heap_caps_malloc(C4_1XY*C4_1XY*C4_1Z*4, MALLOC_CAP_SPIRAM); //malloc(4*C4_1XY*C4_1XY*C4_1Z);
        conv4_1_act = heap_caps_malloc(C4_1XY*C4_1XY*C4_1Z/8, MALLOC_CAP_SPIRAM);
        conv4_2_act = heap_caps_malloc(C4_2XY*C4_2XY*C4_2Z/8, MALLOC_CAP_SPIRAM);
        conv4_3_act_unpacked = heap_caps_malloc(C4_2XY*C4_2XY*C4_2Z*4, MALLOC_CAP_SPIRAM);
        conv4_3_act = heap_caps_malloc(C3_2XY*C3_2XY*C3_2Z/8, MALLOC_CAP_SPIRAM);
        //conv4_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_4_1.npy", WEB_PORT_, "/pack", "pack");
        conv4_1_wgt = heap_caps_malloc(C4_1KXY*C4_1KXY*C4_1Z*C4_1KZ/8, MALLOC_CAP_SPIRAM);
        //conv4_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_4_2.npy", WEB_PORT_, "/pack", "pack");
        conv4_2_wgt = heap_caps_malloc(C4_2KXY*C4_2KXY*C4_2Z*C4_2KZ/8, MALLOC_CAP_SPIRAM);
        conv4_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_4_1.npy", WEB_PORT_, "/float", "float");
        conv4_1_sign = http_get_task(WEB_SERVER_, "conv_sign_4_1.npy", WEB_PORT_, "/pack", "pack");
        conv4_2_mean = http_get_task(WEB_SERVER_, "mean_4_2.npy", WEB_PORT_, "/float", "float");
        conv4_2_var = http_get_task(WEB_SERVER_, "sigma_4_2.npy", WEB_PORT_, "/float", "float");
        conv4_2_gamma = http_get_task(WEB_SERVER_, "gamma_4_2.npy", WEB_PORT_, "/float", "float");
        conv4_2_beta = http_get_task(WEB_SERVER_, "beta_4_2.npy", WEB_PORT_, "/float", "float");
        //TODO : Fix following
        conv4_d_act_unpacked = heap_caps_malloc(4*C4_dXY*C4_dXY*C4_dZ, MALLOC_CAP_SPIRAM);
        //conv4_d_wgt = http_get_task(WEB_SERVER_, "conv_weight_4_d.npy", WEB_PORT_, "/pack", "pack");
        conv4_d_wgt = heap_caps_malloc(C4_dKXY*C4_dKXY*C4_dZ*C4_dKZ/8, MALLOC_CAP_SPIRAM);
        conv4_d_mean = http_get_task(WEB_SERVER_, "mean_4_d.npy", WEB_PORT_, "/float", "float");
        conv4_d_var = http_get_task(WEB_SERVER_, "sigma_4_d.npy", WEB_PORT_, "/float", "float");
        conv4_d_gamma = http_get_task(WEB_SERVER_, "gamma_4_d.npy", WEB_PORT_, "/float", "float");
        conv4_d_beta = http_get_task(WEB_SERVER_, "beta_4_d.npy", WEB_PORT_, "/float", "float");

        conv5_1_act_unpacked = heap_caps_malloc(4*C5_1XY*C5_1XY*C5_1Z, MALLOC_CAP_SPIRAM); //malloc(4*C5_1XY*C5_1XY*C5_1Z);
        conv5_1_act = heap_caps_malloc(C5_1XY*C5_1XY*C5_1Z/32, MALLOC_CAP_SPIRAM);
        conv5_2_act = heap_caps_malloc(C5_2XY*C5_2XY*C5_2Z/32, MALLOC_CAP_SPIRAM);
        conv5_3_act_unpacked = heap_caps_malloc(4*C5_2XY*C5_2XY*C5_2Z, MALLOC_CAP_SPIRAM);
        conv5_3_act = heap_caps_malloc(C3_2XY*C3_2XY*C3_2Z/32, MALLOC_CAP_SPIRAM);
        //conv5_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_5_1.npy", WEB_PORT_, "/pack", "pack");
        conv5_1_wgt = heap_caps_malloc(C5_1KXY*C5_1KXY*C5_1Z*C5_1KZ/8, MALLOC_CAP_SPIRAM);
        conv5_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_5_1.npy", WEB_PORT_, "/float", "float");
        conv5_1_sign = http_get_task(WEB_SERVER_, "conv_sign_5_1.npy", WEB_PORT_, "/pack", "pack");
        //conv5_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_5_1.npy", WEB_PORT_, "/pack", "pack");
        conv5_2_wgt = heap_caps_malloc(C5_2KXY*C5_2KXY*C5_2Z*C5_2KZ/8, MALLOC_CAP_SPIRAM);
        conv5_2_mean = http_get_task(WEB_SERVER_, "mean_5_2.npy", WEB_PORT_, "/float", "float");
        conv5_2_var = http_get_task(WEB_SERVER_, "sigma_5_2.npy", WEB_PORT_, "/float", "float");
        conv5_2_gamma = http_get_task(WEB_SERVER_, "gamma_5_2.npy", WEB_PORT_, "/float", "float");
        conv5_2_beta = http_get_task(WEB_SERVER_, "beta_5_2.npy", WEB_PORT_, "/float", "float");

        fc_in = heap_caps_malloc(4*F1I, MALLOC_CAP_SPIRAM);
        fc_wgt = heap_caps_malloc(F1I*F1O/pckWdt, MALLOC_CAP_SPIRAM);
        fc_out = heap_caps_malloc(4*F1O, MALLOC_CAP_SPIRAM);


        int64_t start = esp_timer_get_time();
        forward();
        int64_t finish = esp_timer_get_time();

        ESP_LOGI(TAG, "forward duration (microseconds): %f", ((float)(finish-start)));

        free(conv1_wgt_unpacked);
        free(conv2_1_act_unpacked);
        free(conv1_mean);
        free(conv1_var);
        free(conv1_gamma);
        free(conv1_beta);

        free(conv2_1_act_unpacked);
        free(conv2_1_act);
        free(conv2_2_act);
        free(conv2_3_act_unpacked);
        free(conv2_1_wgt);
        free(conv2_1_thresh);
        free(conv2_1_sign);
        free(conv2_2_wgt);
        free(conv2_2_mean);
        free(conv2_2_var);
        free(conv2_2_gamma);
        free(conv2_2_beta);

        free(conv3_1_act_unpacked);
        free(conv3_1_act);
        free(conv3_2_act);
        free(conv3_3_act_unpacked);
        free(conv3_1_wgt);
        free(conv3_1_thresh);
        free(conv3_1_sign);
        free(conv3_2_wgt);
        free(conv3_2_mean);
        free(conv3_2_var);
        free(conv3_2_gamma);
        free(conv3_2_beta);

        free(conv4_1_act_unpacked);
        free(conv4_1_act);
        free(conv4_2_act);
        free(conv4_3_act_unpacked);
        free(conv4_1_wgt);
        free(conv4_1_thresh);
        free(conv4_1_sign);
        free(conv4_2_wgt);
        free(conv4_2_mean);
        free(conv4_2_var);
        free(conv4_2_gamma);
        free(conv4_2_beta);
        free(conv4_d_act_unpacked);
        free(conv4_d_wgt);
        free(conv4_d_mean);
        free(conv4_d_var);
        free(conv4_d_gamma);
        free(conv4_d_beta);


        free(conv5_1_act_unpacked);
        free(conv5_1_act);
        free(conv5_2_act);
        free(conv5_3_act_unpacked);
        free(conv5_1_wgt);
        free(conv5_1_thresh);
        free(conv5_1_sign);
        free(conv5_2_wgt);
        free(conv5_2_mean);
        free(conv5_2_var);
        free(conv5_2_gamma);
        free(conv5_2_beta);
    }
}

void app_main()
{   
    ESP_ERROR_CHECK( nvs_flash_init() );
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
     * Read "Establishing Wi-Fi or Ethernet Connection" section in
     * examples/protocols/README.md for more information about this function.
     */
    ESP_ERROR_CHECK(example_connect());

    //https://stackoverflow.com/questions/66278271/task-watchdog-got-triggered-the-tasks-did-not-reset-the-watchdog-in-time
    //Using tskIDLE_PRIORITY to bypass "Task watchdog got triggered." exception
    xTaskCreatePinnedToCore(get_weights, "get_weights", 4096, NULL, tskIDLE_PRIORITY, NULL, 0);
}
