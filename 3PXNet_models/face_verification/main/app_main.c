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

static void task(void *pvParameters) {
    while(1) {
        buffer1 = heap_caps_malloc(1000000, MALLOC_CAP_SPIRAM);
        buffer2 = heap_caps_malloc(1000000, MALLOC_CAP_SPIRAM);
        buffer3 = heap_caps_malloc(1000000, MALLOC_CAP_SPIRAM);
        weight_buffer = heap_caps_malloc(800000, MALLOC_CAP_SPIRAM);
        void *weight_buffer_copy = weight_buffer;

        printf("b1 %p b2 %p b3 %p weight %p.\n", buffer1, buffer2, buffer3, weight_buffer);
        
        printf("Largest free block: %zu. Free size: %zu. Minimum free size: %zu.\n", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM), heap_caps_get_free_size(MALLOC_CAP_SPIRAM), heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));

        conv4_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_4_2.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv5_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_5_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv5_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_5_2.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv1_wgt_unpacked = http_get_task(WEB_SERVER_, "conv_weight_1.npy", WEB_PORT_, "/int", "int", &weight_buffer);
        conv2_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_2_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv2_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_2_2.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv3_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_3_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv3_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_3_2.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv4_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_4_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv4_d_wgt = http_get_task(WEB_SERVER_, "conv_weight_4_d.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        fc_wgt = heap_caps_malloc(F1I*F1O/pckWdt, MALLOC_CAP_SPIRAM);

        conv1_mean = http_get_task(WEB_SERVER_, "mu_1.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv1_var = http_get_task(WEB_SERVER_, "sigma_1.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv1_gamma = http_get_task(WEB_SERVER_, "gamma_1.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv1_beta = http_get_task(WEB_SERVER_, "beta_1.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv2_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_2_1.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv2_1_sign = http_get_task(WEB_SERVER_, "conv_sign_2_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv2_2_mean = http_get_task(WEB_SERVER_, "mu_2_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv2_2_var = http_get_task(WEB_SERVER_, "sigma_2_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv2_2_gamma = http_get_task(WEB_SERVER_, "gamma_2_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv2_2_beta = http_get_task(WEB_SERVER_, "beta_2_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv3_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_3_1.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv3_1_sign = http_get_task(WEB_SERVER_, "conv_sign_3_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv3_2_mean = http_get_task(WEB_SERVER_, "mu_3_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv3_2_var = http_get_task(WEB_SERVER_, "sigma_3_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv3_2_gamma = http_get_task(WEB_SERVER_, "gamma_3_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv3_2_beta = http_get_task(WEB_SERVER_, "beta_3_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv4_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_4_1.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv4_1_sign = http_get_task(WEB_SERVER_, "conv_sign_4_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv4_2_mean = http_get_task(WEB_SERVER_, "mu_4_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv4_2_var = http_get_task(WEB_SERVER_, "sigma_4_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv4_2_gamma = http_get_task(WEB_SERVER_, "gamma_4_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv4_2_beta = http_get_task(WEB_SERVER_, "beta_4_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv4_d_mean = http_get_task(WEB_SERVER_, "mu_4_d.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv4_d_var = http_get_task(WEB_SERVER_, "sigma_4_d.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv4_d_gamma = http_get_task(WEB_SERVER_, "gamma_4_d.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv4_d_beta = http_get_task(WEB_SERVER_, "beta_4_d.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv5_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_5_1.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv5_1_sign = http_get_task(WEB_SERVER_, "conv_sign_5_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer);
        conv5_2_mean = http_get_task(WEB_SERVER_, "mu_5_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv5_2_var = http_get_task(WEB_SERVER_, "sigma_5_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv5_2_gamma = http_get_task(WEB_SERVER_, "gamma_5_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);
        conv5_2_beta = http_get_task(WEB_SERVER_, "beta_5_2.npy", WEB_PORT_, "/float", "float", &weight_buffer);

        
        ESP_LOGI(TAG, "Beginning forward pass");
        int64_t start = esp_timer_get_time();
        forward();
        int64_t finish = esp_timer_get_time();
        ESP_LOGI(TAG, "Finishing forward pass");

        ESP_LOGI(TAG, "forward duration (microseconds): %f", ((float)(finish-start)));

        free(buffer1);
        free(buffer2);
        free(buffer3);
        free(weight_buffer_copy);
    }
}

void app_main()
{
    const void *flash_mmap_ptr;

    ESP_ERROR_CHECK(nvs_flash_init() );
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
     * Read "Establishing Wi-Fi or Ethernet Connection" section in
     * examples/protocols/README.md for more information about this function.
     */
    ESP_ERROR_CHECK(example_connect());

    //https://stackoverflow.com/questions/66278271/task-watchdog-got-triggered-the-tasks-did-not-reset-the-watchdog-in-time
    //Using tskIDLE_PRIORITY to bypass "Task watchdog got triggered." exception
    xTaskCreatePinnedToCore(task, "task", 4096, NULL, tskIDLE_PRIORITY, NULL, 0);
}
