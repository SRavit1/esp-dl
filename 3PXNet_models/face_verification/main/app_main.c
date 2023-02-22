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

static void task(void *pvParameters) {
    while(1) {        
        //copy conv1_act_unpacked to buffer1
        //printf("Memcpy arguments %p %p %d", buffer1, conv1_act_unpacked, C1XY*C1XY*C1Z);
        //fflush(stdout);
        //memcpy(buffer1, conv1_act_unpacked, C1XY*C1XY*C1Z);

        ESP_LOGI(TAG, "Beginning forward pass");
        int64_t start = esp_timer_get_time();
        
        forward();
        
        int64_t finish = esp_timer_get_time();
        ESP_LOGI(TAG, "Finishing forward pass");

        ESP_LOGI(TAG, "forward duration (microseconds): %f", ((float)(finish-start)));
    }
}

void app_main()
{
    ESP_ERROR_CHECK(nvs_flash_init() );
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
     * Read "Establishing Wi-Fi or Ethernet Connection" section in
     * examples/protocols/README.md for more information about this function.
     */
    ESP_ERROR_CHECK(example_connect());

    initialize_weights_buffers();

    //https://stackoverflow.com/questions/66278271/task-watchdog-got-triggered-the-tasks-did-not-reset-the-watchdog-in-time
    //Using tskIDLE_PRIORITY to bypass "Task watchdog got triggered." exception
    xTaskCreatePinnedToCore(task, "task", 4096, NULL, tskIDLE_PRIORITY, NULL, 0);

    cleanup_buffers();
}
