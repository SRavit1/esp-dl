#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_system.h"
#include "esp_log.h"

#include "resnet10_xnor.c"

static const char *TAG = "face_verification";

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
    //https://stackoverflow.com/questions/66278271/task-watchdog-got-triggered-the-tasks-did-not-reset-the-watchdog-in-time
    //Using tskIDLE_PRIORITY to bypass "Task watchdog got triggered." exception
    xTaskCreatePinnedToCore(task, "task", 4096, NULL, tskIDLE_PRIORITY, NULL, 0);
}