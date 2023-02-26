#include <stdio.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "data.h"

const uint8_t arr[] = ARR;

static void task(void *pvParameters) {
    while(1) {
        for (int i = 0; i < N; i++) {
            printf("%d", arr[i]);
        }
        printf("\n");
    }
}

void app_main()
{
    xTaskCreatePinnedToCore(task, "task", 4096, NULL, tskIDLE_PRIORITY, NULL, 0);
}
