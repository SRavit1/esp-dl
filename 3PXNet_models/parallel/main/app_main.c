#include <stdio.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

int face_flag;
xSemaphoreHandle face_semaphore;
int speaker_flag;
xSemaphoreHandle speaker_semaphore;

//face
static void task0(void *pvParameters) {
    while (1) {
        xSemaphoreTake(face_semaphore, 100 / portTICK_PERIOD_MS);
        if (face_flag == 0) {
            printf("face forward pass\n");
            vTaskDelay(2000 / portTICK_PERIOD_MS);

            face_flag = 1;
        }
        xSemaphoreGive(face_semaphore);
    }
}

//speaker
static void task1(void *pvParameters) {
    while (1) {
        xSemaphoreTake(speaker_semaphore, 100 / portTICK_PERIOD_MS);
        if (speaker_flag == 0) {
            printf("speaker forward pass\n");
            vTaskDelay(2000 / portTICK_PERIOD_MS);

            speaker_flag = 1;
        }
        xSemaphoreGive(speaker_semaphore);
    }
}

static void task2(void *pvParameters) {
    while (1) {
        xSemaphoreTake(face_semaphore, 100 / portTICK_PERIOD_MS);
        xSemaphoreTake(speaker_semaphore, 100 / portTICK_PERIOD_MS);

        if (face_flag == 1 && speaker_flag == 1) {
            printf("fusion forward pass\n");
            vTaskDelay(2000 / portTICK_PERIOD_MS);

            face_flag = 0;
            speaker_flag = 0;
        }

        xSemaphoreGive(speaker_semaphore);
        xSemaphoreGive(face_semaphore);
    }
}

void app_main()
{
    face_semaphore = xSemaphoreCreateBinary();
    face_flag = 0;
    speaker_semaphore = xSemaphoreCreateBinary();
    speaker_flag = 0;
    
    xTaskCreatePinnedToCore(task0, "task0", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(task1, "task1", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(task2, "task2", 4096, NULL, 5, NULL, 0);
}
