#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

//#include "input.h"
#include "model.h"

void test(void *arg)
{
    while(1)
    {
        dl_matrix3d_t *image = dl_matrix3d_alloc(1, 32, 32, 3);
        for (int i = 0; i < image->h * image->w; i++)
            image->item[i] = 0;
        dl_matrix3d_t* result = model(image);

        vTaskDelay(100);
    }
}

void app_main()
{
    xTaskCreatePinnedToCore(&test, "test", 4096, NULL, 5, NULL, 0);
}
