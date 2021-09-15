#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "ternarized_high/source.h"

void test(void *arg)
{
    while(1)
    {
        main();

        vTaskDelay(100);
    }
}

void app_main()
{
    xTaskCreatePinnedToCore(&test, "test", 4096, NULL, 5, NULL, 0);
}
