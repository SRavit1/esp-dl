#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "input.h"
#include "model.h"

void test(void *arg)
{
    dl_matrix3d_t *image = dl_matrix3d_alloc(1, 28, 28, 1);
    for (int i = 0; i < image->h * image->w; i++)
    {
        image->item[i] = input_item_array[i] / 255.0f;
    }

    dl_matrix3d_t* o4_2 = mnist_model(image);

    while(1)
    {
        int idx = 0;
        fptp_t max = o4_2->item[0];
        printf("Result:\n");
        for (int i = 0; i < o4_2->c; i++)
        {
            printf("%f\t", o4_2->item[i]);
            if (max < o4_2->item[i])
            {
                max = o4_2->item[i];
                idx = i;
            }
        }
        printf("\nThe number is: %d.\n", idx);


        vTaskDelay(100);
    }
}

void app_main()
{
    xTaskCreatePinnedToCore(&test, "test", 4096, NULL, 5, NULL, 0);
}
