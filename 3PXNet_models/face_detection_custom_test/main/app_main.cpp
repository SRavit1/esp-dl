#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "dl_lib_matrix3d.h"
#include "mtmn.h"

void printMatrix(dl_matrix3d_t* mat) {
    if (!mat) return;
    if (mat->item) {
        for (int i = 0; i < mat->h * mat->w * mat->c; i++) {
            printf("%f ", mat->item[i]);
            if (i%8 == 0) printf("\n");
        }
    }
}

void printOutput(mtmn_net_t* out) {
    if (!out) return;
    printf("Printing network output START===");
    printMatrix(out->category);
    printMatrix(out->offset);
    printMatrix(out->landmark);
    printf("Printing network output END===");
}

void test(void *arg)
{
    dl_matrix3du_t *input = dl_matrix3du_alloc(1, 12, 12, 3);
    for (int i = 0; i < input->h * input->w; i++)
    {
        input->item[i] = 1;
    }

    while(1)
    {
        mtmn_net_t* result = pnet_lite_f(input);
        printOutput(result);


        vTaskDelay(100);
    }
}

extern "C" void app_main()
{
    xTaskCreatePinnedToCore(&test, "test", 4096, NULL, 5, NULL, 0);
}
