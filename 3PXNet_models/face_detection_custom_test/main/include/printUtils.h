#pragma once

#include <stdio.h>

#include "dl_lib_matrix3d.h"
#include "mtmn.h"

void printMatrix(dl_matrix3d_t* mat) {
    if (!mat) return;
    printf("Shape is %d x %d x %d\n", mat->h, mat->w, mat->c);
    printf("Value is [ ");
    if (mat->item) {
        for (int i = 0; i < mat->h * mat->w * mat->c; i++) {
            printf("%f ", mat->item[i]);
            if (i%8 == 7) printf("\n");
        }
    }
    printf("]\n");
}

void printOutput(mtmn_net_t* out) {
    if (!out) return;
    printf("Printing network output START===\n");
    printMatrix(out->category);
    printMatrix(out->offset);
    printMatrix(out->landmark);
    printf("Printing network output END===\n");
}