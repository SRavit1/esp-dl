#include "printUtils.h"

void printMatrix(dl_matrix3d_t* mat) {
    if (!mat) return;
    printf("NWHC shape is (%d, %d, %d, %d)\n", mat->n, mat->w, mat->h, mat->c);
    printf("Value is [ ");
    if (mat->item) {
        //mat->h * mat->w * mat->c
        for (int i = 0; i < 16; i++) {
            printf("%f ,", mat->item[i]);
            if (i%8 == 7) printf("\n");
        }
    }
    printf("]\n");
}

void printMatrixU(dl_matrix3du_t* mat) {
    if (!mat) return;
    printf("NWHC shape is (%d, %d, %d, %d)\n", mat->n, mat->w, mat->h, mat->c);
    printf("Value is [ ");
    int nums_per_row = 24;
    if (mat->item) {
        for (int i = 0; i < mat->h * mat->w * mat->c; i++) {
            printf("%d ,", mat->item[i]);
            if (i%nums_per_row == nums_per_row-1) printf("\n");
        }
    }
    printf("]\n");
}

void printMatrixQu(dl_matrix3dq_t* mat) {
    dl_matrix3d_t* mat_f = dl_matrix3d_from_matrixq(mat);
    printMatrix(mat_f);
    dl_matrix3d_free(mat_f);
}