#include <stdio.h>
#include <stdlib.h>

#include "dl_lib_matrix3d.h"
#include "dl_lib_matrix3dq.h"

void fp_toy() {
    dl_matrix3d_t *input = dl_matrix3d_alloc(1, 8, 8, 3);

    dl_matrix3d_t *kernel = dl_matrix3d_alloc(32, 2, 2, 3);
    dl_matrix3d_t *bias = dl_matrix3d_alloc(1, 1, 1, 32); 

    dl_matrix3d_t *output = dl_matrix3dff_conv_common(input, kernel, bias, 1, 1, PADDING_VALID);

    dl_matrix3d_free(input);
    dl_matrix3d_free(kernel);
    dl_matrix3d_free(bias);
    dl_matrix3d_free(output);
}

void q_toy() {
    int exponent = -5;

    dl_matrix3dq_t *input_q = dl_matrix3dq_alloc(1, 8, 8, 3, exponent);

    dl_matrix3dq_t *kernel_q = dl_matrix3dq_alloc(32, 2, 2, 3, exponent);
    dl_matrix3dq_t *bias_q = dl_matrix3dq_alloc(1, 1, 1, 32, exponent); 

    dl_matrix3dq_t *output_q = dl_matrix3dqq_conv_common(input_q, kernel_q, bias_q, 1, 1, PADDING_VALID, -5, DL_XTENSA_IMPL);

    dl_matrix3dq_free(input_q);
    dl_matrix3dq_free(kernel_q);
    dl_matrix3dq_free(bias_q);
    dl_matrix3dq_free(output_q);
}

void q_soft_toy() {
    int exponent = -5;

    dl_matrix3dq_t *input_q_soft = dl_matrix3dq_alloc(1, 8, 8, 3, exponent);

    dl_matrix3dq_t *kernel_q_soft = dl_matrix3dq_alloc(32, 2, 2, 3, exponent);
    dl_matrix3dq_t *bias_q_soft = dl_matrix3dq_alloc(1, 1, 1, 32, exponent); 

    dl_matrix3dq_t *output_q_soft = dl_matrix3dqq_conv_common(input_q_soft, kernel_q_soft, bias_q_soft, 1, 1, PADDING_VALID, -5, DL_C_IMPL);

    dl_matrix3dq_free(input_q_soft);
    dl_matrix3dq_free(kernel_q_soft);
    dl_matrix3dq_free(bias_q_soft);
    dl_matrix3dq_free(output_q_soft);
}

void app_main()
{
    while(1)
    {
        fp_toy();
        q_toy();
        q_soft_toy();

        printf("\nfinished one iteration");
    }
}
