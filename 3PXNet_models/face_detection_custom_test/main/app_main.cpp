#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "dl_lib_matrix3d.h"
#include "mtmn.h"

#include "printUtils.h"

void test(void *arg)
{
    dl_matrix3du_t *p_net_input = dl_matrix3du_alloc(1, 12, 12, 3);
    for (int i = 0; i < p_net_input->h * p_net_input->w * p_net_input->c; i++)
    {
        p_net_input->item[i] = 1;
    }
    dl_matrix3du_t *r_net_input = dl_matrix3du_alloc(1, 24, 24, 3);
    for (int i = 0; i < r_net_input->h * r_net_input->w * r_net_input->c; i++)
    {
        r_net_input->item[i] = 1;
    }
    dl_matrix3du_t *o_net_input = dl_matrix3du_alloc(1, 48, 48, 3);
    for (int i = 0; i < o_net_input->h * o_net_input->w * o_net_input->c; i++)
    {
        o_net_input->item[i] = 1;
    }

    while(1)
    {
        mtmn_net_t* p_net_result = pnet_lite_f(p_net_input);
        printOutput(p_net_result);


        mtmn_net_t* r_net_result = rnet_lite_f_with_score_verify(r_net_input, 0);
        printOutput(r_net_result);



        mtmn_net_t* o_net_result = onet_lite_f_with_score_verify(o_net_input, 0);
        printOutput(o_net_result);


        vTaskDelay(100);
    }
}

extern "C" void app_main()
{
    xTaskCreatePinnedToCore(&test, "test", 4096, NULL, 5, NULL, 0);
}
