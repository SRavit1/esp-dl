/* HTTP GET Example using plain POSIX sockets

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "protocol_examples_common.h"

#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include "lwip/netdb.h"
#include "lwip/dns.h"

/* Constants that aren't configurable in menuconfig */
/*
#define WEB_SERVER "example.com"
#define WEB_PORT "80"
#define WEB_PATH "/"
*/

//#define WEB_SERVER_ "192.168.1.46"
//#define WEB_SERVER_ "164.67.211.10"
#define WEB_SERVER_ "192.168.204.196"
#define WEB_PORT_ "5000"

static const char *TAG = "example";

struct array {
    void* data;
    int n;
};

struct array readHTTPResponse(int s, char* recv_buf, int recv_buf_size, char* mode, void *data_ptr) {
    const int curr_val_size = 10;
    char curr_val[curr_val_size];
    unsigned int curr_val_counter = 0;

    struct array arr;
    unsigned int data_counter = 0;

    int data_size = 4;
    if (strcmp(mode, "float") == 0) data_size = 4;
    else if (strcmp(mode, "int") == 0) data_size = 2;
    else if (strcmp(mode, "pack") == 0) data_size = 4;

    int phase = 0; //0: headers, 1: size, 2: data
    int size = 0;
    
    int counter = 0;

    int r;
    do {
        if (phase == 2) break; //TEMPORARY; Stops data copying to speed up testing

        bzero(recv_buf, recv_buf_size);
        r = read(s, recv_buf, sizeof(recv_buf)-1);
        for(int i = 0; i < r; i++) {
            char c = recv_buf[i];
            if (c == '|' && phase == 0) {
                phase++;
            } else if (c == ' ' && phase == 1) {//size finished; now allocate data
                phase++;
                arr.n = size;
                arr.data = data_ptr; //heap_caps_malloc(data_size*size, MALLOC_CAP_SPIRAM); //malloc(data_size*size);
                printf("Filling %s array of len %d at PSRAM location %p\n", mode, size, arr.data);
            } else if (c == ',' && phase == 2) {
                if (strcmp(mode, "float") == 0) *((float*) (arr.data+data_size*data_counter)) = atof(curr_val);
                else if (strcmp(mode, "int") == 0) {
                    *((int16_t*) (arr.data+data_size*data_counter)) = (int16_t) atoi(curr_val);
                }
                else if (strcmp(mode, "pack") == 0) {
                    //printf("Storing at location %p\n", arr.data+data_size*data_counter);
                    *((long*) (arr.data+data_size*data_counter)) = strtol(curr_val, NULL, 16);
                }

                data_counter++;

                //zero out curr_val for next value
                bzero(curr_val, curr_val_size);
                curr_val_counter = 0;
            } else if (phase == 1) { //add next digit to size
                size = size*10 + (c-'0');
            } else if (phase == 2) { //add next digit to curr_val
                //printf("Encountered character %c\n", c);
                curr_val[curr_val_counter] = c;
                curr_val_counter++;
            }
            //putchar(c);
        }
        //if (counter % 1000 == 0) printf("Received %d segment of data.\n", counter);
        counter ++;

    } while(r > 0);
    //ESP_LOGI(TAG, "... done reading from socket. Last read return=%d errno=%d.", r, errno);

    return arr;
}

static void* http_get_task(char *WEB_SERVER, char *ARGUMENT, char *WEB_PORT, char *WEB_PATH, char* mode, void **weight_buffer)
{
    //const char *REQUEST = strcat(strcat("GET ", strcat(WEB_PATH, strcat(" HTTP/1.0\r\nHost: ", strcat(WEB_SERVER, strcat(":", strcat(WEB_PORT, strcat("\r\n", strcat("User-Agent: esp-idf/1.0 esp32\r\n", "\r\n")))))))));
    char REQUEST[1000];
    sprintf(REQUEST, "GET %s?%s HTTP/1.0\r\nHost: %s:%s\r\nUser-Agent: esp-idf/1.0 esp32\r\n\r\n", WEB_PATH, ARGUMENT, WEB_SERVER, WEB_PORT);

    struct array arr;

    const struct addrinfo hints = {
        .ai_family = AF_INET,
        .ai_socktype = SOCK_STREAM,
    };
    struct addrinfo *res;
    struct in_addr *addr;
    int s;//, r;
    char recv_buf[64];

    int err = getaddrinfo(WEB_SERVER, WEB_PORT, &hints, &res);

    if(err != 0 || res == NULL) {
        ESP_LOGE(TAG, "DNS lookup failed err=%d res=%p", err, res);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        return 0;
    }

    /* Code to print the resolved IP.

        Note: inet_ntoa is non-reentrant, look at ipaddr_ntoa_r for "real" code */
    addr = &((struct sockaddr_in *)res->ai_addr)->sin_addr;
    //ESP_LOGI(TAG, "DNS lookup succeeded. IP=%s", inet_ntoa(*addr));

    s = socket(res->ai_family, res->ai_socktype, 0);
    if(s < 0) {
        ESP_LOGE(TAG, "... Failed to allocate socket.");
        freeaddrinfo(res);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        return 0;
    }
    //ESP_LOGI(TAG, "... allocated socket");

    if(connect(s, res->ai_addr, res->ai_addrlen) != 0) {
        ESP_LOGE(TAG, "... socket connect failed errno=%d", errno);
        close(s);
        freeaddrinfo(res);
        vTaskDelay(4000 / portTICK_PERIOD_MS);
        return 0;
    }

    //ESP_LOGI(TAG, "... connected");
    freeaddrinfo(res);

    if (write(s, REQUEST, strlen(REQUEST)) < 0) {
        ESP_LOGE(TAG, "... socket send failed");
        close(s);
        vTaskDelay(4000 / portTICK_PERIOD_MS);
        return 0;
    }
    //ESP_LOGI(TAG, "... socket send success");

    struct timeval receiving_timeout;
    receiving_timeout.tv_sec = 5;
    receiving_timeout.tv_usec = 0;
    if (setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &receiving_timeout,
            sizeof(receiving_timeout)) < 0) {
        ESP_LOGE(TAG, "... failed to set socket receiving timeout");
        close(s);
        vTaskDelay(4000 / portTICK_PERIOD_MS);
        return 0;
    }
    //ESP_LOGI(TAG, "... set socket receiving timeout success");

    /* Read HTTP response */
    printf("%s: ", ARGUMENT);
    arr = readHTTPResponse(s, recv_buf, sizeof(recv_buf), mode, *weight_buffer);
    *weight_buffer += arr.n;

    /*
    //For debugging
    if (strcmp(mode, "float") == 0) {
        float *data = (float*) arr.data;
        for (int i = 0; i < arr.n; i++) {
            printf("%.2f ", data[i]);
        }
    }
    else if (strcmp(mode, "int") == 0) {
        uint16_t *data = (uint16_t*) arr.data;
        for (int i = 0; i < arr.n; i++) {
            printf("%u ", (unsigned int) data[i]);
        }
    }
    else if (strcmp(mode, "pack") == 0) {
        int *data = (int*) arr.data;
        for (int i = 0; i < arr.n; i++) {
            printf("%x ", data[i]);
        }
    }
    printf("\n");
    */

    close(s);
    /*
    for(int countdown = 10; countdown >= 0; countdown--) {
        //ESP_LOGI(TAG, "%d... ", countdown);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    //ESP_LOGI(TAG, "Starting again!");
    */
    
    return arr.data;
}

void initialize_weights_buffers() {
    buffer1 = heap_caps_malloc(1000000, MALLOC_CAP_SPIRAM);
    buffer2 = heap_caps_malloc(1000000, MALLOC_CAP_SPIRAM);
    buffer3 = heap_caps_malloc(1000000, MALLOC_CAP_SPIRAM);
    weight_buffer = heap_caps_malloc(800000, MALLOC_CAP_SPIRAM);

    printf("b1 %p b2 %p b3 %p weight %p.\n", buffer1, buffer2, buffer3, weight_buffer);
    printf("Largest free block: %zu. Free size: %zu. Minimum free size: %zu.\n", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM), heap_caps_get_free_size(MALLOC_CAP_SPIRAM), heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));

    void *weight_buffer_copy = weight_buffer;

    conv4_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_4_2.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv5_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_5_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv5_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_5_2.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv1_wgt_unpacked = http_get_task(WEB_SERVER_, "conv_weight_1.npy", WEB_PORT_, "/int", "int", &weight_buffer_copy);
    conv2_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_2_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv2_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_2_2.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv3_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_3_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv3_2_wgt = http_get_task(WEB_SERVER_, "conv_weight_3_2.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv4_1_wgt = http_get_task(WEB_SERVER_, "conv_weight_4_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv4_d_wgt = http_get_task(WEB_SERVER_, "conv_weight_4_d.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    fc_wgt = heap_caps_malloc(F1I*F1O/pckWdt, MALLOC_CAP_SPIRAM);

    conv1_mean = http_get_task(WEB_SERVER_, "mu_1.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv1_var = http_get_task(WEB_SERVER_, "sigma_1.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv1_gamma = http_get_task(WEB_SERVER_, "gamma_1.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv1_beta = http_get_task(WEB_SERVER_, "beta_1.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv2_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_2_1.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv2_1_sign = http_get_task(WEB_SERVER_, "conv_sign_2_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv2_2_mean = http_get_task(WEB_SERVER_, "mu_2_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv2_2_var = http_get_task(WEB_SERVER_, "sigma_2_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv2_2_gamma = http_get_task(WEB_SERVER_, "gamma_2_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv2_2_beta = http_get_task(WEB_SERVER_, "beta_2_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv3_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_3_1.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv3_1_sign = http_get_task(WEB_SERVER_, "conv_sign_3_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv3_2_mean = http_get_task(WEB_SERVER_, "mu_3_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv3_2_var = http_get_task(WEB_SERVER_, "sigma_3_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv3_2_gamma = http_get_task(WEB_SERVER_, "gamma_3_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv3_2_beta = http_get_task(WEB_SERVER_, "beta_3_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv4_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_4_1.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv4_1_sign = http_get_task(WEB_SERVER_, "conv_sign_4_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv4_2_mean = http_get_task(WEB_SERVER_, "mu_4_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv4_2_var = http_get_task(WEB_SERVER_, "sigma_4_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv4_2_gamma = http_get_task(WEB_SERVER_, "gamma_4_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv4_2_beta = http_get_task(WEB_SERVER_, "beta_4_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv4_d_mean = http_get_task(WEB_SERVER_, "mu_4_d.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv4_d_var = http_get_task(WEB_SERVER_, "sigma_4_d.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv4_d_gamma = http_get_task(WEB_SERVER_, "gamma_4_d.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv4_d_beta = http_get_task(WEB_SERVER_, "beta_4_d.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv5_1_thresh = http_get_task(WEB_SERVER_, "conv_thr_5_1.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv5_1_sign = http_get_task(WEB_SERVER_, "conv_sign_5_1.npy", WEB_PORT_, "/pack", "pack", &weight_buffer_copy);
    conv5_2_mean = http_get_task(WEB_SERVER_, "mu_5_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv5_2_var = http_get_task(WEB_SERVER_, "sigma_5_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv5_2_gamma = http_get_task(WEB_SERVER_, "gamma_5_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
    conv5_2_beta = http_get_task(WEB_SERVER_, "beta_5_2.npy", WEB_PORT_, "/float", "float", &weight_buffer_copy);
}

void cleanup_buffers() {
    free(buffer1);
    free(buffer2);
    free(buffer3);
    free(weight_buffer);
}