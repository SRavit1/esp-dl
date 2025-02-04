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
//#define WEB_SERVER_ "164.67.211.125"
#define WEB_SERVER_ "192.168.201.196"
#define WEB_PORT_ "5000"
#define WEB_PATH_ "/"

static const char *TAG = "example";

struct float_array {
    void* data;
    int n;
};

struct float_array readHTTPResponse(int s, char* recv_buf, int recv_buf_size, char* mode) {
    const int curr_val_size = 10;
    char curr_val[curr_val_size];
    unsigned int curr_val_counter = 0;

    struct float_array arr;
    unsigned int data_counter = 0;

    int data_size = 4;
    if (strcmp(mode, "float") == 0) data_size = 4;
    else if (strcmp(mode, "int") == 0) data_size = 4;
    else if (strcmp(mode, "pack") == 0) data_size = 4;

    int phase = 0; //0: headers, 1: size, 2: data
    int size = 0;
    
    int r;
    do {
        bzero(recv_buf, recv_buf_size);
        r = read(s, recv_buf, sizeof(recv_buf)-1);
        for(int i = 0; i < r; i++) {
            char c = recv_buf[i];
            if (c == '|' && phase == 0) {
                phase++;
            } else if (c == ' ' && phase == 1) {//size finished; now allocate data
                phase++;
                printf("Allocating float array of len %d\n", size);
                arr.n = size;
                arr.data = malloc(data_size*size);
            } else if (c == ',' && phase == 2) {
                if (strcmp(mode, "float") == 0) *((float*) (arr.data+data_size*data_counter)) = atof(curr_val);
                else if (strcmp(mode, "int") == 0) *((int*) (arr.data+data_size*data_counter)) = atoi(curr_val);
                else if (strcmp(mode, "pack") == 0) *((long*) (arr.data+data_size*data_counter)) = strtol(curr_val, NULL, 16);

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
    } while(r > 0);
    ESP_LOGI(TAG, "... done reading from socket. Last read return=%d errno=%d.", r, errno);

    return arr;
}

static float* http_get_task(char *WEB_SERVER, char *ARGUMENT, char *WEB_PORT, char *WEB_PATH)
{
    float *ret = 0;

    //const char *REQUEST = strcat(strcat("GET ", strcat(WEB_PATH, strcat(" HTTP/1.0\r\nHost: ", strcat(WEB_SERVER, strcat(":", strcat(WEB_PORT, strcat("\r\n", strcat("User-Agent: esp-idf/1.0 esp32\r\n", "\r\n")))))))));
    char REQUEST[1000];
    sprintf(REQUEST, "GET %s?%s HTTP/1.0\r\nHost: %s:%s\r\nUser-Agent: esp-idf/1.0 esp32\r\n\r\n", WEB_PATH, ARGUMENT, WEB_SERVER, WEB_PORT);

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
    ESP_LOGI(TAG, "DNS lookup succeeded. IP=%s", inet_ntoa(*addr));

    s = socket(res->ai_family, res->ai_socktype, 0);
    if(s < 0) {
        ESP_LOGE(TAG, "... Failed to allocate socket.");
        freeaddrinfo(res);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        return 0;
    }
    ESP_LOGI(TAG, "... allocated socket");

    if(connect(s, res->ai_addr, res->ai_addrlen) != 0) {
        ESP_LOGE(TAG, "... socket connect failed errno=%d", errno);
        close(s);
        freeaddrinfo(res);
        vTaskDelay(4000 / portTICK_PERIOD_MS);
        return 0;
    }

    ESP_LOGI(TAG, "... connected");
    freeaddrinfo(res);

    if (write(s, REQUEST, strlen(REQUEST)) < 0) {
        ESP_LOGE(TAG, "... socket send failed");
        close(s);
        vTaskDelay(4000 / portTICK_PERIOD_MS);
        return 0;
    }
    ESP_LOGI(TAG, "... socket send success");

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
    ESP_LOGI(TAG, "... set socket receiving timeout success");

    /* Read HTTP response */
    struct float_array arr = readHTTPResponse(s, recv_buf, sizeof(recv_buf), "float");
    float *data = (float*) arr.data;
    for (int i = 0; i < arr.n; i++) {
        printf("%.2f ", data[i]);
    }
    printf("\n");
    /*
    for (int i = 0; i < 2; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
    */

    close(s);
    for(int countdown = 10; countdown >= 0; countdown--) {
        ESP_LOGI(TAG, "%d... ", countdown);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    ESP_LOGI(TAG, "Starting again!");
    
    return ret;
}

static void get_weights(void *pvParameters) {
    while(1) {
        float *weight = http_get_task(WEB_SERVER_, "conv_weight_1.npy", WEB_PORT_, WEB_PATH_);
    }
}