#include "model.h"
#include "dl_lib_matrix3d.h"
#include "fd_forward.h"
#include "esp_log.h"

#include "printUtils.h"

static const char *TAG = "app_process";

dl_matrix3d_t* model_forward(dl_matrix3du_t *input) {
	/*
      x = Conv2d(3, num_filters, kernel_size=3, stride=1, padding='same')(x)
      x = BatchNorm2d(num_filters)(x)
      x = F.relu(x)
      
      y = Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding='same')(x)
      y = BatchNorm2d(num_filters)(y)
      y = F.relu(y)
      y = Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding='same')(y)
      y = BatchNorm2d(num_filters)(y)

      x = x + y
      x = F.relu(x)

      y = F.pad(x, (0, 1, 0, 1))
      y = Conv2d(num_filters, num_filters_2, kernel_size=3, stride=2, padding='valid')(y)
      y = BatchNorm2d(num_filters_2)(y)
      y = F.relu(y)
      y = Conv2d(num_filters_2, num_filters_2, kernel_size=3, stride=1, padding='same')(y)
      y = BatchNorm2d(num_filters_2)(y)
      
      x = Conv2d(num_filters, num_filters_2, kernel_size=1, stride=2, padding='valid')(x)
      x = x + y
      x = F.relu(x)

      y = F.pad(x, (0, 1, 0, 1))
      y = Conv2d(num_filters_2, num_filters_3, kernel_size=3, stride=2, padding='valid')(y)
      y = BatchNorm2d(num_filters_3)(y)
      y = F.relu(y)
      
      y = Conv2d(num_filters_3, num_filters_3, kernel_size=3, stride=1, padding='same')(y)
      y = BatchNorm2d(num_filters_3)(y)

      x = Conv2d(num_filters_2, num_filters_3, kernel_size=1, stride=2, padding='valid')(x)
      x = x + y
      x = F.relu(x)

      x = AvgPool2d((8, 8))(x)
      y = torch.flatten(x, start_dim=1)
      x = Linear(64, 10)(y)
    */
	dl_matrix3d_t *x, *y;
	dl_matrix3d_t *x_temp, *y_temp;

	x = dl_matrix3duf_conv_common(input, &conv1_filter, &conv1_bias, 1, 1, PADDING_SAME);
	//dl_matrix3d_batch_normalize(x, &batchnorm1_scale, &batchnorm1_offset);
	dl_matrix3d_relu(x);

	y = dl_matrix3dff_conv_common(x, &conv2_filter, &conv2_bias, 1, 1, PADDING_SAME);
	//dl_matrix3d_batch_normalize(y, &batchnorm2_scale, &batchnorm2_offset);
	dl_matrix3d_relu(y);

	y_temp = dl_matrix3dff_conv_common(y, &conv3_filter, &conv3_bias, 1, 1, PADDING_SAME);
	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm3_scale, &batchnorm3_offset);

	x_temp = dl_matrix3d_add(x, y);
	dl_matrix3d_free(x);
	x = x_temp;
	dl_matrix3d_relu(x);

	y_temp = dl_matrix3dff_conv_common(x, &conv4_filter, &conv4_bias, 2, 2, PADDING_SAME);
	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm4_scale, &batchnorm4_offset);
	dl_matrix3d_relu(y);
	y_temp = dl_matrix3dff_conv_common(y, &conv5_filter, &conv5_bias, 1, 1, PADDING_SAME);
	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm5_scale, &batchnorm5_offset);

	x = dl_matrix3dff_conv_common(x, &conv6_filter, &conv6_bias, 2, 2, PADDING_VALID);
	x_temp = dl_matrix3d_add(x, y);
	dl_matrix3d_free(x);
	x = x_temp;
	dl_matrix3d_relu(x);

	y_temp = dl_matrix3dff_conv_common(x, &conv7_filter, &conv7_bias, 2, 2, PADDING_SAME);
	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm6_scale, &batchnorm6_offset);
	dl_matrix3d_relu(y);

	y_temp = dl_matrix3dff_conv_common(y, &conv8_filter, &conv8_bias, 1, 1, PADDING_SAME);
	dl_matrix3d_free(y);
	y = y_temp;
	//dl_matrix3d_batch_normalize(y, &batchnorm7_scale, &batchnorm7_offset);

	x_temp = dl_matrix3dff_conv_common(x, &conv9_filter, &conv9_bias, 2, 2, PADDING_VALID);
	dl_matrix3d_free(x);
	x = x_temp;
	x_temp = dl_matrix3d_add(x, y);
	dl_matrix3d_free(x);
	x = x_temp;
	dl_matrix3d_relu(x);

	x_temp = dl_matrix3d_pooling(x, 8, 8, 8, 8, PADDING_VALID, DL_POOLING_AVG);
	dl_matrix3d_free(x);
	x = x_temp;
	y = x;

	y->c *= y->w * y->h;
	y->w = 1;
	y->h = 1;

	x = dl_matrix3d_alloc(1, 1, 1, 10);
	
	dl_matrix3dff_fc_with_bias(x, y, &fc1_filter, &fc1_bias);
	
	return x;
}