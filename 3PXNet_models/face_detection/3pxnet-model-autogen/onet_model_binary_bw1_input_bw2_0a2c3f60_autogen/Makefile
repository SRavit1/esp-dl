CC=gcc
INFERENCE_LIB_DIR=../../3pxnet-inference-library
#INFERENCE_LIB_DIR=../../3pxnet-inference-library-orig

all:
	$(CC) \
	-I$(INFERENCE_LIB_DIR)/util \
	-I$(INFERENCE_LIB_DIR)/xnor \
	-I$(INFERENCE_LIB_DIR)/3pxnet \
	-I$(INFERENCE_LIB_DIR)/bwn \
	-I.. \
	-DPCK32 \
	-DARCH64 \
	-DGCC \
	-DPCNTSW \
	-Wno-overflow \
	-Wno-pointer-sign \
	-lm \
	-g \
	-Og \
	-o source \
	$(INFERENCE_LIB_DIR)/xnor/xnor_cn.c \
	$(INFERENCE_LIB_DIR)/xnor/xnor_fc.c \
	$(INFERENCE_LIB_DIR)/bwn/bwn_dense_cn.c \
	$(INFERENCE_LIB_DIR)/util/xnor_base.c \
	../3pxnet_util.c \
	source.c
