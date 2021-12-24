#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include "datatypes.h"
#include "utils.h"
#include "xnor_base.h"
#include "xnor_fc.h"
#include "3pxnet_fc.h"
#include "3pxnet_cn.h"
#include "xnor_fc.h"
#include "bwn_dense_cn.h"
#include "85.h" 
#include "88.h" 
#include "91.h" 
#include "94.h" 
#include "96.h" 
#include "97.h" 
#include "98.h" 
#include "99.h" 
#include "bn8_running_mean.h" 
#include "bn8_running_var.h" 
#include "bn8_bias.h" 
#include "bn8_weight.h" 
#include "bn1.h" 
#include "bn2.h" 
#include "bn3.h" 
#include "bn4.h" 
#include "image.h"
static int8_t l1_act[] = IMAGES ; 
static uint8_t   labels[] = LABELS; 
#define C1KXY 3
#define C1XY   48
#define C1Z   3
#define C1KZ 32
#define C1PD 0
#define C1PL 3
//static int8_t l1wght[] = _85 ;
static pckDtype l1wght[] = _85 ;
#define C2KXY 3
#define C2XY ((2*C1PD+C1XY-C1KXY+1)/C1PL) 
#define C2Z 32
#define C2KZ 32
static pckDtype l2act_bin[C2XY*C2XY*C2Z/pckWdt]; 
#define C2PD 0
#define C2PL 3
static pckDtype l2wght[] = _88 ;
#define C3KXY 3
#define C3XY ((2*C2PD+C2XY-C2KXY+1)/C2PL) 
#define C3Z 32
#define C3KZ 32
static pckDtype l3act_bin[C3XY*C3XY*C3Z/pckWdt]; 
#define C3PD 0
#define C3PL 1 
static pckDtype l3wght[] = _91 ;
#define C4KXY 2
#define C4XY ((2*C3PD+C3XY-C3KXY+1)/C3PL) 
#define C4Z 32
#define C4KZ 64
static pckDtype l4act_bin[C4XY*C4XY*C4Z/pckWdt]; 
#define C4PD 0
#define C4PL 1 
static pckDtype l4wght[] = _94 ;
#define F5I  64
#define F5NPI  0
#define F5O  128
static pckDtype l5wght[] = _96 ;
static pckDtype l5act_bin[F5I/pckWdt]; 
#define F6I  128
#define F6NPI  0
#define F6O  2
static pckDtype l6wght[] = _97 ;
static pckDtype l6act_bin[F5O/pckWdt]; 
#define F7I  128
#define F7NPI  0
#define F7O  4
static pckDtype l7wght[] = _98 ;
static pckDtype l7act_bin[F6O/pckWdt]; 
#define F8I  128
#define F8NPI  0
#define F8O  10
static pckDtype l8wght[] = _99 ;
static pckDtype l8act_bin[F7O/pckWdt]; 
static float output[10]; 
static pckDtype bn1thr[] = bn1_thresh ; 
static pckDtype bn1sign[] = bn1_sign ; 
static pckDtype bn1offset[] = bn1_offset ; 
static pckDtype bn2thr[] = bn2_thresh ; 
static pckDtype bn2sign[] = bn2_sign ; 
static pckDtype bn2offset[] = bn2_offset ; 
static pckDtype bn3thr[] = bn3_thresh ; 
static pckDtype bn3sign[] = bn3_sign ; 
static pckDtype bn3offset[] = bn3_offset ; 
static pckDtype bn4thr[] = bn4_thresh ; 
static pckDtype bn4sign[] = bn4_sign ; 
static pckDtype bn4offset[] = bn4_offset ; 
static bnDtype bn8mean[] = _bn8_running_mean ; 
static bnDtype bn8var[] = _bn8_running_var ; 
static bnDtype bn8gamma[] = _bn8_weight ; 
static bnDtype bn8beta[] = _bn8_bias ; 

/*
 * @details unrolls depth-width-height pAct into Im2Col vector used by CnBnMulti function
 * Only works when one filter can fit into a single pack (i.e. CKX*CKY*CZ <= pckWdt)
 */
void unRollActivations(pckDtype* __restrict pActRolled, pckDtype* __restrict pAct, const uint16_t dpth,
		const uint16_t wdth, const uint16_t hght, const uint16_t kwdt,
		const uint16_t khgt, const uint16_t pad, const uint16_t pool, const uint8_t in_bit) {
    
	pckDtype* pIn = pAct;
    const uint16_t  yCoeff = wdth * in_bit;
    const uint16_t  xCoeff = in_bit;

	const uint16_t yCoeff_rolled = wdth*in_bit;
	const uint16_t xCoeff_rolled = in_bit;
    for (int y = 0; y < (hght + 2 * pad - khgt + 1)/pool; y++) {
        for (int x = 0; x < (wdth + 2 * pad - kwdt + 1)/pool; x++) {
            for (int yy = 0; yy < pool; yy++) {
                for (int xx = 0; xx < pool; xx++) {
                    uint16_t pIn_offset = (y * pool + yy) * yCoeff + (x * pool + xx)* xCoeff;
                    pIn = pAct + pIn_offset;
                   	
                   	//pIn corresponds to input region of size CKX x CKY x CZ with
                   	//	top-left corner at x=(x*pool+xx), y=(y*pool+yy)
					for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
						*pIn = 0;
						for (uint8_t z = 0; z < dpth; z++) {
							for (uint8_t yy_kernel = 0; yy_kernel < khgt; yy_kernel++) {
								for (uint8_t xx_kernel = 0; xx_kernel < kwdt; xx_kernel++) {
									const uint8_t x_rolled = x*pool + xx;
									const uint8_t y_rolled = y*pool + yy;
									const uint16_t pActRolled_offset = (y_rolled+yy_kernel)*yCoeff_rolled+(x_rolled+xx_kernel)*xCoeff_rolled+bitw;
									const pckDtype bit = (*(pActRolled+pActRolled_offset)&(1<<z))>>z;
									*pIn = (*pIn<<1) | bit;
								}
							}
						}
						pIn++;
					}          
                }
            }           
        }
    }
}


int main(){ 
	/*
	pckDtype pckTest[2] = {-1, 0}; //10 => 2 - 1 = 1
	int intTest[32];
	pckDtype_to_int(pckTest, intTest, 32, 32, 2);
	printf("test values: ");
	print_int_array(intTest, 32);
	*/

	testlayer1();

	/*
	int res;
	CnBnMulti(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset,2,1);
	res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL,1,1);
	res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL,1,1);
	res = FcXnorWrap(l5act_bin, l5wght, F5I, F5O, l6act_bin, bn1thr, bn1sign, bn1offset,1,1);
	int res0 = FcXnorNoBinWrap(l6act_bin, l6wght, F6I, F6O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
	int res1 = FcXnorNoBinWrap(l7act_bin, l7wght, F7I, F7O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
	int res2 = FcXnorNoBinWrap(l8act_bin, l8wght, F8I, F8O, output, bn4mean, bn4var, bn4gamma, bn4beta,1,1);
	*/
}


/*
int testlayer1() {
	const int input_bw = 2;
	const int output_bw = 1;
	//const int pckWdt = 32;

	//uint8_t *curr_im = l1_act;
	//TODO: Need to divide curr_im length by pack width
	const unsigned int curr_im_len = (unsigned int) (1*3*48*48);
	pckDtype curr_im[curr_im_len*input_bw];
	for (unsigned int i = 0; i < curr_im_len*input_bw; i++)
		curr_im[i] = -1;

	int curr_im_int[curr_im_len];
	pckDtype_to_int(curr_im, curr_im_int, curr_im_len, 32, 2);
	printf("input values (cwhn): ");
	print_int_array(curr_im_int, 10);

	int l1wght_len = 32*3*3*3;
	int l1wght_int[l1wght_len];
	pckDtype_to_int(l1wght, l1wght_int, l1wght_len, 32, 1);
	printf("parameter values (cwhn): ");
	print_int_array(l1wght_int, 10);
	
	int count = 0;
	for (int i = 0; i < l1wght_len; i++)
		if (l1wght_int[i] != -1) count++;
	printf("param count of nonzero %d/%d\n", count, l1wght_len);

	int res;
	CnBnMulti(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset,2,1);
	
	int l2act_bin_len = C2XY*C2XY*C2Z;
	int l2act_bin_int[l2act_bin_len];
	pckDtype_to_int(l2act_bin, l2act_bin_int, l2act_bin_len, 32, 1);
	printf("output values (cwhn): ");
	print_int_array(l2act_bin_int, 10);
}
*/

int testlayer1() {
	const int input_bw = 2;
	const int output_bw = 1;

	//uint8_t *curr_im = l1_act;
	const int l1act_len = (int) (1*C1Z*C1XY*C1XY);
	const int l1act_len_pck = (int) (C1XY*C1XY*((int) ceil(C1Z*1./pckWdt))*input_bw);//(int) ceil(1.0*l1act_len*input_bw/pckWdt);

	pckDtype l1act_bin[l1act_len_pck];
	for (unsigned int i = 0; i < l1act_len_pck; i++)
		l1act_bin[i] = -1; //0xaaaaaaaa;
	int l1act_int[l1act_len];
	pckDtype_to_int(l1act_bin, l1act_int, l1act_len, pckWdt, input_bw);	
	printf("input values (cwhn): ");
	print_int_array(l1act_int, 32);

	const int l1act_bin_unrolled_len = (C1XY-C1KXY+1)*(C1XY-C1KXY+1)*input_bw;
	printf("l1act_bin_unrolled_len %d\n", l1act_bin_unrolled_len);
	pckDtype l1act_bin_unrolled[l1act_bin_unrolled_len];
	//unRollActivations_old(l1act_bin, C1XY, C1XY, C1Z, C1KXY, C1KXY, input_bw,l1act_bin_unrolled);
	unRollActivations(l1act_bin, l1act_bin_unrolled, C1Z, C1XY, C1XY, C1KXY, C1KXY, C1PD, C1PL, input_bw);

	//manually setting unrolled input values
	/*for (int i = 0; i < 4314; i++)
		l1act_bin_unrolled[i] = 0x07ffffff;*/

	int l1wght_len = C1Z*C1KXY*C1KXY*C1KZ;
	int l1wght_int[l1wght_len];
	pckDtype_to_int_var_bits(l1wght, l1wght_int, l1wght_len, pckWdt, 1, 27);
	printf("parameter values (nhwc): ");
	print_int_array(l1wght_int, 32);
	int count = 0;
	for (int i = 0; i < l1wght_len; i++)
		if (l1wght_int[i] >= 0) count++;
	printf("param count of nonnegative %d/%d\n", count, l1wght_len);
	printf("C1Z %u C1XY %u C1XY %u C1Z %u C1KXY %u C1KXY %u C1KZ %u C1PD  %u C1PL %u\n", C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL);

	//CnBnMulti(l1act_bn, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL,NULL,2,1);
	CnBnMulti(l1act_bin_unrolled, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr,bn1sign,bn1offset,2,1);
	
	int l2act_len = C2XY*C2XY*C2Z;
	int l2act_int[l2act_len];
	pckDtype_to_int(l2act_bin, l2act_int, l2act_len, pckWdt, output_bw);
	printf("output values (nhwc): ");
	print_int_array(l2act_int, 128);

	count = 0;
	for (int i = 0; i < l2act_len; i++)
		if (l2act_int[i] >= 0) count++;
	printf("output count of nonnegative %d/%d\n", count, l2act_len);
}

int testlayer2() {
	const int input_bw = 1;
	const int output_bw = 1;

	//uint8_t *curr_im = l1_act;
	const int l2act_len = (int) (1*C2Z*C2XY*C2XY);
	const int l2act_len_pck = (int) ceil(1.0*l2act_len*input_bw/pckWdt);

	//pckDtype l2act_bin[l2act_len_pck]; //defined above
	for (unsigned int i = 0; i < l2act_len_pck; i++)
		l2act_bin[i] = -1; //0xaaaaaaaa;
	int l2act_int[l2act_len];
	pckDtype_to_int(l2act_bin, l2act_int, l2act_len, pckWdt, input_bw);
	printf("input values (cwhn): ");
	print_int_array(l2act_int, 32);

	int l2wght_len = C2Z*C2KXY*C2KXY*C2KZ;
	int l2wght_int[l2wght_len];
	pckDtype_to_int(l2wght, l2wght_int, l2wght_len, pckWdt, 1);
	printf("parameter values (nhwc): ");
	print_int_array(l2wght_int, 10);
	int count = 0;
	for (int i = 0; i < l2wght_len; i++)
		if (l2wght_int[i] >= 0) count++;
	printf("param count of nonnegative %d/%d\n", count, l2wght_len);
	printf("C2Z %u C2XY %u C2XY %u C2Z %u C2KXY %u C2KXY %u C2KZ %u C2PD  %u C2PL %u\n", C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, C2PD, C2PL);

	int res;
	//res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, NULL, NULL,NULL,1,1);
	res = CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn2thr, bn2sign,bn2offset,1,1);
	
	int l3act_len = C3XY*C3XY*C3Z;
	int l3act_int[l3act_len];
	pckDtype_to_int(l3act_bin, l3act_int, l3act_len, pckWdt, output_bw);
	printf("output values (nhwc): ");
	print_int_array(l3act_int, 128);

	count = 0;
	for (int i = 0; i < l3act_len; i++)
		if (l3act_int[i] >= 0) count++;
	printf("output count of nonnegative %d/%d\n", count, l3act_len);
}

int testlayer3() {
	const int input_bw = 1;
	const int output_bw = 1;

	//uint8_t *curr_im = l1_act;
	const int l3act_len = (int) (1*C3Z*C3XY*C3XY);
	const int l3act_len_pck = (int) ceil(1.0*l3act_len*input_bw/pckWdt);

	//pckDtype l3act_bin[l3act_len_pck]; //defined above
	for (unsigned int i = 0; i < l3act_len_pck; i++)
		l3act_bin[i] = -1; //0xaaaaaaaa;
	int l3act_int[l3act_len];
	pckDtype_to_int(l3act_bin, l3act_int, l3act_len, pckWdt, input_bw);
	printf("input values (cwhn): ");
	print_int_array(l3act_int, 32);

	int l3wght_len = 32*3*3*32;
	int l3wght_int[l3wght_len];
	pckDtype_to_int(l3wght, l3wght_int, l3wght_len, pckWdt, 1);
	printf("parameter values (nhwc): ");
	print_int_array(l3wght_int, 10);
	int count = 0;
	for (int i = 0; i < l3wght_len; i++)
		if (l3wght_int[i] >= 0) count++;
	printf("param count of nonnegative %d/%d\n", count, l3wght_len);
	printf("C3Z %u C3XY %u C3XY %u C3Z %u C3KXY %u C3KXY %u C3KZ %u C3PD  %u C3PL %u\n", C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, C3PD, C3PL);

	int res;
	//res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, NULL, NULL,NULL,1,1);
	res = CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn3thr, bn3sign,bn3offset,1,1);
	
	int l4act_len = C4XY*C4XY*C4Z;
	int l4act_int[l4act_len];
	pckDtype_to_int(l4act_bin, l4act_int, l4act_len, pckWdt, output_bw);
	printf("output values (nhwc): ");
	print_int_array(l4act_int, 128);

	count = 0;
	for (int i = 0; i < l4act_len; i++)
		if (l4act_int[i] >= 0) count++;
	printf("output count of nonnegative %d/%d\n", count, l4act_len);
}

int testlayer4() {
	const int input_bw = 1;
	const int output_bw = 1;

	//uint8_t *curr_im = l1_act;
	const int l4act_len = (int) (1*C4Z*C4XY*C4XY);
	const int l4act_len_pck = (int) ceil(1.0*l4act_len*input_bw/pckWdt);

	//pckDtype l2act_bin[l2act_len_pck]; //defined above
	for (unsigned int i = 0; i < l4act_len_pck; i++)
		l4act_bin[i] = -1; //0xaaaaaaaa;
	int l4act_int[l4act_len];
	pckDtype_to_int(l4act_bin, l4act_int, l4act_len, pckWdt, input_bw);
	printf("input values (cwhn): ");
	print_int_array(l4act_int, 32);

	int l4wght_len = C4Z*C4KXY*C4KXY*C4KZ;
	int l4wght_int[l4wght_len];
	pckDtype_to_int(l4wght, l4wght_int, l4wght_len, pckWdt, 1);
	printf("parameter values (nhwc): ");
	print_int_array(l4wght_int, 10);
	int count = 0;
	for (int i = 0; i < l4wght_len; i++)
		if (l4wght_int[i] >= 0) count++;
	printf("param count of nonnegative %d/%d\n", count, l4wght_len);
	printf("C4Z %u C4XY %u C4XY %u C4Z %u C4KXY %u C4KXY %u C4KZ %u C4PD  %u C4PL %u\n", C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, C4PD, C4PL);

	int res;
	//res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, NULL, NULL,NULL,1,1);
	res = CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, bn4thr, bn4sign,bn4offset,1,1);
	
	int l5act_len = F5I;
	int l5act_int[l5act_len];
	pckDtype_to_int(l5act_bin, l5act_int, l5act_len, pckWdt, output_bw);
	printf("output values (nhwc): ");
	print_int_array(l5act_int, 64);

	count = 0;
	for (int i = 0; i < l5act_len; i++)
		if (l5act_int[i] >= 0) count++;
	printf("output count of nonnegative %d/%d\n", count, l5act_len);
}
