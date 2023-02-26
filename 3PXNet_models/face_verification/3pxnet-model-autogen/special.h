#include <datatypes.h>
#include <math.h>
#include <stdio.h>

#ifndef ESP
#define ESP
#endif

void printPackedArray(pckDtype *arr, int n) {
   for (int i = 0; i < n; i++)
      printf("%x ", arr[i]);
   printf("\n");
}

void printFloatArray(float *arr, int n) {
   for (int i = 0; i < n; i++)
      printf("%f ", arr[i]);
   printf("\n");
}

void pack_int8(int8_t* src, pckDtype* target, int n) {
   //assert n%pckWdt == 0
   for (int i = 0; i < n/pckWdt; i++) {
      for (int j = 0; j < pckWdt; j++) {
         int value = src[i*pckWdt + j]>=0 ? 1 : 0;
         target[i] &= ~(1 << (pckWdt-j-1)); //all 1s except 0 at index j
         target[i] |= (value << (pckWdt-j-1)); //all 0s except sign(src[i*pcWdt+j]) at index j
         //printf("%f %d %x\n", src[i*pckWdt + j], value, target[i]);
      }
   }
}

void pack(bnDtype* src, pckDtype* target, int n) {
   //assert n%pckWdt == 0
   for (int i = 0; i < n/pckWdt; i++) {
      for (int j = 0; j < pckWdt; j++) {
         int value = src[i*pckWdt + j]>=0 ? 1 : 0;
         target[i] &= ~(1 << (pckWdt-j-1)); //all 1s except 0 at index j
         target[i] |= (value << (pckWdt-j-1)); //all 0s except sign(src[i*pcWdt+j]) at index j
         //printf("%f %d %x\n", src[i*pckWdt + j], value, target[i]);
      }
   }
}

void reluPack(pckDtype *arr, int n) {
   for (int i = 0; i < n; i++)
      *arr++ = -1;
}

void reluFloat(bnDtype *arr, int n) {
   for (int i = 0; i < n; i++) {
      *arr = *arr < 0 ? 0 : *arr;
      arr++;
   }
}

//target += src;
void addFloat(bnDtype *src, bnDtype *target, int n) {
   for (int i = 0; i < n; i++) {
      target[i] += src[i];
   }
}

void relu_int8(int8_t *arr, int n) {
   for (int i = 0; i < n; i++) {
      *arr = *arr < 0 ? 0 : *arr;
      arr++;
   }
}

//target += src;
void add_int8(int8_t *src, int8_t *target, int n) {
   for (int i = 0; i < n; i++) {
      target[i] += src[i];
   }
}

//height, width, depth
void averagePool1_1(bnDtype *in, bnDtype *out, const uint16_t dpth, const uint16_t wdth, const uint16_t hght) {
   for (uint16_t i = 0; i < dpth; i++) out[i]=0;

   const uint16_t yCoeff = wdth*dpth;
   const uint16_t xCoeff = dpth;
   
   for (uint16_t i = 0; i < hght; i++) {
      for (uint16_t j = 0; j < wdth; j++) {
         for (uint16_t k = 0; k < dpth; k++) {
            out[k] += in[i*yCoeff + j*xCoeff  + k];
         }
      }
   }

   for (uint16_t i = 0; i < dpth; i++) out[i]/=hght*wdth;
}

void averagePool1_1_int8(int8_t *in, int8_t *out, const uint16_t dpth, const uint16_t wdth, const uint16_t hght) {
   for (uint16_t i = 0; i < dpth; i++) out[i]=0;

   const uint16_t yCoeff = wdth*dpth;
   const uint16_t xCoeff = dpth;
   
   for (uint16_t i = 0; i < hght; i++) {
      for (uint16_t j = 0; j < wdth; j++) {
         for (uint16_t k = 0; k < dpth; k++) {
            out[k] += in[i*yCoeff + j*xCoeff  + k];
         }
      }
   }

   for (uint16_t i = 0; i < dpth; i++) out[i]/=hght*wdth;
}

void bwn_fc(bnDtype *in, pckDtype *krn, bnDtype *out, const uint16_t numIn, const uint16_t numOut) {
   for (int i = 0; i < numOut; i++) {
      out[i] = 0;
      for (int j = 0; j < numIn; j++) {
         if ((krn[(i*numIn + j)/pckWdt]&(1<<(pckWdt - 1 - (i*numIn + j)%pckWdt)))!=0) out[i] += in[j];
         else out[i] -= in[j];
      }
   }
}

void bwn_fc_unpacked(bnDtype *in, bnDtype *krn, bnDtype *out, const uint16_t numIn, const uint16_t numOut) {
   for (int i = 0; i < numOut; i++) {
      out[i] = 0;
      for (int j = 0; j < numIn; j++) {
         if (krn[i*numIn + j]) out[i] += in[j];
         else out[i] -= in[j];
      }
   }
}

void normalize(bnDtype *in, int n) {
   float magnitude = 1e-3;
   for (int i = 0; i < n; i++) {
      magnitude += in[i]*in[i];
   }
   magnitude = sqrt(magnitude);

   for (int i = 0; i < n; i++) {
      in[i] /= magnitude;
   }
}

struct identity_block_conf {
   int8_t *C_1_act_unpacked; pckDtype *C_1_act; pckDtype *C_2_act;int8_t *C_3_act_unpacked; pckDtype *C_1_wgt; pckDtype *C_2_wgt; 
   bnDtype *C_2_mean; bnDtype *C_2_var; bnDtype *C_2_gamma; bnDtype *C_2_beta; bnDtype *C_1_thresh; pckDtype *C_1_sign;
   uint8_t C_1XY; uint8_t C_1Z; uint8_t C_1KXY; uint8_t C_1KZ; uint8_t C_1PD; uint8_t C_1PL;
   uint8_t C_2XY; uint8_t C_2Z; uint8_t C_2KXY; uint8_t C_2KZ; uint8_t C_2PD; uint8_t C_2PL; uint8_t C_2OXY;
};

struct convolutional_block_conf {
   int8_t *C_1_act_unpacked; pckDtype *C_1_act; pckDtype *C_2_act; int8_t *C_3_act_unpacked; pckDtype *C_1_wgt; pckDtype *C_2_wgt; 
   bnDtype *C_2_mean; bnDtype *C_2_var; bnDtype *C_2_gamma; bnDtype *C_2_beta; bnDtype *C_1_thresh; pckDtype *C_1_sign;
   uint8_t C_1XY; uint8_t C_1Z; uint8_t C_1KXY; uint8_t C_1KZ; uint8_t C_1PD; uint8_t C_1PL;
   uint8_t C_2XY; uint8_t C_2Z; uint8_t C_2KXY; uint8_t C_2KZ; uint8_t C_2PD; uint8_t C_2PL; uint8_t C_2OXY;
   int8_t *C_d_act_unpacked; pckDtype *C_d_wgt; bnDtype *C_d_mean; bnDtype *C_d_var; bnDtype *C_d_gamma; bnDtype *C_d_beta;
   uint8_t C_dKXY; uint8_t C_dKZ; uint8_t C_dPD; uint8_t C_dPL; uint8_t C_dOXY;
};

//output: C_3_act_unpacked
void identity_block(struct identity_block_conf s)
{
   #ifdef ESP
   int64_t s1, s2;
   s1 = esp_timer_get_time();
   #endif

   pack_int8(s.C_1_act_unpacked, s.C_1_act, s.C_1XY*s.C_1XY*s.C_1Z);
   //printPackedArray(s.C_1_act, s.C_2XY*s.C_2XY*s.C_2Z/pckWdt);

   #ifdef ESP
   s2 = esp_timer_get_time();
   printf("\tidentity block _ time (mus): %f\n", ((float)(s2-s1)));
   //fflush(stdout);
   s1 = esp_timer_get_time();
   //printFloatArray(buffer2, 100);
   #endif

   CnXnorWrap(s.C_1_act, s.C_1_wgt, s.C_1Z, s.C_1XY, s.C_1XY, s.C_1Z, s.C_1KXY, s.C_1KXY, s.C_1KZ, s.C_2_act, s.C_1PD, s.C_1PL, s.C_1_thresh, s.C_1_sign);
   //CnXnorWrap(s.C_1_act, s.C_1_wgt, s.C_1Z, s.C_1XY, s.C_1XY, s.C_1Z, s.C_1KXY, s.C_1KXY, s.C_1KZ, s.C_2_act, 0, s.C_1PL, s.C_1_thresh, s.C_1_sign);
   //printPackedArray(s.C_2_act+s.C_1XY*s.C_1XY*s.C_1Z/pckWdt-20, 20);
   
   #ifdef ESP
   s2 = esp_timer_get_time();
   printf("\tidentity block _ time (mus): %f\n", ((float)(s2-s1)));
   //fflush(stdout);
   s1 = s2;
   //printFloatArray(buffer2, 100);
   #endif

   reluPack(s.C_2_act, s.C_2XY*s.C_2XY*s.C_2Z/pckWdt);
   
   #ifdef ESP
   s2 = esp_timer_get_time();
   printf("\tidentity block _ time (mus): %f\n", ((float)(s2-s1)));
   //fflush(stdout);
   s1 = s2;
   //printFloatArray(buffer2, 100);
   #endif

   CnXnorNoBinWrap(s.C_2_act, s.C_2_wgt, s.C_2Z, s.C_2XY, s.C_2XY, s.C_2Z, s.C_2KXY, s.C_2KXY, s.C_2KZ, s.C_3_act_unpacked, s.C_2PD, s.C_2PL, s.C_2_mean, s.C_2_var, s.C_2_gamma, s.C_2_beta);
   
   #ifdef ESP
   s2 = esp_timer_get_time();
   printf("\tidentity block _ time (mus): %f\n", ((float)(s2-s1)));
   //fflush(stdout);
   s1 = s2;
   //printFloatArray(buffer2, 100);
   #endif

   //printFloatArray(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   //printFloatArray(s.C_1_act_unpacked, s.C_1XY*s.C_1XY*s.C_1Z);

   //printFloatArray(s.C_1_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
   //printFloatArray(s.C_3_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
   
   //addFloat(s.C_1_act_unpacked, s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   add_int8(s.C_1_act_unpacked, s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   
   #ifdef ESP
   s2 = esp_timer_get_time();
   printf("\tidentity block _ time (mus): %f\n", ((float)(s2-s1)));
   //fflush(stdout);
   s1 = s2;
   //printFloatArray(buffer2, 100);
   #endif

   //reluFloat(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   relu_int8(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   
   #ifdef ESP
   s2 = esp_timer_get_time();
   printf("\tidentity block _ time (mus): %f\n", ((float)(s2-s1)));
   //fflush(stdout);
   s1 = s2;
   //printFloatArray(buffer2, 100);
   #endif

   //printFloatArray(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   //printFloatArray(s.C_3_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
}

//output: C_3_act_unpacked
void convolutional_block(struct convolutional_block_conf s)
{
   pack_int8(s.C_1_act_unpacked, s.C_1_act, s.C_1XY*s.C_1XY*s.C_1Z);
   //printPackedArray(C_1_act, sizeof(C_1_act)/sizeof(pckDtype));
   CnXnorWrap(s.C_1_act, s.C_1_wgt, s.C_1Z, s.C_1XY, s.C_1XY, s.C_1Z, s.C_1KXY, s.C_1KXY, s.C_1KZ, s.C_2_act, s.C_1PD, s.C_1PL, s.C_1_thresh, s.C_1_sign);
   //printPackedArray(C_2_act, sizeof(C_2_act)/sizeof(pckDtype));
   reluPack(s.C_2_act, s.C_2XY*s.C_2XY*s.C_2Z/pckWdt);
   CnXnorNoBinWrap(s.C_2_act, s.C_2_wgt, s.C_2Z, s.C_2XY, s.C_2XY, s.C_2Z, s.C_2KXY, s.C_2KXY, s.C_2KZ, s.C_3_act_unpacked, s.C_2PD, s.C_2PL, s.C_2_mean, s.C_2_var, s.C_2_gamma, s.C_2_beta);
   //printFloatArray(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   
   //printPackedArray(s.C_1_act, C4_1XY*C4_1XY*C4_1Z/pckWdt);
   CnXnorNoBinWrap(s.C_1_act, s.C_d_wgt, s.C_1Z, s.C_1XY, s.C_1XY, s.C_1Z, s.C_dKXY, s.C_dKXY, s.C_dKZ, s.C_d_act_unpacked, s.C_dPD, s.C_dPL, s.C_d_mean, s.C_d_var, s.C_d_gamma, s.C_d_beta);
   //printFloatArray(s.C_d_act_unpacked, s.C_dOXY*s.C_dOXY*s.C_dKZ);
   
   //printFloatArray(s.C_d_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
   //printFloatArray(s.C_3_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
   /*
   addFloat(s.C_d_act_unpacked, s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   reluFloat(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   */

   add_int8(s.C_1_act_unpacked, s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   relu_int8(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);

   //printFloatArray(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   //printFloatArray(s.C_3_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
}

//Source: https://github.com/SRavit1/3pxnet-copy/blob/resnet18/3pxnet-inference/val/cn_reference.c

/**
 * @details Reference Fully Connected Layer 
 * @param[in] pAct    - pointer to the packed activation vector (Y/X/Z - depth first)
 * @param[in] pKrn    - pointer to the packed weight vector (K/Y/X/Z - depth first)
 * @param[in] dpth    - Depth (Z)
 * @param[in] wdth    - Width (X)
 * @param[in] hght    - Height (Y)
 * @param[in] kwdt    - Kernel width (KX)
 * @param[in] khgt    - Kernel height (KY)
 * @param[in] knum    - # of kernels
 * @param[out] pOut   - pointer to the output vector (Y/X/Z - depth first)
 * @param[in] pool    - pooling size (stride assumed 1)
 * @param[in] pMean   - pointer to mean vector (if NULL, Batch Norm is skipped)
 * @param[in] pSig    - pointer to sqrt(variance) vector
 * @param[in] pGamma  - pointer to gamma vector
 * @param[in] pBeta   - pointer to beta vector
 */
void refCn_bwn(int16_t * pAct, int16_t * pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, int8_t * pOut,
           const uint16_t pool, float * pMean, float * pSig, float * pGamma, float * pBeta) {

   // Local pointer to activations
   int16_t *pIn = pAct;
   // Local weight pointer
   int16_t *pWgt = pKrn;
   // Batch norm pointers
   float   *pMn = pMean;
   float   *pSg = pSig;
   float   *pGm = pGamma;
   float   *pBt = pBeta;
   // For maxpooling
   float   maxTemp = 0.0;
   // Temp output
   float   outTemp = 0.0;

   // Outer loop - Y
   for (uint16_t y = 0; y < ((hght-khgt+1)/pool); y++) {
      // X
      for (uint16_t x = 0; x < ((wdth-kwdt+1)/pool); x++) {
         // Starting kernel pointer (beginning of first kernel)
         pWgt = pKrn;
         // Starting batch norm pointers
         pMn = pMean;
         pSg = pSig;
         pGm = pGamma;
         pBt = pBeta;
         // Kernel
         for (uint16_t kn = 0; kn < knum; kn++) {
            // Mpool patches
            maxTemp = -10e8;
            for (uint16_t yy = 0; yy < pool; yy++) {
               for (uint16_t xx = 0; xx < pool; xx++) {
                  // Clear output
                  outTemp = 0.0;
                  // Set starting pointer
                  pIn = pAct + (y*pool+yy)*wdth*dpth + (x*pool+xx)*dpth;
                  // Set the weight pointer
                  pWgt = pKrn + kn*khgt*kwdt*dpth;
                  // Kernel-Y
                  for (uint16_t ky = 0; ky < khgt; ky++) {
                     // Kernel-X
                     for (uint16_t kx = 0; kx < kwdt; kx++) {
                        // Z
                        for (uint16_t kz = 0; kz < dpth; kz++) {
                           int16_t value = *pIn++;
                           if ((*pWgt++) == 1) outTemp += value;
                           else outTemp -= value;
                        }
                     } // KX
                     // Move to the next row
                     pIn += (wdth-kwdt)*dpth;
                  } // KY
                  // Running maxpool
                  if (outTemp > maxTemp) { maxTemp = outTemp; }
               }
            }
            // Optional Batch Norm - skip if pMean is NULL
            if (pMean) {
               maxTemp = (int8_t) ((*pGm++)*(((float)maxTemp - *pMn++)/ *pSg++) + *pBt++);
            }
            *pOut++ = maxTemp;
         } // Kernel
      } // X
   } // Y
}


void CnBnMulti(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth,
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt,
    const uint16_t khgt, const uint16_t knum, const uint16_t pad, const uint16_t pool,
    pckDtype* __restrict pOut, pckDtype* __restrict thresh, pckDtype* sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit) {
    // Input counter
    uint16_t inCnt = dpth;
    // Output counter
    uint16_t outCnt = knum;
    // Activations
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    // temporary output value (before binarization)
    uint32_t xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    int32_t pckTemp[out_bit];
    int out = 0;
    uint8_t packing_num = kwdt * khgt * kdpt;
    uint16_t  yCoeff = (wdth-kwdt+1) * in_bit;
    uint16_t  xCoeff = in_bit;
    uint16_t  kCoeff = khgt * kwdt * kdpt;
    // Packing/shifting  index
    uint8_t pckIdx = pckWdt - 1;
    pckDtype* threshLoc = thresh;
    pckDtype* signLoc = sign;
    pckDtype* offsetLoc = offset;
    memset(pckTemp, 0, sizeof(int32_t) * out_bit);
    int max_temp = 0x80000000;
    int mask_pad = 0;
    for (int y = 0; y < (hght + 2 * pad - khgt + 1)/pool; y++) {
        for (int x = 0; x < (wdth + 2 * pad - kwdt + 1)/pool; x++) {  
            outCnt = knum;
            threshLoc = thresh;
            signLoc = sign;
            offsetLoc = offset;
            while (outCnt) {
                max_temp = 0x80000000;
                for (int yy = 0; yy < pool; yy++) {
                    for (int xx = 0; xx < pool; xx++) {
                        pWgt = pKrn;
                        mask_pad = mask(27);
                        packing_num = khgt * kwdt;
                        memset(outTemp, 0, in_bit * sizeof(int32_t));
                        out = 0;
                        inCnt = dpth;
                        pIn = pAct + (y * pool + yy) * yCoeff + (x * pool + xx)* xCoeff;
                        pWgt = pKrn + (knum - outCnt);
                        //while (inCnt) {
                        for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                            // XNOR multiplication
                            xnorTemp[bitw] = (~(*pIn++ ^ *pWgt) & mask_pad);
                            // popcount//Accummulate
                            outTemp[bitw] += popcount(xnorTemp[bitw]);
                        }
                        //pWgt++;
                        //}
                        //pAct -= in_bit * dpth;
                        int8_t packing_offset = pckWdt - packing_num;
                        for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                            // Adjust the output value
                            outTemp[bitw] = outTemp[bitw] - (dpth * pckWdt - outTemp[bitw]);
                            // Get the int full precision value 
                            outTemp[bitw] += packing_offset * dpth;
                            out += (outTemp[bitw] << (in_bit - bitw - 1));
                        }
                        if (out > max_temp) {
                            max_temp = out;
                        }                        
                    }
                    //printf("\n");
                    //pAct += in_bit * dpth;
                }
                // Quantization
                int int_part = max_temp >> (in_bit) << (16);/// pow(2, in_bit);
                int frac_part = (mask(in_bit) & max_temp) << (16 - in_bit);
                //float w = out / 4.0;
                //printf("%.2f, ", w);
                //temp = (temp & out) << (16 - in_bit);
                int out_temp = int_part + frac_part;
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    int temp = out_temp > *threshLoc;
                    // Shift 
                    pckTemp[bitw] |= (temp << (pckIdx));
                    out_temp = (temp ^ (1 & ((*signLoc) >> (pckIdx))) ? out_temp + ((*offsetLoc) >> (bitw + 1)) : out_temp - ((*offsetLoc) >> (bitw + 1)));
                }
                threshLoc++;
                offsetLoc++;
                // Full output block - write out
                if (pckIdx == 0) {
                    for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                        *pOut++ = ~(pckTemp[bitw] ^ (*signLoc));
                    }
                    signLoc++;
                    pckIdx = pckWdt - 1;
                    memset(pckTemp, 0, sizeof(int32_t) * out_bit);
                }
                else {
                    pckIdx--;
                }
                outCnt--;
            }            
        }
    }
}