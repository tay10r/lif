#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

  void NICE__MatVecMul(const float* restrict W, const float* restrict x, float* restrict y);

  void NICE__VecAdd(const float* restrict x, const float* restrict z, float* restrict y);

  void NICE__LeakyReLU(const float* restrict x, float* restrict y);

  void NICE__MatVecMul_256x48(const float* restrict A, const float* restrict x, float* restrict out);

  void NICE__MatVecMul_48x256(const float* restrict A, const float* restrict x, float* restrict out);

  void NICE__MatVecMul_256x192(const float* restrict A, const float* restrict x, float* restrict out);

  void NICE__MatVecMul_192x256(const float* restrict A, const float* restrict x, float* restrict out);

#ifdef __cplusplus
} /* extern "C" */
#endif
