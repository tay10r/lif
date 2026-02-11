#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

  void nice__mat_vec_mul(const float* restrict W, const float* restrict x, float* restrict y);

  void nice__vec_add(const float* restrict x, const float* restrict z, float* restrict y);

  void nice__relu(const float* restrict x, float* restrict y);

  void nice__mat_vec_mul_4x64(const float* restrict A, const float* restrict x, float* restrict out);

  void nice__mat_vec_mul_64x4(const float* restrict A, const float* restrict x, float* restrict out);

  void nice__softsign(const float* restrict x, float* restrict y);

#ifdef __cplusplus
} /* extern "C" */
#endif
