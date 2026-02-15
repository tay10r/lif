#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

  void NICE__LeakyReLU(const float* restrict x, float* restrict y);

  void NICE__LeakyReLU_Grad(const float* restrict x,
                            const float* restrict grad_y,
                            float* restrict grad_x);

  void NICE__SoftSign(const float* restrict x, float* restrict y);

  void NICE__SoftSign_Grad(const float* restrict x,
                           const float* restrict grad_y,
                           float* restrict grad_x);

  float NICE__MSE(const float* restrict y, const float* restrict y_target);

  void NICE__MSE_Grad(const float* restrict y,
                      const float* restrict y_target,
                      float* restrict grad_y);

  void NICE__VecAdd(const float* restrict x,
                    const float* restrict z,
                    float* restrict y);

  void NICE__VecAdd_Grad(const float* restrict grad_y,
                         float* restrict grad_x,
                         float* restrict grad_z);

  void NICE__MatVecMul_32x32(const float* restrict A,
                             const float* restrict x,
                             float* restrict out);

  void NICE__MatVecMul_32x32_Grad(const float* restrict A,
                                  const float* restrict x,
                                  const float* restrict grad_out,
                                  float* restrict grad_A,
                                  float* restrict grad_x);

  void NICE__MatVecMul_32x192(const float* restrict A,
                              const float* restrict x,
                              float* restrict out);

  void NICE__MatVecMul_32x192_Grad(const float* restrict A,
                                   const float* restrict x,
                                   const float* restrict grad_out,
                                   float* restrict grad_A,
                                   float* restrict grad_x);

  void NICE__MatVecMul_192x32(const float* restrict A,
                              const float* restrict x,
                              float* restrict out);

  void NICE__MatVecMul_192x32_Grad(const float* restrict A,
                                   const float* restrict x,
                                   const float* restrict grad_out,
                                   float* restrict grad_A,
                                   float* restrict grad_x);
#ifdef __cplusplus
} /* extern "C" */
#endif
