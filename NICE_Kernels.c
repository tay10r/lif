#include "NICE_Kernels.h"

#include <math.h>

#define BLOCK_SIZE 8
#define DIM_X (3 * BLOCK_SIZE * BLOCK_SIZE)
#define LANES 4
#define LATENT_DIM 32

void
NICE__LeakyReLU(const float* restrict x, float* restrict y)
{
  for (int i = 0; i < LATENT_DIM; i++) {
    const float x_i = x[i];
    y[i] = x_i * ((x_i > 0.0F) ? 1.0F : 0.01F);
  }
}

void
NICE__LeakyReLU_Grad(const float* restrict x,
                     const float* restrict grad_y,
                     float* restrict grad_x)
{
  for (int i = 0; i < LATENT_DIM; i++) {
    const float x_i = x[i];
    grad_x[i] = grad_y[i] * ((x_i > 0.0F) ? 1.0F : 0.01F);
  }
}

void
NICE__SoftSign(const float* restrict x, float* restrict y)
{
  for (int i = 0; i < LATENT_DIM; i++) {
    // y[i] = x[i] / (1.0F + fabsf(x[i]));
    y[i] = x[i] > 0.0F ? 1.0F : -1.0F;
  }
}

void
NICE__SoftSign_Grad(const float* restrict x,
                    const float* restrict grad_y,
                    float* restrict grad_x)
{
  for (int i = 0; i < LATENT_DIM; i++) {
    const float x_i = x[i];
    const float d = 1.0F / (1.0F + fabsf(x_i));
    grad_x[i] = grad_y[i] * d * d;
  }
}

float
NICE__MSE(const float* restrict y, const float* restrict y_target)
{
  float sum = 0.0F;

  for (int i = 0; i < DIM_X; i++) {

    const float d = y[i] - y_target[i];

    sum += d * d;
  }

  return sum * (1.0F / (float)DIM_X);
}

void
NICE__MatVecMul_32x32(const float* restrict W,
                      const float* restrict x,
                      float* restrict y)
{
  for (int i = 0; i < LATENT_DIM; i += LANES) {

    float lanes[LANES] = { 0, 0, 0, 0 };

    for (int j = 0; j < LATENT_DIM; j++) {
      const float x_j = x[j];
      lanes[0] += W[(i + 0) * LATENT_DIM + j] * x_j;
      lanes[1] += W[(i + 1) * LATENT_DIM + j] * x_j;
      lanes[2] += W[(i + 2) * LATENT_DIM + j] * x_j;
      lanes[3] += W[(i + 3) * LATENT_DIM + j] * x_j;
    }

    y[i + 0] = lanes[0];
    y[i + 1] = lanes[1];
    y[i + 2] = lanes[2];
    y[i + 3] = lanes[3];
  }
}

void
NICE__MatVecMul_32x192(const float* restrict W,
                       const float* restrict x,
                       float* restrict y)
{
  // multiply a 32-D vector by a 192x32 matrix, producing a 192-D vector

  for (int i = 0; i < DIM_X; i += LANES) {

    float lanes[LANES] = { 0, 0, 0, 0 };

    for (int j = 0; j < LATENT_DIM; j++) {
      const float x_j = x[j];
      lanes[0] += W[(i + 0) * LATENT_DIM + j] * x_j;
      lanes[1] += W[(i + 1) * LATENT_DIM + j] * x_j;
      lanes[2] += W[(i + 2) * LATENT_DIM + j] * x_j;
      lanes[3] += W[(i + 3) * LATENT_DIM + j] * x_j;
    }

    y[i + 0] = lanes[0];
    y[i + 1] = lanes[1];
    y[i + 2] = lanes[2];
    y[i + 3] = lanes[3];
  }
}

void
NICE__MatVecMul_192x32(const float* restrict W,
                       const float* restrict x,
                       float* restrict y)
{
  // multiply a 192-D vector by a 32x192 matrix, producing a 32-D vector

  for (int i = 0; i < LATENT_DIM; i += LANES) {

    float lanes[LANES] = { 0, 0, 0, 0 };

    for (int j = 0; j < DIM_X; j++) {
      const float x_j = x[j];
      lanes[0] += W[(i + 0) * DIM_X + j] * x_j;
      lanes[1] += W[(i + 1) * DIM_X + j] * x_j;
      lanes[2] += W[(i + 2) * DIM_X + j] * x_j;
      lanes[3] += W[(i + 3) * DIM_X + j] * x_j;
    }

    y[i + 0] = lanes[0];
    y[i + 1] = lanes[1];
    y[i + 2] = lanes[2];
    y[i + 3] = lanes[3];
  }
}

void
NICE__VecAdd(const float* restrict x,
             const float* restrict z,
             float* restrict y)
{
  for (int i = 0; i < LATENT_DIM; i++) {
    y[i] = x[i] + z[i];
  }
}

void
NICE__VecAdd_Grad(const float* restrict grad_y,
                  float* restrict grad_x,
                  float* restrict grad_z)
{
  for (int i = 0; i < LATENT_DIM; i++) {
    const float g = grad_y[i];
    grad_x[i] = g;
    grad_z[i] = g;
  }
}

void
NICE__MSE_Grad(const float* restrict y,
               const float* restrict y_target,
               float* restrict grad_y)
{
  const float invN = 1.0F / (float)DIM_X;

  for (int i = 0; i < DIM_X; i++) {
    const float d = y[i] - y_target[i];
    grad_y[i] = 2.0F * d * invN;
  }
}

void
NICE__MatVecMul_32x32_Grad(const float* restrict W,
                           const float* restrict x,
                           const float* restrict grad_y,
                           float* restrict grad_W,
                           float* restrict grad_x)
{
  for (int j = 0; j < LATENT_DIM; j++) {
    grad_x[j] = 0.0F;
  }

  for (int i = 0; i < LATENT_DIM; i += LANES) {

    const float gy0 = grad_y[i + 0];
    const float gy1 = grad_y[i + 1];
    const float gy2 = grad_y[i + 2];
    const float gy3 = grad_y[i + 3];

    for (int j = 0; j < LATENT_DIM; j++) {
      const float x_j = x[j];

      grad_W[(i + 0) * LATENT_DIM + j] += gy0 * x_j;
      grad_W[(i + 1) * LATENT_DIM + j] += gy1 * x_j;
      grad_W[(i + 2) * LATENT_DIM + j] += gy2 * x_j;
      grad_W[(i + 3) * LATENT_DIM + j] += gy3 * x_j;

      grad_x[j] += W[(i + 0) * LATENT_DIM + j] * gy0;
      grad_x[j] += W[(i + 1) * LATENT_DIM + j] * gy1;
      grad_x[j] += W[(i + 2) * LATENT_DIM + j] * gy2;
      grad_x[j] += W[(i + 3) * LATENT_DIM + j] * gy3;
    }
  }
}

void
NICE__MatVecMul_32x192_Grad(const float* restrict W,
                            const float* restrict x,
                            const float* restrict grad_y,
                            float* restrict grad_W,
                            float* restrict grad_x)
{
  for (int j = 0; j < LATENT_DIM; j++) {
    grad_x[j] = 0.0F;
  }

  for (int i = 0; i < DIM_X; i += LANES) {

    const float gy0 = grad_y[i + 0];
    const float gy1 = grad_y[i + 1];
    const float gy2 = grad_y[i + 2];
    const float gy3 = grad_y[i + 3];

    for (int j = 0; j < LATENT_DIM; j++) {
      const float x_j = x[j];

      grad_W[(i + 0) * LATENT_DIM + j] += gy0 * x_j;
      grad_W[(i + 1) * LATENT_DIM + j] += gy1 * x_j;
      grad_W[(i + 2) * LATENT_DIM + j] += gy2 * x_j;
      grad_W[(i + 3) * LATENT_DIM + j] += gy3 * x_j;

      grad_x[j] += W[(i + 0) * LATENT_DIM + j] * gy0;
      grad_x[j] += W[(i + 1) * LATENT_DIM + j] * gy1;
      grad_x[j] += W[(i + 2) * LATENT_DIM + j] * gy2;
      grad_x[j] += W[(i + 3) * LATENT_DIM + j] * gy3;
    }
  }
}

void
NICE__MatVecMul_192x32_Grad(const float* restrict W,
                            const float* restrict x,
                            const float* restrict grad_y,
                            float* restrict grad_W,
                            float* restrict grad_x)
{
  for (int j = 0; j < DIM_X; j++) {
    grad_x[j] = 0.0F;
  }

  for (int i = 0; i < LATENT_DIM; i += LANES) {

    const float gy0 = grad_y[i + 0];
    const float gy1 = grad_y[i + 1];
    const float gy2 = grad_y[i + 2];
    const float gy3 = grad_y[i + 3];

    for (int j = 0; j < DIM_X; j++) {
      const float x_j = x[j];

      grad_W[(i + 0) * DIM_X + j] += gy0 * x_j;
      grad_W[(i + 1) * DIM_X + j] += gy1 * x_j;
      grad_W[(i + 2) * DIM_X + j] += gy2 * x_j;
      grad_W[(i + 3) * DIM_X + j] += gy3 * x_j;

      grad_x[j] += W[(i + 0) * DIM_X + j] * gy0;
      grad_x[j] += W[(i + 1) * DIM_X + j] * gy1;
      grad_x[j] += W[(i + 2) * DIM_X + j] * gy2;
      grad_x[j] += W[(i + 3) * DIM_X + j] * gy3;
    }
  }
}
