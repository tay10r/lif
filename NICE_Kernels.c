#include "NICE_Kernels.h"

#include <math.h>

#define DIM 64
#define LANES 4
#define LATENT_DIM 4

void
nice__mat_vec_mul(const float* restrict W, const float* restrict x, float* restrict y)
{
  for (int i = 0; i < DIM; i += LANES) {
    float lanes[LANES] = { 0, 0, 0, 0 };
    for (int j = 0; j < DIM; j++) {
      const float x_j = x[j];
      lanes[0] += W[(i + 0) * DIM + j] * x_j;
      lanes[1] += W[(i + 1) * DIM + j] * x_j;
      lanes[2] += W[(i + 2) * DIM + j] * x_j;
      lanes[3] += W[(i + 3) * DIM + j] * x_j;
    }
    y[i + 0] = lanes[0];
    y[i + 1] = lanes[1];
    y[i + 2] = lanes[2];
    y[i + 3] = lanes[3];
  }
}

void
nice__vec_add(const float* restrict x, const float* restrict z, float* restrict y)
{
  for (int i = 0; i < DIM; i++) {
    y[i] = x[i] + z[i];
  }
}

void
nice__relu(const float* restrict x, float* restrict y)
{
  for (int i = 0; i < DIM; i++) {
    y[i] = (x[i] > 0.0F) ? x[i] : 0.0F;
  }
}

void
nice__mat_vec_mul_4x64(const float* restrict A, const float* restrict x, float* restrict out)
{
  float lanes[LATENT_DIM] = { 0, 0, 0, 0 };

  for (int i = 0; i < DIM; i++) {
    const float x_i = x[i];
    lanes[0] += A[0 * DIM + i] * x_i;
    lanes[1] += A[1 * DIM + i] * x_i;
    lanes[2] += A[2 * DIM + i] * x_i;
    lanes[3] += A[3 * DIM + i] * x_i;
  }

  out[0] = lanes[0];
  out[1] = lanes[1];
  out[2] = lanes[2];
  out[3] = lanes[3];
}

void
nice__mat_vec_mul_64x4(const float* restrict A, const float* restrict x, float* restrict out)
{
  const float x0 = x[0];
  const float x1 = x[1];
  const float x2 = x[2];
  const float x3 = x[3];

  for (int i = 0; i < DIM; ++i) {

    const float* restrict row = A + i * LATENT_DIM;

    out[i] = row[0] * x0 + row[1] * x1 + row[2] * x2 + row[3] * x3;
  }
}

void
nice__softsign(const float* restrict x, float* restrict y)
{
  for (int i = 0; i < LATENT_DIM; i++) {
    y[i] = x[i] / (1.0F + fabsf(x[i]));
  }
}
