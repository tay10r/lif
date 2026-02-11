#include "NICE.h"

#include "NICE_Kernels.h"
#include "NICE_Weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 8
#define DIM (BLOCK_SIZE * BLOCK_SIZE)
#define SCALE (1.0F / 255.0F)
#define LATENT_DIM 4
#define NUM_ENCODER_WEIGHTS 33536
#define NUM_DECODER_WEIGHTS 33536

#define LINEAR(x, y, p)                                                                                                \
  do {                                                                                                                 \
    float linear_tmp[DIM];                                                                                             \
    nice__mat_vec_mul(p, x, linear_tmp);                                                                               \
    p += DIM * DIM;                                                                                                    \
    nice__vec_add(linear_tmp, p, y);                                                                                   \
    p += DIM;                                                                                                          \
  } while (0)

#define LINEAR_RESNET(x, y, p)                                                                                         \
  do {                                                                                                                 \
    const float* residual = x;                                                                                         \
    float resnet_tmp[DIM];                                                                                             \
    LINEAR(x, resnet_tmp, p);                                                                                          \
    float resnet_tmp2[DIM];                                                                                            \
    nice__relu(resnet_tmp, resnet_tmp2);                                                                               \
    float resnet_tmp3[DIM];                                                                                            \
    LINEAR(resnet_tmp2, resnet_tmp3, p);                                                                               \
    float resnet_tmp4[DIM];                                                                                            \
    nice__vec_add(residual, resnet_tmp3, resnet_tmp4);                                                                 \
    nice__relu(resnet_tmp4, y);                                                                                        \
  } while (0)

struct NICE_Engine
{
  float* encoder;

  float* decoder;

  void* user_data;

  NICE_ErrorHandler error_handler;
};

static const float*
GetEncoder(const NICE_Engine* engine)
{
  return engine->encoder ? engine->encoder : NICE_Encoder;
}

static const float*
GetDecoder(const NICE_Engine* engine)
{
  return engine->decoder ? engine->decoder : NICE_Decoder;
}

static void
NotifyError(NICE_Engine* engine, const char* what)
{
  if (engine->error_handler) {
    engine->error_handler(engine->user_data, what);
  }
}

NICE_Engine*
NICE_NewEngine()
{
  NICE_Engine* self = malloc(sizeof(NICE_Engine));
  if (!self) {
    return NULL;
  }
  memset(self, 0, sizeof(NICE_Engine));
  return self;
}

void
NICE_DestroyEngine(NICE_Engine* engine)
{
  free(engine->encoder);
  free(engine->decoder);
  free(engine);
}

NICE_Result
NICE_LoadEncoder(NICE_Engine* engine, const char* filename)
{
  const size_t s = sizeof(float) * NUM_ENCODER_WEIGHTS;

  void* tmp = malloc(s);
  if (!tmp) {
    NotifyError(engine, "failed to allocate encoder weights");
    return NICE_FAILURE;
  }

  FILE* file = fopen(filename, "rb");
  if (!file) {
    free(tmp);
    NotifyError(engine, "failed to open encoder weights file");
    return NICE_FAILURE;
  }

  const size_t read_size = fread(tmp, 1, s, file);

  fclose(file);

  if (read_size != s) {
    free(tmp);
    NotifyError(engine, "failed to read encoder weights file");
    return NICE_FAILURE;
  }

  free(engine->encoder);

  engine->encoder = (float*)tmp;

  return NICE_SUCCESS;
}

NICE_Result
NICE_LoadDecoder(NICE_Engine* engine, const char* filename)
{
  const size_t s = sizeof(float) * NUM_DECODER_WEIGHTS;

  void* tmp = malloc(s);
  if (!tmp) {
    NotifyError(engine, "failed to allocate decoder weights");
    return NICE_FAILURE;
  }

  FILE* file = fopen(filename, "rb");
  if (!file) {
    free(tmp);
    NotifyError(engine, "failed to open decoder weights file");
    return NICE_FAILURE;
  }

  const size_t read_size = fread(tmp, 1, s, file);

  fclose(file);

  if (read_size != s) {
    free(tmp);
    NotifyError(engine, "failed to read decoder weights file");
    return NICE_FAILURE;
  }

  free(engine->decoder);

  engine->decoder = (float*)tmp;

  return NICE_SUCCESS;
}

static void
EncodeForward(const float* restrict x, float* restrict y, const float* restrict p)
{
  float z0[DIM];
  LINEAR_RESNET(x, z0, p);

  float z1[DIM];
  LINEAR_RESNET(z0, z1, p);

  float z2[DIM];
  LINEAR_RESNET(z1, z2, p);

  float z3[DIM];
  LINEAR_RESNET(z2, z3, p);

  float z4[LATENT_DIM];
  nice__mat_vec_mul_4x64(p, z3, z4);
  p += LATENT_DIM * DIM;

  float z5[LATENT_DIM];
  nice__softsign(z4, z5);

  for (int i = 0; i < LATENT_DIM; i++) {
    /* map [-1, 1] to [0, 1] */
    y[i] = z5[i] * 0.5F + 0.5F;
  }
}

static void
Quantize4(const float* restrict y, unsigned char* restrict b)
{
  for (int i = 0; i < 4; i++) {
    const int a0 = (int)(roundf(y[i] * 255.0F));
    const int a1 = (a0 < 0) ? 0 : a0;
    const int a2 = (a1 > 255) ? 255 : a1;
    b[i] = (unsigned char)(a2);
  }
}

void
NICE_Encode(const NICE_Engine* engine, const unsigned char* rgb, const int pitch, unsigned char* bits)
{
  float x[DIM];

  float y[LATENT_DIM];

  for (int c = 0; c < 3; ++c) {

    for (int i = 0; i < BLOCK_SIZE; ++i) {
      for (int j = 0; j < BLOCK_SIZE; ++j) {
        x[i * BLOCK_SIZE + j] = ((float)rgb[i * pitch + j * 3 + c]) * SCALE;
      }
    }

    EncodeForward(x, y, GetEncoder(engine));

    Quantize4(y, &bits[c * 4]);
  }
}

static void
DecodeForward(const float* restrict y, float* restrict x, const float* restrict p)
{
  float z0[DIM];
  nice__mat_vec_mul_64x4(p, y, z0);
  p += DIM * LATENT_DIM;

  float z1[DIM];
  LINEAR_RESNET(z0, z1, p);

  float z2[DIM];
  LINEAR_RESNET(z1, z2, p);

  float z3[DIM];
  LINEAR_RESNET(z2, z3, p);

  LINEAR_RESNET(z3, x, p);
}

static void
Dequantize4(const unsigned char* restrict b, float* restrict y)
{
  for (int i = 0; i < 4; i++) {
    y[i] = ((float)b[i]) * (1.0f / 255.0f);
  }
}

static unsigned char
ClampToU8(float v)
{
  const int v0 = (int)v;
  const int v1 = (v0 < 0) ? 0 : v0;
  const int v2 = (v1 > 255) ? 255 : v1;
  return (unsigned char)v2;
}

void
NICE_Decode(const NICE_Engine* engine, const unsigned char* restrict bits, int pitch, unsigned char* restrict rgb)
{
  float y[LATENT_DIM];
  float x[DIM];

  for (int c = 0; c < 3; ++c) {

    Dequantize4(&bits[c * 4], y);

    DecodeForward(y, x, GetDecoder(engine));

    for (int i = 0; i < BLOCK_SIZE; ++i) {

      for (int j = 0; j < BLOCK_SIZE; ++j) {

        const float xf = x[i * BLOCK_SIZE + j];

        const float pf = xf * (1.0F / SCALE);

        rgb[i * pitch + j * 3 + c] = ClampToU8(pf);
      }
    }
  }
}

void
NICE_EncodeTile(const NICE_Engine* engine, const unsigned char* rgb, int pitch, unsigned char* bits)
{
#pragma omp parallel for

  for (int i = 0; i < 100; i++) {

    const int bx = i % 10;
    const int by = i / 10;

    const unsigned char* block_rgb = rgb + (by * BLOCK_SIZE) * pitch + (bx * BLOCK_SIZE) * 3;

    unsigned char* block_bits = bits + i * (3 * LATENT_DIM);

    NICE_Encode(engine, block_rgb, pitch, block_bits);
  }
}

void
NICE_DecodeTile(const NICE_Engine* engine, const unsigned char* bits, int pitch, unsigned char* rgb)
{
#pragma omp parallel for

  for (int i = 0; i < 100; i++) {

    const int bx = i % 10;
    const int by = i / 10;

    unsigned char* block_rgb = rgb + (by * BLOCK_SIZE) * pitch + (bx * BLOCK_SIZE) * 3;

    const unsigned char* block_bits = bits + i * (3 * LATENT_DIM);

    NICE_Decode(engine, block_bits, pitch, block_rgb);
  }
}
