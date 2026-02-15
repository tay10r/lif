#include "NICE.h"

#include "NICE_Kernels.h"
#include "NICE_Weights.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LATENT_DIM 32

// TODO
#define NUM_ENCODER_WEIGHTS 1
#define NUM_DECODER_WEIGHTS 1

struct Tape
{
  float x[3 * NICE_BLOCK_SIZE * NICE_BLOCK_SIZE];

  /* 1: mat mul */
  float t0[LATENT_DIM];

  /* 2: mat mul */
  float t1[LATENT_DIM];

  /* 3: bias */
  float t2[LATENT_DIM];

  /* 4: activation */
  float t3[LATENT_DIM];

  /* 5: mat mul */
  float t4[LATENT_DIM];

  /* 6: activation */
  float z[LATENT_DIM];

  /* 7: mat mul */
  float t5[LATENT_DIM];

  /* 8: mat mul */
  float t6[LATENT_DIM];

  /* 9: bias */
  float t7[LATENT_DIM];

  /* 10: activation */
  float t8[LATENT_DIM];

  /* 11: mat mul */
  float y[3 * NICE_BLOCK_SIZE * NICE_BLOCK_SIZE];
};

struct Encoder
{
  float w0[3 * LATENT_DIM * NICE_BLOCK_SIZE * NICE_BLOCK_SIZE];

  float w1[LATENT_DIM * LATENT_DIM];

  float b0[LATENT_DIM];

  float w2[LATENT_DIM * LATENT_DIM];
};

struct Decoder
{
  float w3[LATENT_DIM * LATENT_DIM];

  float w4[LATENT_DIM * LATENT_DIM];

  float b1[LATENT_DIM];

  float w5[3 * LATENT_DIM * NICE_BLOCK_SIZE * NICE_BLOCK_SIZE];
};

struct NICE_Engine
{
  struct Encoder encoder;

  struct Decoder decoder;

  void* user_data;

  NICE_ErrorHandler error_handler;
};

static void
NotifyError(NICE_Engine* engine, const char* what)
{
  if (engine->error_handler) {
    engine->error_handler(engine->user_data, what);
  }
}

static float
frand_uniform(unsigned* state)
{
  *state = 1664525u * (*state) + 1013904223u;
  return (float)((*state >> 8) & 0xFFFFFF) / 16777216.0f; // [0,1)
}

static void
init_uniform(float* w, int n, float scale, unsigned* state)
{
  for (int i = 0; i < n; ++i) {
    float u = frand_uniform(state) * 2.0f - 1.0f; // [-1,1]
    w[i] = u * scale;
  }
}

static void
InitWeights(NICE_Engine* e, unsigned seed)
{
  unsigned st = seed ? seed : 1u;

  const int fan_in_w0 = 3 * NICE_BLOCK_SIZE * NICE_BLOCK_SIZE;
  const int fan_in_32 = LATENT_DIM;

  const float s0 = sqrtf(2.0f / (float)fan_in_w0);
  const float s1 = sqrtf(2.0f / (float)fan_in_32);

  init_uniform(e->encoder.w0, (int)(sizeof(e->encoder.w0) / sizeof(float)), s0, &st);
  init_uniform(e->encoder.w1, (int)(sizeof(e->encoder.w1) / sizeof(float)), s1, &st);
  init_uniform(e->encoder.w2, (int)(sizeof(e->encoder.w2) / sizeof(float)), s1, &st);

  init_uniform(e->decoder.w3, (int)(sizeof(e->decoder.w3) / sizeof(float)), s1, &st);
  init_uniform(e->decoder.w4, (int)(sizeof(e->decoder.w4) / sizeof(float)), s1, &st);
  init_uniform(e->decoder.w5, (int)(sizeof(e->decoder.w5) / sizeof(float)), s1, &st);

  memset(e->encoder.b0, 0, sizeof e->encoder.b0);
  memset(e->decoder.b1, 0, sizeof e->decoder.b1);
}

NICE_Engine*
NICE_NewEngine()
{
  NICE_Engine* self = malloc(sizeof(NICE_Engine));
  if (!self) {
    return NULL;
  }
  memset(self, 0, sizeof(NICE_Engine));
  InitWeights(self, 42);
  return self;
}

void
NICE_DestroyEngine(NICE_Engine* engine)
{
  free(engine);
}

void
NICE_SetErrorHandler(NICE_Engine* engine, void* user_data, NICE_ErrorHandler handler)
{
  engine->error_handler = handler;
  engine->user_data = user_data;
}

static void
EncoderForward(const struct Encoder* encoder, struct Tape* tape)
{
  NICE__MatVecMul_192x32(encoder->w0, tape->x, tape->t0);
  NICE__MatVecMul_32x32(encoder->w1, tape->t0, tape->t1);
  NICE__VecAdd(tape->t1, encoder->b0, tape->t2);
  NICE__LeakyReLU(tape->t2, tape->t3);
  NICE__MatVecMul_32x32(encoder->w2, tape->t3, tape->t4);
  NICE__SoftSign(tape->t4, tape->z);
}

static void
DecoderForward(const struct Decoder* decoder, struct Tape* tape)
{
  NICE__MatVecMul_32x32(decoder->w3, tape->z, tape->t5);
  NICE__MatVecMul_32x32(decoder->w4, tape->t5, tape->t6);
  NICE__VecAdd(tape->t6, decoder->b1, tape->t7);
  NICE__LeakyReLU(tape->t7, tape->t8);
  NICE__MatVecMul_32x192(decoder->w5, tape->t8, tape->y);
}

static void
InitTape(struct Tape* tape, const unsigned char* restrict rgb, const int pitch)
{
  for (int i = 0; i < (NICE_BLOCK_SIZE * NICE_BLOCK_SIZE); i++) {
    const int y = i / NICE_BLOCK_SIZE;
    const int x = i % NICE_BLOCK_SIZE;
    const int j = y * pitch + x * 3;
    tape->x[i + (NICE_BLOCK_SIZE * NICE_BLOCK_SIZE) * 0] = ((float)rgb[j + 0]) * (1.0F / 255.0F);
    tape->x[i + (NICE_BLOCK_SIZE * NICE_BLOCK_SIZE) * 1] = ((float)rgb[j + 1]) * (1.0F / 255.0F);
    tape->x[i + (NICE_BLOCK_SIZE * NICE_BLOCK_SIZE) * 2] = ((float)rgb[j + 2]) * (1.0F / 255.0F);
  }
}

void
NICE_Encode(const NICE_Engine* engine, const unsigned char* rgb, const int pitch, unsigned char* bits)
{
  struct Tape tape;

  InitTape(&tape, rgb, pitch);

  EncoderForward(&engine->encoder, &tape);

  const int num_bytes = LATENT_DIM / 8;

  for (int i = 0; i < num_bytes; i++) {

    unsigned char result = 0;

    const float* z = &tape.z[i * 8];

    for (int j = 0; j < 8; j++) {
      result |= (z[j] != 0.0F ? 1 : 0) << j;
    }

    bits[i] = result;
  }
}

inline static unsigned char
clamp_u8(const int x0)
{
  const int x1 = (x0 > 255) ? 255 : x0;
  const int x2 = (x1 < 0) ? 0 : x1;
  return (unsigned char)x2;
}

void
NICE_Decode(const NICE_Engine* engine, const unsigned char* restrict bits, const int pitch, unsigned char* restrict rgb)
{
  struct Tape tape;

  const int num_bytes = LATENT_DIM / 8;

  for (int i = 0; i < num_bytes; i++) {

    for (int j = 0; j < 8; j++) {
      tape.z[i * 8 + j] = ((1 << j) & bits[i]) ? 1.0F : 0.0F;
    }
  }

  DecoderForward(&engine->decoder, &tape);

#define K (NICE_BLOCK_SIZE * NICE_BLOCK_SIZE)

  for (int y = 0; y < NICE_BLOCK_SIZE; y++) {

    for (int x = 0; x < NICE_BLOCK_SIZE; x++) {

      const int r = (int)(tape.y[0 * K + y * NICE_BLOCK_SIZE + x] * 255.0F);
      const int g = (int)(tape.y[1 * K + y * NICE_BLOCK_SIZE + x] * 255.0F);
      const int b = (int)(tape.y[2 * K + y * NICE_BLOCK_SIZE + x] * 255.0F);

      rgb[(y * pitch + x * 3) + 0] = clamp_u8(r);
      rgb[(y * pitch + x * 3) + 1] = clamp_u8(g);
      rgb[(y * pitch + x * 3) + 2] = clamp_u8(b);
    }
  }
}

void
NICE_EncodeTile(const NICE_Engine* engine, const unsigned char* rgb, const int pitch, unsigned char* bits)
{
  int i = 0;

#pragma omp parallel for

  for (i = 0; i < 100; i++) {

    const int bx = i % 10;
    const int by = i / 10;

    const unsigned char* block_rgb = rgb + (by * NICE_BLOCK_SIZE) * pitch + (bx * NICE_BLOCK_SIZE) * 3;

    unsigned char* block_bits = bits + i * (3 * LATENT_DIM);

    NICE_Encode(engine, block_rgb, pitch, block_bits);
  }
}

void
NICE_DecodeTile(const NICE_Engine* engine, const unsigned char* bits, int pitch, unsigned char* rgb)
{
  int i = 0;

#pragma omp parallel for

  for (i = 0; i < 100; i++) {

    const int bx = i % 10;
    const int by = i / 10;

    unsigned char* block_rgb = rgb + (by * NICE_BLOCK_SIZE) * pitch + (bx * NICE_BLOCK_SIZE) * 3;

    const unsigned char* block_bits = bits + i * (3 * LATENT_DIM);

    NICE_Decode(engine, block_bits, pitch, block_rgb);
  }
}

float
NICE_TrainBlock(NICE_Engine* engine, const unsigned char* rgb, const int pitch, const float learning_rate)
{
  struct Tape tape;

  InitTape(&tape, rgb, pitch);

  EncoderForward(&engine->encoder, &tape);

  DecoderForward(&engine->decoder, &tape);

  const float loss = NICE__MSE(tape.y, tape.x);

  struct Tape grad;

  memset(&grad, 0, sizeof(grad));

  NICE__MSE_Grad(tape.y, tape.x, grad.y);

  float grad_w5[3 * LATENT_DIM * NICE_BLOCK_SIZE * NICE_BLOCK_SIZE] = { 0 };

  NICE__MatVecMul_32x192_Grad(engine->decoder.w5, tape.t8, grad.y, grad_w5, grad.t8);

  NICE__LeakyReLU_Grad(tape.t7, grad.t8, grad.t7);

  float grad_b1[LATENT_DIM] = { 0 };

  NICE__VecAdd_Grad(grad.t7, grad.t6, grad_b1);

  float grad_w4[LATENT_DIM * LATENT_DIM] = { 0 };

  NICE__MatVecMul_32x32_Grad(engine->decoder.w4, tape.t5, grad.t6, grad_w4, grad.t5);

  float grad_w3[LATENT_DIM * LATENT_DIM] = { 0 };

  NICE__MatVecMul_32x32_Grad(engine->decoder.w3, tape.z, grad.t5, grad_w3, grad.z);

  NICE__SoftSign_Grad(tape.t4, grad.z, grad.t4);

  float grad_w2[LATENT_DIM * LATENT_DIM] = { 0 };

  NICE__MatVecMul_32x32_Grad(engine->encoder.w2, tape.t3, grad.t4, grad_w2, grad.t3);

  NICE__LeakyReLU_Grad(tape.t2, grad.t3, grad.t2);

  float grad_b0[LATENT_DIM] = { 0 };

  NICE__VecAdd_Grad(grad.t2, grad.t1, grad_b0);

  float grad_w1[LATENT_DIM * LATENT_DIM] = { 0 };

  NICE__MatVecMul_32x32_Grad(engine->encoder.w1, tape.t0, grad.t1, grad_w1, grad.t0);

  float grad_w0[3 * LATENT_DIM * NICE_BLOCK_SIZE * NICE_BLOCK_SIZE] = { 0 };

  NICE__MatVecMul_192x32_Grad(engine->encoder.w0, tape.x, grad.t0, grad_w0, grad.x);

  const int n_w0 = 3 * LATENT_DIM * NICE_BLOCK_SIZE * NICE_BLOCK_SIZE;
  const int n_w1 = LATENT_DIM * LATENT_DIM;
  const int n_b0 = LATENT_DIM;
  const int n_w2 = LATENT_DIM * LATENT_DIM;

  const int n_w3 = LATENT_DIM * LATENT_DIM;
  const int n_w4 = LATENT_DIM * LATENT_DIM;
  const int n_b1 = LATENT_DIM;
  const int n_w5 = 3 * LATENT_DIM * NICE_BLOCK_SIZE * NICE_BLOCK_SIZE;

  for (int i = 0; i < n_w0; ++i)
    engine->encoder.w0[i] -= learning_rate * grad_w0[i];
  for (int i = 0; i < n_w1; ++i)
    engine->encoder.w1[i] -= learning_rate * grad_w1[i];
  for (int i = 0; i < n_b0; ++i)
    engine->encoder.b0[i] -= learning_rate * grad_b0[i];
  for (int i = 0; i < n_w2; ++i)
    engine->encoder.w2[i] -= learning_rate * grad_w2[i];

  for (int i = 0; i < n_w3; ++i)
    engine->decoder.w3[i] -= learning_rate * grad_w3[i];
  for (int i = 0; i < n_w4; ++i)
    engine->decoder.w4[i] -= learning_rate * grad_w4[i];
  for (int i = 0; i < n_b1; ++i)
    engine->decoder.b1[i] -= learning_rate * grad_b1[i];
  for (int i = 0; i < n_w5; ++i)
    engine->decoder.w5[i] -= learning_rate * grad_w5[i];

  return loss;
}
