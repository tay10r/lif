#include <NICE.h>

#include <stdio.h>
#include <stdlib.h>

#include <stb_image.h>

namespace {

class Image final
{
public:
};

class Program final
{
public:
  ~Program() { NICE_DestroyEngine(engine_); }

  [[nodiscard]] auto run() -> bool
  {
    //
    return true;
  }

private:
  NICE_Engine* engine_{ NICE_NewEngine() };
};

} // namespace

int
main()
{
  Program program;

  const float learning_rate = 1.0e-3F;

  NICE_Engine* engine = NICE_NewEngine();
  if (!engine) {
    fprintf(stderr, "failed to create engine\n");
    return EXIT_FAILURE;
  }

  int w = 0;
  int h = 0;
  stbi_uc* train_data = stbi_load(TRAIN_DATASET, &w, &h, NULL, 3);
  if (!train_data) {
    fprintf(stderr, "failed to load training data\n");
    NICE_DestroyEngine(engine);
    return EXIT_FAILURE;
  }

  const int x_blocks = w / NICE_BLOCK_SIZE;
  const int y_blocks = h / NICE_BLOCK_SIZE;

  const int num_epochs = 100;

  for (int epoch = 0; epoch < num_epochs; epoch++) {

    float loss_sum = { 0.0F };

    for (int y = 0; y < y_blocks; y++) {

      for (int x = 0; x < x_blocks; x++) {

        const int pixel_x = x * NICE_BLOCK_SIZE;
        const int pixel_y = y * NICE_BLOCK_SIZE;
        const stbi_uc* block = train_data + (pixel_y * w + pixel_x) * 3;

        const float loss = NICE_TrainBlock(engine, block, w * 3, learning_rate);

        loss_sum += loss;
      }
    }

    const float loss_avg = loss_sum * (1.0F / ((float)x_blocks * y_blocks));

    printf("[%d]: %.06f\n", epoch, loss_avg);
  }

  stbi_image_free(train_data);

  NICE_DestroyEngine(engine);

  return EXIT_SUCCESS;
}
