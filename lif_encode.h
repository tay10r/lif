#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

  typedef void (*lif_tile_encode_callback)(void* user_data, int x, int y, const unsigned char* tile_data);

  void lif_encode(const unsigned char* rgb, int w, int h, void* user_data, lif_tile_encode_callback cb);

#ifdef __cplusplus
} /* extern "C" */
#endif
