#pragma once

#define NICE_BLOCK_SIZE 8

#define NICE_TILE_SIZE 80

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * @brief Enumerates a success or failure result from certain API functions.
   * */
  enum NICE_Result
  {
    NICE_FAILURE = 0,
    NICE_SUCCESS = 1
  };

  typedef enum NICE_Result NICE_Result;

  /**
   * @brief The type of the function that can be passed error information to.
   * */
  typedef void (*NICE_ErrorHandler)(void* user_data, const char* what);

  /**
   * @brief The type definition for a compression or decompression engine.
   * */
  typedef struct NICE_Engine NICE_Engine;

  /**
   * @brief Creates a new engine instance.
   * */
  NICE_Engine* NICE_NewEngine();

  /**
   * @brief Releases the memory associated with an engine instance.
   * */
  void NICE_DestroyEngine(NICE_Engine* engine);

  /**
   * @brief Registers the callback to receive error information, when an error occurs.
   * */
  void NICE_SetErrorHandler(NICE_Engine* engine, void* user_data, NICE_ErrorHandler handler);

  /**
   * @brief Loads an encoder, from the file containing its trained weights.
   * */
  NICE_Result NICE_LoadEncoder(NICE_Engine* engine, const char* filename);

  /**
   * @brief Loads a decoder, from the file containing its trained weights.
   * */
  NICE_Result NICE_LoadDecoder(NICE_Engine* engine, const char* filename);

  /**
   * @brief Encodes a single 8x8 RGB block into a 12-byte buffer.
   * */
  void NICE_Encode(const NICE_Engine* engine, const unsigned char* rgb, int pitch, unsigned char* bits);

  /**
   * @brief Decodes a 12-byte buffer into a single 8x8 RGB block.
   * */
  void NICE_Decode(const NICE_Engine* engine, const unsigned char* bits, int pitch, unsigned char* rgb);

  /**
   * @brief Encodes an 80x80 RGB tile into a 1200-byte buffer.
   * */
  void NICE_EncodeTile(const NICE_Engine* engine, const unsigned char* rgb, int pitch, unsigned char* bits);

  /**
   * @brief Decodes an 80x80 RGB tile from a 1200-byte buffer.
   * */
  void NICE_DecodeTile(const NICE_Engine* engine, const unsigned char* bits, int pitch, unsigned char* rgb);

#ifdef __cplusplus
} /* extern "C" */
#endif
