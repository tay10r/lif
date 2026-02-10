NICE
====

### Neural Image Compression Engine

This is a multi-threaded, vectorized, fixed-bitrate image compression algorithm that uses a neural network to encode an
8x8 block to a 32-bit code (32-bit grayscale, 96-bit RGB). The compression ratio is a fixed 6.25%.

The big benefit of using this algorithm is that you can optionally fine tune it for a certain setting (such as outdoors,
low-lighting, a specific camera, etc) to get better quality for the same number of bits per pixel.
