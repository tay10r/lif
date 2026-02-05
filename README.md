About
=====

This is an image compression algorithm that uses a neural network to encode 8x8 RGB blocks into latent space 32D vectors.
The compression ratio is `32/192`, which is about 17%. The quality is relatively good, comparable to a high quality JPEG
image. The neural network itself is split into an encoder and a decoder, each only a few hundred kilobytes in size. The
encoder and decoder are compiled into a dependency-free C library, making it easy to use in other projects.

The project is called `LIF`, which is an acronym for "latent image format". It is kind of a bad name though, because
I primarily made this for streaming video over UDP and have not yet given this compression algorithm a file format.
The name will remain `LIF` because eventually it will get a file format.

### Examples

Here are some examples of the algorithm.
The left is the original image, the right is the image after compression.

Baboon

<img width="240" height="240" alt="Image" src="https://github.com/user-attachments/assets/baab6cc0-7f80-4126-8663-fadbd587e377" />
<img width="240" height="240" alt="Image" src="https://github.com/user-attachments/assets/fe489dd4-884c-420a-980d-6b60a6d3327f" />

Lenna

<img width="240" height="240" alt="Image" src="https://github.com/user-attachments/assets/d5c45aad-27cb-4c89-82b9-e3558f120078" />
<img width="240" height="240" alt="Image" src="https://github.com/user-attachments/assets/bda50dcd-e0b5-44e1-a9e8-c3404edd8a95" />

### Project Structure

Right now it's a little adhoc. The code for training the network is in `src/`.
Once a network is trained, it is exported as an ONNX file and then converted to C with `onnx2c`, using `scripts/export.sh`.
The C library consists of the `.c` and `.h` files in the top level directory, which are dependency-free.
You can build them, along with any extra development code, with CMake.
