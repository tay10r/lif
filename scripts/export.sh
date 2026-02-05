#!/usr/bin/bash

if [ ! -e scripts/export.sh ]; then
  echo "run this from repo root"
  exit 1
fi

if [ ! -e onnx2c ]; then
  git clone https://github.com/tay10r/onnx2c.git
  pushd onnx2c
  git checkout feature/func-name
  git submodule update --init
  popd
fi

mkdir onnx2c/build

cmake -B onnx2c/build -S onnx2c

cmake --build onnx2c/build --target onnx2c

onnx2c=onnx2c/build/onnx2c

$onnx2c --func-name lif_encoder_forward out/encoder.onnx >lif_encoder_forward.c
$onnx2c --func-name lif_decoder_forward out/decoder.onnx >lif_decoder_forward.c
