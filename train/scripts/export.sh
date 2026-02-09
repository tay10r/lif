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

model=m0_small
#model=m0_medium
#model=m0_large
#model=m1_medium

$onnx2c -d batch_size:1 --no-globals --func-name linket_encoder_forward out/models/$model/encoder.onnx >../linket_encoder_forward.c
$onnx2c -d batch_size:1 --no-globals --func-name linket_decoder_forward out/models/$model/decoder.onnx >../linket_decoder_forward.c
