cd ../rust-gpu-cli-builder
cargo run --release -- ../nice-grass/shader --spirv-metadata name-variables --target spirv-unknown-spv1.4 --capabilities RuntimeDescriptorArray Int64 --extensions SPV_EXT_descriptor_indexing --multimodule --output ../nice-grass/compiled-shaders/normal
cd ../nice-grass
./compile-rt-shaders.bat
