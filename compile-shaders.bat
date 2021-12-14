cd ../rust-gpu-cli-builder
cargo run --release -- ../nice-grass/shader --spirv-metadata name-variables --target spirv-unknown-spv1.3 --capabilities RuntimeDescriptorArray Int64 --extensions SPV_EXT_descriptor_indexing --multimodule --output ../nice-grass/compiled-shaders/normal
cargo run --release -- ../nice-grass/shader --spirv-metadata name-variables --target spirv-unknown-spv1.3 --capabilities RuntimeDescriptorArray Int64 RayQueryKHR --extensions SPV_EXT_descriptor_indexing SPV_KHR_ray_query --multimodule --output ../nice-grass/compiled-shaders/ray-tracing
