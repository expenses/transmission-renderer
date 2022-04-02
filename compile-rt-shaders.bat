cd ../rust-gpu-cli-builder
cargo run --release -- ../nice-grass/shader --spirv-metadata name-variables --target spirv-unknown-spv1.4 --capabilities RuntimeDescriptorArray Int64 RayQueryKHR GroupNonUniform GroupNonUniformBallot --extensions SPV_EXT_descriptor_indexing SPV_KHR_ray_query SPV_KHR_non_semantic_info --multimodule --output ../nice-grass/compiled-shaders/ray-tracing
