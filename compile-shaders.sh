cd ../rust-gpu-cli-builder &&
cargo run --release -- ../transmission-renderer/shader --target spirv-unknown-spv1.3 --capabilities RuntimeDescriptorArray --extensions SPV_EXT_descriptor_indexing --multimodule --output ../transmission-renderer/compiled-shaders/normal &&
cargo run --release -- ../transmission-renderer/shader --target spirv-unknown-spv1.3 --capabilities RuntimeDescriptorArray RayQueryKHR --extensions SPV_EXT_descriptor_indexing SPV_KHR_ray_query --multimodule --output ../transmission-renderer/compiled-shaders/ray-tracing
