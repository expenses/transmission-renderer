#![allow(clippy::float_cmp)]

use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{
    AccelerationStructure as AccelerationStructureLoader,
    DeferredHostOperations as DeferredHostOperationsLoader, Surface as SurfaceLoader,
    Swapchain as SwapchainLoader,
};
use ash::vk;
use ash_abstractions::CStrList;
use std::f32::consts::PI;
use std::ffi::CStr;
use structopt::StructOpt;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Fullscreen;

use glam::{Mat4, Quat, UVec2, Vec2, Vec3, Vec3Swizzles, Vec4};
use shared_structs::{Instance, Light, PushConstants, Similarity};

mod acceleration_structures;
mod descriptor_sets;
mod model_loading;
mod pipelines;
mod profiling;
mod render_passes;

use acceleration_structures::{
    build_acceleration_structures_from_primitives,
    build_top_level_acceleration_structure_from_instances,
};
use descriptor_sets::{DescriptorSetLayouts, DescriptorSets};
use model_loading::{load_gltf, ImageManager};
use pipelines::Pipelines;
use profiling::ProfilingContext;
use render_passes::RenderPasses;

fn perspective_matrix_reversed(width: u32, height: u32) -> Mat4 {
    let aspect_ratio = width as f32 / height as f32;
    let vertical_fov = 59.0_f32.to_radians();

    let focal_length = 1.0 / (vertical_fov / 2.0).tan();

    let a = Z_NEAR / (Z_FAR - Z_NEAR);
    let b = Z_FAR * a;

    Mat4::from_cols(
        Vec4::new(focal_length / aspect_ratio, 0.0, 0.0, 0.0),
        Vec4::new(0.0, -focal_length, 0.0, 0.0),
        Vec4::new(0.0, 0.0, a, -1.0),
        Vec4::new(0.0, 0.0, b, 0.0),
    )
}

pub const Z_NEAR: f32 = 0.01;
pub const Z_FAR: f32 = 500.0;

pub const MAX_IMAGES: u32 = 193;
pub const NUM_CLUSTERS_X: u32 = 12;
pub const NUM_CLUSTERS_Y: u32 = 8;
pub const NUM_DEPTH_SLICES: u32 = 16;
pub const NUM_CLUSTERS: u32 = NUM_CLUSTERS_X * NUM_CLUSTERS_Y * NUM_DEPTH_SLICES;

#[derive(StructOpt)]
struct Opt {
    /// The name of the model inside the glTF-Sample-Models directory to render.
    gltf_sample_model_name: String,
    /// A scale factor to be applied to the model.
    #[structopt(short, long, default_value = "1.0")]
    scale: f32,
    /// Override the default roughness factor of the model.
    /// Doesn't effect models that use a texture for roughness.
    #[structopt(long)]
    roughness_override: Option<f32>,
    /// Log allocator leaks on shutdown. Off by default because it makes panics hard to debug.
    #[structopt(long)]
    log_leaks: bool,
    /// Render a model external to the glTF-Sample-Models directory, in which case the full path needs to be specified.
    #[structopt(long)]
    external_model: bool,
    ///
    #[structopt(short, long)]
    ray_tracing: bool,
}

fn main() -> anyhow::Result<()> {
    let entire_setup_span = tracy_client::span!("Entire Setup");

    let opt = Opt::from_args();

    {
        use simplelog::*;

        CombinedLogger::init(vec![TermLogger::new(
            LevelFilter::Info,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        )])?;
    }

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Transmission Renderer")
        .build(&event_loop)?;

    let entry = unsafe { ash::Entry::new() }?;

    let api_version = vk::API_VERSION_1_2;

    let app_info = vk::ApplicationInfo::builder()
        .application_name(c_str_macro::c_str!("Transmission Renderer"))
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(api_version);

    let instance_extensions = CStrList::new({
        let mut instance_extensions = ash_window::enumerate_required_extensions(&window)?;
        instance_extensions.extend(&[DebugUtilsLoader::name()]);
        instance_extensions
    });

    let enabled_layers = CStrList::new(vec![CStr::from_bytes_with_nul(
        b"VK_LAYER_KHRONOS_validation\0",
    )?]);

    let mut extensions = vec![SwapchainLoader::name()];

    if opt.ray_tracing {
        extensions.extend_from_slice(&[
            DeferredHostOperationsLoader::name(),
            AccelerationStructureLoader::name(),
            vk::KhrRayQueryFn::name(),
            vk::KhrShaderNonSemanticInfoFn::name(),
        ]);
    }

    let device_extensions = CStrList::new(extensions);

    let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(ash_abstractions::vulkan_debug_utils_callback));

    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(instance_extensions.pointers())
                .enabled_layer_names(enabled_layers.pointers())
                .push_next(&mut debug_messenger_info),
            None,
        )
    }?;

    let debug_utils_loader = DebugUtilsLoader::new(&entry, &instance);

    let surface_loader = SurfaceLoader::new(&entry, &instance);

    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }?;

    let (physical_device, graphics_queue_family, surface_format) =
        match ash_abstractions::select_physical_device(
            &instance,
            &device_extensions,
            &surface_loader,
            surface,
            vk::Format::B8G8R8A8_SRGB,
        )? {
            Some(selection) => selection,
            None => {
                log::info!("No suitable device found 💔. Exiting program");
                return Ok(());
            }
        };

    let physical_device_properties =
        unsafe { instance.get_physical_device_properties(physical_device) };

    let surface_caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }?;

    let device = {
        let queue_info = [*vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(graphics_queue_family)
            .queue_priorities(&[1.0])];

        let device_features = vk::PhysicalDeviceFeatures::builder().shader_int64(true);

        let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::builder()
            .runtime_descriptor_array(true)
            .draw_indirect_count(true)
            .descriptor_binding_partially_bound(true)
            .buffer_device_address(true);

        let mut ray_query_features =
            vk::PhysicalDeviceRayQueryFeaturesKHR::builder().ray_query(true);

        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true);

        let mut device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_features(&device_features)
            .enabled_extension_names(device_extensions.pointers())
            .enabled_layer_names(enabled_layers.pointers())
            .push_next(&mut vulkan_1_2_features);

        if opt.ray_tracing {
            device_info = device_info
                .push_next(&mut ray_query_features)
                .push_next(&mut acceleration_structure_features);
        }

        unsafe { instance.create_device(physical_device, &device_info, None) }?
    };

    let render_passes = RenderPasses::new(&device, &debug_utils_loader, surface_format.format)?;
    let descriptor_set_layouts = DescriptorSetLayouts::new(&device)?;

    let pipeline_cache =
        unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }?;

    let pipelines = Pipelines::new(
        &device,
        &render_passes,
        &descriptor_set_layouts,
        pipeline_cache,
        opt.ray_tracing,
    )?;

    let descriptor_sets = DescriptorSets::allocate(&device, &descriptor_set_layouts)?;

    let queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

    ash_abstractions::set_object_name(&device, &debug_utils_loader, queue, "graphics queue")?;

    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::builder().queue_family_index(graphics_queue_family),
            None,
        )
    }?;

    let command_buffers = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }?;

    let command_buffer = command_buffers[0];

    let mut allocator =
        gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings {
                log_leaks_on_shutdown: opt.log_leaks,
                ..Default::default()
            },
            buffer_device_address: true,
        })?;

    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
    }?;

    let mut init_resources = ash_abstractions::InitResources {
        command_buffer,
        device: &device,
        allocator: &mut allocator,
        debug_utils_loader: Some(&debug_utils_loader),
    };

    let mut image_manager = ImageManager::default();
    let mut buffers_to_cleanup = Vec::new();

    let ggx_lut_id = {
        let _span = tracy_client::span!("Loading ggx_lut.png");

        use image::GenericImageView;

        let decoded_image = image::load_from_memory_with_format(
            include_bytes!("../ggx_lut.png"),
            image::ImageFormat::Png,
        )?;

        let rgba_image = decoded_image.to_rgba8();

        let (image, staging_buffer) = ash_abstractions::load_image_from_bytes(
            &ash_abstractions::LoadImageDescriptor {
                bytes: &*rgba_image,
                extent: vk::Extent3D {
                    width: decoded_image.width(),
                    height: decoded_image.height(),
                    depth: 1,
                },
                view_ty: vk::ImageViewType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                name: "ggx lut",
                next_accesses: &[
                    vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_layout: vk_sync::ImageLayout::Optimal,
                mip_levels: 1,
            },
            &mut init_resources,
        )?;

        buffers_to_cleanup.push(staging_buffer);

        image_manager.push_image(image)
    };

    let window_size = window.inner_size();

    let mut extent = vk::Extent2D {
        width: window_size.width,
        height: window_size.height,
    };

    let mut model_staging_buffers = ModelStagingBuffers::default();
    let mut max_draw_counts = MaxDrawCounts::default();

    load_gltf(
        &model_loading::path_for_gltf_model("Sponza"),
        &mut init_resources,
        &mut image_manager,
        &mut buffers_to_cleanup,
        &mut model_staging_buffers,
        &mut max_draw_counts,
        Similarity::IDENTITY,
        None,
    )?;

    load_gltf(
        &if opt.external_model {
            std::path::PathBuf::from(&opt.gltf_sample_model_name)
        } else {
            model_loading::path_for_gltf_model(&opt.gltf_sample_model_name)
        },
        &mut init_resources,
        &mut image_manager,
        &mut buffers_to_cleanup,
        &mut model_staging_buffers,
        &mut max_draw_counts,
        Similarity {
            translation: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: opt.scale,
        },
        opt.roughness_override,
    )?;

    dbg!(max_draw_counts);

    let num_instances = model_staging_buffers.instances.len() as u32;
    let num_primitives = model_staging_buffers.primitives.len() as u32;

    let model_buffers = model_staging_buffers.upload(&mut init_resources)?;

    // todo: reduce this it model.num_opaque + model2.num_opaque

    let draw_buffers = DrawBuffers::new(max_draw_counts, &mut init_resources)?;

    let mut depthbuffer = create_depthbuffer(extent.width, extent.height, &mut init_resources)?;
    let mut hdr_framebuffer = create_hdr_framebuffer(
        extent.width,
        extent.height,
        1,
        "hdr framebuffer",
        vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE,
        &mut init_resources,
    )?;

    let mut opaque_mip_levels = mip_levels_for_size(extent.width, extent.height);

    let mut opaque_sampled_hdr_framebuffer = create_hdr_framebuffer(
        extent.width,
        extent.height,
        opaque_mip_levels,
        "opaque sampled hdr framebuffer",
        vk::ImageUsageFlags::TRANSFER_DST,
        &mut init_resources,
    )?;

    let basic_subresource_range = *vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    let mut opaque_sampled_hdr_framebuffer_top_mip_view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .image(opaque_sampled_hdr_framebuffer.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .subresource_range(basic_subresource_range),
            None,
        )
    }?;

    // We need to transition the mips of the image because they copy the mip 0 layout.
    // todo: do this in a nicer way.
    {
        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(1)
            .level_count(opaque_mip_levels - 1)
            .layer_count(1);

        vk_sync::cmd::pipeline_barrier(
            init_resources.device,
            init_resources.command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                next_accesses: &[vk_sync::AccessType::TransferWrite],
                next_layout: vk_sync::ImageLayout::Optimal,
                image: opaque_sampled_hdr_framebuffer.image,
                range: subresource_range,
                discard_contents: true,
                ..Default::default()
            }],
        );
    }

    dbg!(NUM_CLUSTERS);

    let lights = [
        Light::new_point(Vec3::new(0.0, 0.8, 0.0), Vec3::X, 5.0),
        Light::new_point(Vec3::new(8.0, 0.8, 0.0), Vec3::Y, 10.0),
        Light::new_spot(
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(1.0, 1.0, 0.5),
            50.0,
            Vec3::Z,
            0.7,
            0.8,
        ),
        Light::new_spot(
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(1.0, 1.0, 0.5),
            50.0,
            -Vec3::Z,
            0.7,
            0.8,
        ),
    ];

    let light_buffers = LightBuffers {
        lights: ash_abstractions::Buffer::new(
            unsafe { cast_slice(&lights) },
            "lights",
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &mut init_resources,
        )?,
        cluster_light_counts: ash_abstractions::Buffer::new_of_size(
            4 * NUM_CLUSTERS as u64,
            "cluster light counts",
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            &mut init_resources,
        )?,
        cluster_light_indices: ash_abstractions::Buffer::new_of_size(
            4 * NUM_CLUSTERS as u64 * shared_structs::MAX_LIGHTS_PER_CLUSTER as u64,
            "cluster light indices",
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &mut init_resources,
        )?,
    };

    let cluster_data_buffer = ash_abstractions::Buffer::new_of_size(
        NUM_CLUSTERS as u64 * std::mem::size_of::<shared_structs::ClusterAabb>() as u64,
        "cluster data buffer",
        vk::BufferUsageFlags::STORAGE_BUFFER,
        &mut init_resources,
    )?;

    let tonemapping_params = colstodian::tonemap::BakedLottesTonemapperParams::from(
        colstodian::tonemap::LottesTonemapperParams {
            ..Default::default()
        },
    );

    let query_pool = profiling::QueryPool::new(&mut init_resources)?;

    let mut camera = dolly::rig::CameraRig::builder()
        .with(dolly::drivers::Position::new(Vec3::new(0.0, 3.0, 1.0)))
        .with(dolly::drivers::YawPitch::new().pitch_degrees(-15.0))
        .with(dolly::drivers::Smooth::new_position_rotation(0.5, 0.25))
        .build();

    let mut view_matrix = Mat4::look_at_rh(
        camera.final_transform.position,
        camera.final_transform.position + camera.final_transform.forward(),
        camera.final_transform.up(),
    );

    let mut perspective_matrix = perspective_matrix_reversed(extent.width, extent.height);

    let num_clusters = UVec2::new(NUM_CLUSTERS_X, NUM_CLUSTERS_Y);

    let mut sun_velocity = Vec2::ZERO;
    let mut sun = Sun {
        pitch: 1.1,
        yaw: 4.8,
    };

    let mut uniforms = shared_structs::Uniforms {
        sun_dir: sun.as_normal().into(),
        sun_intensity: Vec3::splat(3.0).into(),
        ggx_lut_texture_index: ggx_lut_id,
        num_clusters,
        cluster_size_in_pixels: Vec2::new(extent.width as f32, extent.height as f32)
            / num_clusters.as_vec2(),
        debug_clusters: 0,
        // todo: these values are a bit nonsense as I based it off the filament implementation:
        // https://google.github.io/filament/Filament.md.html#imagingpipeline/lightpath/clusteredforwardrendering
        // which uses opengl and a depth range of -1 to 1.
        light_clustering_coefficients: shared_structs::LightClusterCoefficients::new(
            Z_NEAR,
            Z_FAR,
            NUM_DEPTH_SLICES,
        ),
    };

    dbg!(&uniforms);

    let mut uniforms_buffer = ash_abstractions::Buffer::new(
        unsafe { bytes_of(&uniforms) },
        "uniforms buffer",
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        &mut init_resources,
    )?;

    let mut acceleration_structure_debugging_uniforms =
        shared_structs::AccelerationStructureDebuggingUniforms {
            proj_inverse: Mat4::IDENTITY,
            view_inverse: Mat4::IDENTITY,
            size: UVec2::ZERO,
        };

    let mut acceleration_structure_debugging_uniforms_buffer = ash_abstractions::Buffer::new(
        unsafe { bytes_of(&acceleration_structure_debugging_uniforms) },
        "acceleration structure debugging uniforms buffer",
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        &mut init_resources,
    )?;

    let (top_level_acceleration_structure, acceleration_structures) = if opt.ray_tracing {
        let acceleration_structure_loader = AccelerationStructureLoader::new(&instance, &device);

        let acceleration_structure_properties = {
            let mut acceleration_structure_properties =
                vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();

            let mut device_properties_2 = vk::PhysicalDeviceProperties2::builder()
                .push_next(&mut acceleration_structure_properties);

            unsafe {
                instance.get_physical_device_properties2(physical_device, &mut device_properties_2)
            }

            acceleration_structure_properties
        };

        let acceleration_structures = build_acceleration_structures_from_primitives(
            &model_buffers,
            &model_staging_buffers,
            &acceleration_structure_properties,
            &acceleration_structure_loader,
            &mut init_resources,
            &mut buffers_to_cleanup,
        )?;

        vk_sync::cmd::pipeline_barrier(
            &device,
            init_resources.command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::AccelerationStructureBuildWrite],
                next_accesses: &[vk_sync::AccessType::AccelerationStructureBuildRead],
            }),
            &[],
            &[],
        );

        let acceleration_structure_instances = model_staging_buffers
            .instances
            .iter()
            .filter(|instance| {
                let primitive = &model_staging_buffers.primitives[instance.primitive_id as usize];
                primitive.draw_buffer_index < 2
            })
            .map(|instance| vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: transpose_matrix_for_acceleration_structure_instance(
                        instance.transform.unpack().as_mat4(),
                    ),
                },
                instance_custom_index_and_mask: Packed24_8::new(0, 0xff).0,
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0).0,
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: acceleration_structures[instance.primitive_id as usize]
                        .buffer
                        .device_address(&device),
                },
            })
            .collect::<Vec<_>>();

        let acceleration_structure_instances_buffer = ash_abstractions::Buffer::new_with_alignment(
            unsafe { cast_slice(&acceleration_structure_instances) },
            "acceleration structure instances",
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            // Must be aligned to 16 bytes:
            // https://vulkan.lunarg.com/doc/view/1.2.198.0/windows/1.2-extensions/vkspec.html#VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03715
            16,
            &mut init_resources,
        )?;

        let top_level_acceleration_structure =
            build_top_level_acceleration_structure_from_instances(
                &acceleration_structure_instances_buffer,
                acceleration_structure_instances.len() as u32,
                &acceleration_structure_properties,
                &acceleration_structure_loader,
                &mut init_resources,
                &mut buffers_to_cleanup,
            )?;

        buffers_to_cleanup.push(acceleration_structure_instances_buffer);

        (
            Some(top_level_acceleration_structure),
            acceleration_structures,
        )
    } else {
        (None, Vec::new())
    };

    let mut draw_framebuffer = create_draw_framebuffer(
        &device,
        extent,
        render_passes.draw,
        &hdr_framebuffer,
        opaque_sampled_hdr_framebuffer_top_mip_view,
        &depthbuffer,
    )?;

    let mut transmission_framebuffer = create_transmission_framebuffer(
        &device,
        extent,
        render_passes.transmission,
        &hdr_framebuffer,
        &depthbuffer,
    )?;

    let mut keyboard_state = KeyboardState::default();

    let sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )
    }?;

    let clamp_sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )
    }?;

    fn buffer_info(buffer: &ash_abstractions::Buffer) -> [vk::DescriptorBufferInfo; 1] {
        [vk::DescriptorBufferInfo {
            buffer: buffer.buffer,
            range: vk::WHOLE_SIZE,
            offset: 0,
        }]
    }

    unsafe {
        device.update_descriptor_sets(
            &[
                *image_manager.write_descriptor_set(descriptor_sets.main, 0),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.main)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[*vk::DescriptorImageInfo::builder().sampler(sampler)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.main)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&model_buffers.materials)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.main)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info(&uniforms_buffer)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.main)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[*vk::DescriptorImageInfo::builder().sampler(clamp_sampler)]),
                // Instance buffer
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.instance_buffer)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&model_buffers.instances)),
                // Frustum culling
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.frustum_culling)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&model_buffers.primitives)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.frustum_culling)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&draw_buffers.instance_count_buffer)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.frustum_culling)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&draw_buffers.draw_counts_buffer)),
                // frustum buffers
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.frustum_culling)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&draw_buffers.opaque.buffer)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.frustum_culling)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&draw_buffers.alpha_clip.buffer)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.frustum_culling)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&draw_buffers.transmission.buffer)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.frustum_culling)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&draw_buffers.transmission_alpha_clip.buffer)),
                // lights
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.lights)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&light_buffers.lights)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.lights)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&light_buffers.cluster_light_counts)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.lights)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&light_buffers.cluster_light_indices)),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.acceleration_structure_debugging)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info(
                        &acceleration_structure_debugging_uniforms_buffer,
                    )),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets.cluster_data)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info(&cluster_data_buffer)),
            ],
            &[],
        )
    }

    descriptor_sets.update_framebuffers(&device, &hdr_framebuffer, &opaque_sampled_hdr_framebuffer);

    unsafe {
        record_write_cluster_data(
            &init_resources,
            &pipelines,
            &descriptor_sets,
            perspective_matrix,
            UVec2::new(extent.width, extent.height),
        );
    }

    {
        let _span = tracy_client::span!("Flushing init resources");
        end_init_resources(init_resources, queue)?;
    }

    for buffer in buffers_to_cleanup.drain(..) {
        buffer.cleanup_and_drop(&device, &mut allocator)?;
    }

    let mut push_constants = shared_structs::PushConstants {
        // Updated every frame.
        proj_view: Default::default(),
        view_position: Default::default(),
        framebuffer_size: UVec2::new(extent.width, extent.height),
        acceleration_structure_address: top_level_acceleration_structure
            .as_ref()
            .map(|a| a.buffer.device_address(&device))
            .unwrap_or(0),
    };

    // Swapchain

    let mut image_count = (surface_caps.min_image_count + 1).max(3);
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    log::info!("Using {} swapchain images at a time.", image_count);

    let swapchain_loader = SwapchainLoader::new(&instance, &device);

    let mut swapchain_info = *vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let mut swapchain =
        ash_abstractions::Swapchain::new(&device, &swapchain_loader, swapchain_info)?;

    let mut swapchain_image_framebuffers =
        create_swapchain_image_framebuffers(&device, extent, &swapchain, &render_passes)?;

    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    let create_named_semaphore = |name| {
        let semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }?;

        ash_abstractions::set_object_name(&device, &debug_utils_loader, semaphore, name)?;

        Ok::<_, anyhow::Error>(semaphore)
    };

    let present_semaphore = create_named_semaphore("present semaphore")?;
    let render_semaphore = create_named_semaphore("render semaphore")?;
    let render_fence = unsafe { device.create_fence(&fence_info, None)? };

    let mut cursor_grab = false;

    let mut screen_center =
        winit::dpi::LogicalPosition::new(extent.width as f64 / 2.0, extent.height as f64 / 2.0);

    let mut profiling_ctx =
        query_pool.into_profiling_context(&device, physical_device_properties.limits)?;

    drop(entire_setup_span);

    let mut toggle = false;

    event_loop.run(move |event, _, control_flow| {
        let loop_closure = || -> anyhow::Result<()> {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(key),
                                ..
                            },
                        ..
                    } => {
                        let is_pressed = state == ElementState::Pressed;

                        match key {
                            VirtualKeyCode::W => keyboard_state.forwards = is_pressed,
                            VirtualKeyCode::S => keyboard_state.backwards = is_pressed,
                            VirtualKeyCode::A => keyboard_state.left = is_pressed,
                            VirtualKeyCode::D => keyboard_state.right = is_pressed,
                            VirtualKeyCode::Up => keyboard_state.sun_up = is_pressed,
                            VirtualKeyCode::Right => keyboard_state.sun_cw = is_pressed,
                            VirtualKeyCode::Left => keyboard_state.sun_ccw = is_pressed,
                            VirtualKeyCode::Down => keyboard_state.sun_down = is_pressed,
                            VirtualKeyCode::F11 => {
                                if is_pressed {
                                    if window.fullscreen().is_some() {
                                        window.set_fullscreen(None);
                                    } else {
                                        window.set_fullscreen(Some(Fullscreen::Borderless(None)))
                                    }
                                }
                            }
                            VirtualKeyCode::F => {
                                if is_pressed {
                                    uniforms.debug_clusters = 1 - uniforms.debug_clusters;
                                }
                            }
                            VirtualKeyCode::G => {
                                if is_pressed {
                                    cursor_grab = !cursor_grab;

                                    if cursor_grab {
                                        window.set_cursor_position(screen_center)?;
                                    }

                                    window.set_cursor_visible(!cursor_grab);
                                    window.set_cursor_grab(cursor_grab)?;
                                }
                            }
                            VirtualKeyCode::T if is_pressed => {
                                toggle = !toggle;
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if cursor_grab {
                            let position = position.to_logical::<f64>(window.scale_factor());

                            window.set_cursor_position(screen_center)?;

                            camera
                                .driver_mut::<dolly::drivers::YawPitch>()
                                .rotate_yaw_pitch(
                                    0.1 * (screen_center.x - position.x) as f32,
                                    0.1 * (screen_center.y - position.y) as f32,
                                );
                        }
                    }
                    WindowEvent::Resized(new_size) => {
                        extent.width = new_size.width;
                        extent.height = new_size.height;

                        push_constants.framebuffer_size = UVec2::new(extent.width, extent.height);

                        uniforms.cluster_size_in_pixels =
                            Vec2::new(extent.width as f32, extent.height as f32)
                                / num_clusters.as_vec2();

                        uniforms_buffer.write_mapped(unsafe { bytes_of(&uniforms) }, 0)?;

                        perspective_matrix =
                            perspective_matrix_reversed(extent.width, extent.height);

                        screen_center = winit::dpi::LogicalPosition::new(
                            extent.width as f64 / 2.0,
                            extent.height as f64 / 2.0,
                        );

                        swapchain_info.image_extent = extent;
                        swapchain_info.old_swapchain = swapchain.swapchain;

                        unsafe {
                            device.queue_wait_idle(queue)?;
                        }

                        swapchain = ash_abstractions::Swapchain::new(
                            &device,
                            &swapchain_loader,
                            swapchain_info,
                        )?;

                        unsafe {
                            device.reset_command_pool(
                                command_pool,
                                vk::CommandPoolResetFlags::empty(),
                            )?;

                            device.begin_command_buffer(
                                command_buffer,
                                &vk::CommandBufferBeginInfo::builder()
                                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                            )?;
                        }

                        depthbuffer.cleanup(&device, &mut allocator)?;
                        hdr_framebuffer.cleanup(&device, &mut allocator)?;
                        opaque_sampled_hdr_framebuffer.cleanup(&device, &mut allocator)?;

                        let mut init_resources = ash_abstractions::InitResources {
                            command_buffer,
                            device: &device,
                            allocator: &mut allocator,
                            debug_utils_loader: Some(&debug_utils_loader),
                        };

                        depthbuffer =
                            create_depthbuffer(extent.width, extent.height, &mut init_resources)?;

                        hdr_framebuffer = create_hdr_framebuffer(
                            extent.width,
                            extent.height,
                            1,
                            "hdr framebuffer",
                            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE,
                            &mut init_resources,
                        )?;

                        opaque_mip_levels = mip_levels_for_size(extent.width, extent.height);

                        opaque_sampled_hdr_framebuffer = create_hdr_framebuffer(
                            extent.width,
                            extent.height,
                            opaque_mip_levels,
                            "opaque sampled hdr framebuffer",
                            vk::ImageUsageFlags::TRANSFER_DST,
                            &mut init_resources,
                        )?;

                        opaque_sampled_hdr_framebuffer_top_mip_view = unsafe {
                            device.create_image_view(
                                &vk::ImageViewCreateInfo::builder()
                                    .image(opaque_sampled_hdr_framebuffer.image)
                                    .view_type(vk::ImageViewType::TYPE_2D)
                                    .format(vk::Format::R16G16B16A16_SFLOAT)
                                    .subresource_range(basic_subresource_range),
                                None,
                            )
                        }?;

                        // We need to transition the mips of the image because they copy the mip 0 layout.
                        // todo: do this in a nicer way.
                        {
                            let subresource_range = *vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(1)
                                .level_count(opaque_mip_levels - 1)
                                .layer_count(1);

                            vk_sync::cmd::pipeline_barrier(
                                init_resources.device,
                                init_resources.command_buffer,
                                None,
                                &[],
                                &[vk_sync::ImageBarrier {
                                    previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                                    next_accesses: &[vk_sync::AccessType::TransferWrite],
                                    next_layout: vk_sync::ImageLayout::Optimal,
                                    image: opaque_sampled_hdr_framebuffer.image,
                                    range: subresource_range,
                                    discard_contents: true,
                                    ..Default::default()
                                }],
                            );
                        }

                        unsafe {
                            record_write_cluster_data(
                                &init_resources,
                                &pipelines,
                                &descriptor_sets,
                                perspective_matrix,
                                UVec2::new(extent.width, extent.height),
                            );
                        }

                        drop(init_resources);

                        unsafe {
                            device.end_command_buffer(command_buffer)?;
                            let fence =
                                device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

                            device.queue_submit(
                                queue,
                                &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
                                fence,
                            )?;

                            device.wait_for_fences(&[fence], true, u64::MAX)?;
                        }

                        descriptor_sets.update_framebuffers(
                            &device,
                            &hdr_framebuffer,
                            &opaque_sampled_hdr_framebuffer,
                        );

                        swapchain_image_framebuffers = create_swapchain_image_framebuffers(
                            &device,
                            extent,
                            &swapchain,
                            &render_passes,
                        )?;

                        draw_framebuffer = create_draw_framebuffer(
                            &device,
                            extent,
                            render_passes.draw,
                            &hdr_framebuffer,
                            opaque_sampled_hdr_framebuffer_top_mip_view,
                            &depthbuffer,
                        )?;
                        transmission_framebuffer = create_transmission_framebuffer(
                            &device,
                            extent,
                            render_passes.transmission,
                            &hdr_framebuffer,
                            &depthbuffer,
                        )?;
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    let delta_time = 1.0 / 60.0;

                    let forwards = keyboard_state.forwards as i32 - keyboard_state.backwards as i32;
                    let right = keyboard_state.right as i32 - keyboard_state.left as i32;

                    let move_vec = camera.final_transform.rotation
                        * Vec3::new(right as f32, 0.0, -forwards as f32).clamp_length_max(1.0);

                    let speed = 3.0;
                    //let speed = 100.0;

                    camera
                        .driver_mut::<dolly::drivers::Position>()
                        .translate(move_vec * delta_time * speed);

                    camera.update(delta_time);

                    push_constants.proj_view = {
                        view_matrix = Mat4::look_at_rh(
                            camera.final_transform.position,
                            camera.final_transform.position + camera.final_transform.forward(),
                            camera.final_transform.up(),
                        );
                        perspective_matrix * view_matrix
                    };
                    push_constants.view_position = camera.final_transform.position.into();

                    {
                        let acceleration = 0.002;
                        let max_velocity = 0.05;

                        if keyboard_state.sun_up {
                            sun_velocity.y += acceleration;
                        }

                        if keyboard_state.sun_down {
                            sun_velocity.y -= acceleration;
                        }

                        if keyboard_state.sun_cw {
                            sun_velocity.x += acceleration;
                        }

                        if keyboard_state.sun_ccw {
                            sun_velocity.x -= acceleration;
                        }

                        let magnitude = sun_velocity.length();
                        if magnitude > max_velocity {
                            let clamped_magnitude = magnitude.min(max_velocity);
                            sun_velocity *= clamped_magnitude / magnitude;
                        }

                        sun.yaw -= sun_velocity.x;
                        sun.pitch = (sun.pitch + sun_velocity.y).min(PI / 2.0).max(0.0);

                        sun_velocity *= 0.95;
                    }

                    uniforms.sun_dir = sun.as_normal().into();
                    uniforms_buffer.write_mapped(unsafe { bytes_of(&uniforms) }, 0)?;

                    acceleration_structure_debugging_uniforms.view_inverse = view_matrix.inverse();
                    acceleration_structure_debugging_uniforms.proj_inverse =
                        perspective_matrix.inverse();
                    acceleration_structure_debugging_uniforms.size =
                        UVec2::new(extent.width, extent.height);
                    acceleration_structure_debugging_uniforms_buffer.write_mapped(
                        unsafe { bytes_of(&acceleration_structure_debugging_uniforms) },
                        0,
                    )?;

                    window.request_redraw();
                }
                Event::RedrawRequested(_) => unsafe {
                    let _redraw_span = tracy_client::span!("RedrawRequested");

                    device.wait_for_fences(&[render_fence], true, u64::MAX)?;

                    device.reset_fences(&[render_fence])?;

                    device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

                    let swapchain_image_index = match swapchain_loader.acquire_next_image(
                        swapchain.swapchain,
                        u64::MAX,
                        present_semaphore,
                        vk::Fence::null(),
                    ) {
                        Ok((swapchain_image_index, _suboptimal)) => swapchain_image_index,
                        Err(error) => {
                            log::warn!("Next frame error: {:?}", error);
                            return Ok(());
                        }
                    };

                    let tonemap_framebuffer =
                        swapchain_image_framebuffers[swapchain_image_index as usize];

                    {
                        device.begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        )?;

                        record(RecordParams {
                            device: &device,
                            command_buffer,
                            pipelines: &pipelines,
                            draw_buffers: &draw_buffers,
                            light_buffers: &light_buffers,
                            profiling_ctx: &mut profiling_ctx,
                            model_buffers: &model_buffers,
                            render_passes: &render_passes,
                            draw_framebuffer,
                            tonemap_framebuffer,
                            transmission_framebuffer,
                            descriptor_sets: &descriptor_sets,
                            hdr_framebuffer: &hdr_framebuffer,
                            opaque_sampled_hdr_framebuffer: &opaque_sampled_hdr_framebuffer,
                            toggle,
                            dynamic: DynamicRecordParams {
                                extent,
                                num_primitives,
                                num_instances,
                                num_lights: lights.len() as u32,
                                push_constants,
                                view_matrix,
                                perspective_matrix,
                                opaque_mip_levels,
                                tonemapping_params,
                            },
                        })?;

                        device.end_command_buffer(command_buffer)?;
                    }

                    {
                        let _submission_span = tracy_client::span!("Command buffer submission");

                        device.queue_submit(
                            queue,
                            &[*vk::SubmitInfo::builder()
                                .wait_semaphores(&[present_semaphore])
                                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                                .command_buffers(&[command_buffer])
                                .signal_semaphores(&[render_semaphore])],
                            render_fence,
                        )?;
                    }

                    {
                        let _presentation_span = tracy_client::span!("Queue presentation");

                        swapchain_loader.queue_present(
                            queue,
                            &vk::PresentInfoKHR::builder()
                                .wait_semaphores(&[render_semaphore])
                                .swapchains(&[swapchain.swapchain])
                                .image_indices(&[swapchain_image_index]),
                        )?;
                    }

                    profiling_ctx.collect(&device)?;

                    tracy_client::finish_continuous_frame!();
                },
                Event::LoopDestroyed => {
                    unsafe {
                        device.queue_wait_idle(queue)?;
                    }

                    {
                        depthbuffer.cleanup(&device, &mut allocator)?;
                        light_buffers.cleanup(&device, &mut allocator)?;
                        image_manager.cleanup(&device, &mut allocator)?;
                        uniforms_buffer.cleanup(&device, &mut allocator)?;
                        draw_buffers.cleanup(&device, &mut allocator)?;
                        hdr_framebuffer.cleanup(&device, &mut allocator)?;
                        opaque_sampled_hdr_framebuffer.cleanup(&device, &mut allocator)?;
                        model_buffers.cleanup(&device, &mut allocator)?;
                        acceleration_structure_debugging_uniforms_buffer
                            .cleanup(&device, &mut allocator)?;

                        for acceleration_structure in &acceleration_structures {
                            acceleration_structure
                                .buffer
                                .cleanup(&device, &mut allocator)?;
                        }

                        if let Some(acceleration_structure) =
                            top_level_acceleration_structure.as_ref()
                        {
                            acceleration_structure
                                .buffer
                                .cleanup(&device, &mut allocator)?;
                        }
                    }
                }
                _ => {}
            }

            Ok(())
        };

        if let Err(loop_closure) = loop_closure() {
            log::error!("Error: {}", loop_closure);
        }
    });
}

struct LightBuffers {
    lights: ash_abstractions::Buffer,
    cluster_light_counts: ash_abstractions::Buffer,
    cluster_light_indices: ash_abstractions::Buffer,
}

impl LightBuffers {
    fn cleanup(
        &self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> anyhow::Result<()> {
        self.lights.cleanup(device, allocator)?;
        self.cluster_light_counts.cleanup(device, allocator)?;
        self.cluster_light_indices.cleanup(device, allocator)?;
        Ok(())
    }
}

unsafe fn record_write_cluster_data(
    init_resources: &ash_abstractions::InitResources,
    pipelines: &Pipelines,
    descriptor_sets: &DescriptorSets,
    perspective_matrix: Mat4,
    screen_dimensions: UVec2,
) {
    init_resources.device.cmd_bind_pipeline(
        init_resources.command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipelines.write_cluster_data,
    );

    init_resources.device.cmd_bind_descriptor_sets(
        init_resources.command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipelines.write_cluster_data_pipeline_layout,
        0,
        &[descriptor_sets.main, descriptor_sets.cluster_data],
        &[],
    );

    init_resources.device.cmd_push_constants(
        init_resources.command_buffer,
        pipelines.write_cluster_data_pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes_of(&shared_structs::WriteClusterDataPushConstants {
            inverse_perspective: perspective_matrix.inverse(),
            screen_dimensions,
        }),
    );

    init_resources.device.cmd_dispatch(
        init_resources.command_buffer,
        dispatch_count(NUM_CLUSTERS_X, 4),
        dispatch_count(NUM_CLUSTERS_Y, 4),
        dispatch_count(NUM_DEPTH_SLICES, 4),
    );
}

struct RecordParams<'a> {
    device: &'a ash::Device,
    command_buffer: vk::CommandBuffer,
    pipelines: &'a Pipelines,
    draw_buffers: &'a DrawBuffers,
    light_buffers: &'a LightBuffers,
    profiling_ctx: &'a ProfilingContext,
    model_buffers: &'a ModelBuffers,
    render_passes: &'a RenderPasses,
    draw_framebuffer: vk::Framebuffer,
    transmission_framebuffer: vk::Framebuffer,
    tonemap_framebuffer: vk::Framebuffer,
    hdr_framebuffer: &'a ash_abstractions::Image,
    opaque_sampled_hdr_framebuffer: &'a ash_abstractions::Image,
    descriptor_sets: &'a DescriptorSets,
    dynamic: DynamicRecordParams,
    toggle: bool,
}

struct DynamicRecordParams {
    extent: vk::Extent2D,
    num_primitives: u32,
    num_instances: u32,
    num_lights: u32,
    push_constants: PushConstants,
    view_matrix: Mat4,
    perspective_matrix: Mat4,
    opaque_mip_levels: u32,
    tonemapping_params: colstodian::tonemap::BakedLottesTonemapperParams,
}

unsafe fn record(params: RecordParams) -> anyhow::Result<()> {
    let _span = tracy_client::span!("Command buffer recording");

    let RecordParams {
        device,
        command_buffer,
        pipelines,
        draw_buffers,
        light_buffers,
        profiling_ctx,
        render_passes,
        draw_framebuffer,
        transmission_framebuffer,
        tonemap_framebuffer,
        toggle,
        dynamic:
            DynamicRecordParams {
                num_primitives,
                num_instances,
                num_lights,
                push_constants,
                view_matrix,
                perspective_matrix,
                extent,
                opaque_mip_levels,
                tonemapping_params,
            },
        model_buffers,
        hdr_framebuffer,
        opaque_sampled_hdr_framebuffer,
        descriptor_sets,
    } = params;

    let draw_clear_values = [
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        },
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
    ];

    let tonemap_clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }];

    let area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };

    let draw_render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_passes.draw)
        .framebuffer(draw_framebuffer)
        .render_area(area)
        .clear_values(&draw_clear_values);

    let transmission_render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_passes.transmission)
        .framebuffer(transmission_framebuffer)
        .render_area(area);

    let tonemap_render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(render_passes.tonemap)
        .framebuffer(tonemap_framebuffer)
        .render_area(area)
        .clear_values(&tonemap_clear_values);

    let viewport = *vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(extent.width as f32)
        .height(extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    profiling_ctx.reset(device, command_buffer);

    let all_commands_profiling_zone = profiling_zone!(
        "all commands",
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        device,
        command_buffer,
        profiling_ctx
    );

    {
        let _profiling_zone = profiling_zone!(
            "frustum culling",
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            device,
            command_buffer,
            profiling_ctx
        );

        {
            let profiling_zone = profiling_zone!(
                "zeroing the instance count buffer",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_fill_buffer(
                command_buffer,
                draw_buffers.instance_count_buffer.buffer,
                0,
                vk::WHOLE_SIZE,
                0,
            );

            drop(profiling_zone);

            let profiling_zone = profiling_zone!(
                "zeroing the draw count buffer",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_fill_buffer(
                command_buffer,
                draw_buffers.draw_counts_buffer.buffer,
                0,
                vk::WHOLE_SIZE,
                0,
            );

            device.cmd_fill_buffer(
                command_buffer,
                light_buffers.cluster_light_counts.buffer,
                0,
                vk::WHOLE_SIZE,
                0,
            );

            drop(profiling_zone);
        }

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::TransferWrite],
                next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
            }),
            &[],
            &[],
        );

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipelines.frustum_culling_pipeline_layout,
            0,
            &[
                descriptor_sets.frustum_culling,
                descriptor_sets.instance_buffer,
            ],
            &[],
        );

        let trunc_row = |index| perspective_matrix.row(index).truncate();

        // Get the left and top planes (the ones that satisfy 'x + w < 0' and 'y + w < 0') (I think, don't quote me on this)
        // https://github.com/zeux/niagara/blob/98f5d5ae2b48e15e145e3ad13ae7f4f9f1e0e297/src/niagara.cpp#L822-L823
        let frustum_x = (trunc_row(3) + trunc_row(0)).normalize();
        let frustum_y = (trunc_row(3) + trunc_row(1)).normalize();

        device.cmd_push_constants(
            command_buffer,
            pipelines.frustum_culling_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytes_of(&shared_structs::CullingPushConstants {
                view: view_matrix,
                frustum_x_xz: frustum_x.xz(),
                frustum_y_yz: frustum_y.yz(),
                z_near: Z_NEAR,
            }),
        );

        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipelines.frustum_culling,
        );

        {
            let _profiling_zone = profiling_zone!(
                "frustum culling compute shader",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_dispatch(command_buffer, dispatch_count(num_instances, 64), 1, 1);
        }

        {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.assign_lights_to_clusters,
            );

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.lights_pipeline_layout,
                0,
                &[descriptor_sets.lights, descriptor_sets.cluster_data],
                &[],
            );

            device.cmd_push_constants(
                command_buffer,
                pipelines.lights_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&shared_structs::AssignLightsPushConstants { view_matrix }),
            );

            device.cmd_dispatch(
                command_buffer,
                dispatch_count(NUM_CLUSTERS, 8),
                dispatch_count(num_lights, 8),
                1,
            );
        }

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_accesses: &[vk_sync::AccessType::ComputeShaderReadOther],
            }),
            &[],
            &[],
        );

        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipelines.demultiplex_draws,
        );

        {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.frustum_culling_pipeline_layout,
                0,
                &[
                    descriptor_sets.frustum_culling,
                    descriptor_sets.instance_buffer,
                ],
                &[],
            );

            let _profiling_zone = profiling_zone!(
                "demultiplex draws compute shader",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_dispatch(command_buffer, dispatch_count(num_primitives, 64), 1, 1);
        }

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_accesses: &[vk_sync::AccessType::IndirectBuffer],
            }),
            &[],
            &[],
        );
    }

    device.cmd_begin_render_pass(
        command_buffer,
        &draw_render_pass_info,
        vk::SubpassContents::INLINE,
    );

    device.cmd_set_scissor(command_buffer, 0, &[area]);
    device.cmd_set_viewport(command_buffer, 0, &[viewport]);

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.pipeline_layout,
        0,
        &[
            descriptor_sets.main,
            descriptor_sets.instance_buffer,
            descriptor_sets.lights,
        ],
        &[],
    );

    device.cmd_push_constants(
        command_buffer,
        pipelines.pipeline_layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        bytes_of(&push_constants),
    );

    device.cmd_bind_vertex_buffers(
        command_buffer,
        0,
        &[
            model_buffers.position.buffer,
            model_buffers.normal.buffer,
            model_buffers.uv.buffer,
        ],
        &[0, 0, 0],
    );

    device.cmd_bind_index_buffer(
        command_buffer,
        model_buffers.index.buffer,
        0,
        vk::IndexType::UINT32,
    );

    {
        let _depth_profiling_zone =
            profiling_zone!("depth pre pass", device, command_buffer, profiling_ctx);

        {
            let _profiling_zone = profiling_zone!(
                "depth pre pass opaque",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.depth_pre_pass,
            );

            draw_buffers
                .opaque
                .record(device, &draw_buffers.draw_counts_buffer, 0, command_buffer);
        }

        {
            let _profiling_zone = profiling_zone!(
                "depth pre pass alpha clipped",
                device,
                command_buffer,
                profiling_ctx
            );

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.depth_pre_pass_alpha_clip,
            );

            draw_buffers.alpha_clip.record(
                device,
                &draw_buffers.draw_counts_buffer,
                1,
                command_buffer,
            );
        }
    }

    device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);

    /*{
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipelines.cluster_debugging,
        );

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipelines.cluster_debugging_pipeline_layout,
            0,
            &[
                descriptor_sets.cluster_data,
            ],
            &[],
        );

        device.cmd_draw(command_buffer, NUM_CLUSTERS * 8, 1, 0, 0);
    }*/

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.normal,
    );

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.pipeline_layout,
        0,
        &[
            descriptor_sets.main,
            descriptor_sets.instance_buffer,
            descriptor_sets.lights,
        ],
        &[],
    );

    {
        let _profiling_zone = profiling_zone!("main opaque", device, command_buffer, profiling_ctx);
        draw_buffers
            .opaque
            .record(device, &draw_buffers.draw_counts_buffer, 0, command_buffer);
    }

    {
        let _profiling_zone =
            profiling_zone!("main alpha clipped", device, command_buffer, profiling_ctx);
        draw_buffers
            .alpha_clip
            .record(device, &draw_buffers.draw_counts_buffer, 1, command_buffer);
    }

    device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);

    {
        let _profiling_zone = profiling_zone!(
            "depth pre pass transmissive",
            device,
            command_buffer,
            profiling_ctx
        );

        {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.depth_pre_pass_transmissive,
            );

            draw_buffers.transmission.record(
                device,
                &draw_buffers.draw_counts_buffer,
                2,
                command_buffer,
            );
        }

        {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.depth_pre_pass_transmissive_alpha_clip,
            );

            draw_buffers.transmission_alpha_clip.record(
                device,
                &draw_buffers.draw_counts_buffer,
                3,
                command_buffer,
            );
        }
    }

    device.cmd_end_render_pass(command_buffer);

    {
        let _profiling_zone = profiling_zone!(
            "opaque framebuffer mipchain",
            device,
            command_buffer,
            profiling_ctx
        );

        ash_abstractions::generate_mips(
            device,
            command_buffer,
            opaque_sampled_hdr_framebuffer.image,
            extent.width as i32,
            extent.height as i32,
            opaque_mip_levels,
            &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
            vk_sync::ImageLayout::Optimal,
        );
    }

    device.cmd_begin_render_pass(
        command_buffer,
        &transmission_render_pass_info,
        vk::SubpassContents::INLINE,
    );

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.transmission_pipeline_layout,
        0,
        &[
            descriptor_sets.main,
            descriptor_sets.instance_buffer,
            descriptor_sets.lights,
            descriptor_sets.opaque_sampled_hdr_framebuffer,
        ],
        &[],
    );

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.transmission,
    );

    {
        let _profiling_zone = profiling_zone!(
            "opaque transmissive objects",
            device,
            command_buffer,
            profiling_ctx
        );

        draw_buffers.transmission.record(
            device,
            &draw_buffers.draw_counts_buffer,
            2,
            command_buffer,
        );
    }

    {
        let _profiling_zone = profiling_zone!(
            "alpha clip transmissive objects",
            device,
            command_buffer,
            profiling_ctx
        );

        draw_buffers.transmission_alpha_clip.record(
            device,
            &draw_buffers.draw_counts_buffer,
            3,
            command_buffer,
        );
    }

    device.cmd_end_render_pass(command_buffer);

    if let Some(pipeline) = pipelines
        .acceleration_structure_debugging
        .filter(|_| toggle)
    {
        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[
                    vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_layout: vk_sync::ImageLayout::Optimal,
                image: hdr_framebuffer.image,
                range: subresource_range,
                discard_contents: true,
                ..Default::default()
            }],
        );

        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipelines.acceleration_structure_debugging_layout,
            0,
            &[descriptor_sets.acceleration_structure_debugging],
            &[],
        );

        device.cmd_push_constants(
            command_buffer,
            pipelines.acceleration_structure_debugging_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytes_of(&push_constants),
        );

        device.cmd_dispatch(
            command_buffer,
            dispatch_count(extent.width, 8),
            dispatch_count(extent.height, 8),
            1,
        );

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_accesses: &[
                    vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_layout: vk_sync::ImageLayout::Optimal,
                image: hdr_framebuffer.image,
                range: subresource_range,
                ..Default::default()
            }],
        );
    }

    device.cmd_begin_render_pass(
        command_buffer,
        &tonemap_render_pass_info,
        vk::SubpassContents::INLINE,
    );

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.tonemap_pipeline_layout,
        0,
        &[descriptor_sets.main, descriptor_sets.hdr_framebuffer],
        &[],
    );

    device.cmd_push_constants(
        command_buffer,
        pipelines.tonemap_pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        bytes_of(&tonemapping_params),
    );

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipelines.tonemap,
    );

    {
        let _profiling_zone = profiling_zone!("tonemapping", device, command_buffer, profiling_ctx);

        device.cmd_draw(command_buffer, 3, 1, 0, 0);
    }

    device.cmd_end_render_pass(command_buffer);

    {
        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(1)
            .level_count(opaque_mip_levels - 1)
            .layer_count(1);

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[
                    vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_accesses: &[vk_sync::AccessType::TransferWrite],
                next_layout: vk_sync::ImageLayout::Optimal,
                image: opaque_sampled_hdr_framebuffer.image,
                range: subresource_range,
                discard_contents: true,
                ..Default::default()
            }],
        );
    }

    drop(all_commands_profiling_zone);

    Ok(())
}

fn create_transmission_framebuffer(
    device: &ash::Device,
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    hdr_framebuffer: &ash_abstractions::Image,
    depthbuffer: &ash_abstractions::Image,
) -> anyhow::Result<vk::Framebuffer> {
    Ok(unsafe {
        device.create_framebuffer(
            &vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&[depthbuffer.view, hdr_framebuffer.view])
                .width(extent.width)
                .height(extent.height)
                .layers(1),
            None,
        )
    }?)
}

fn create_draw_framebuffer(
    device: &ash::Device,
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    hdr_framebuffer: &ash_abstractions::Image,
    opaque_sampled_hdr_framebuffer_top_mip_view: vk::ImageView,
    depthbuffer: &ash_abstractions::Image,
) -> anyhow::Result<vk::Framebuffer> {
    Ok(unsafe {
        device.create_framebuffer(
            &vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&[
                    depthbuffer.view,
                    hdr_framebuffer.view,
                    opaque_sampled_hdr_framebuffer_top_mip_view,
                ])
                .width(extent.width)
                .height(extent.height)
                .layers(1),
            None,
        )
    }?)
}

fn create_swapchain_image_framebuffers(
    device: &ash::Device,
    extent: vk::Extent2D,
    swapchain: &ash_abstractions::Swapchain,
    render_passes: &RenderPasses,
) -> anyhow::Result<Vec<vk::Framebuffer>> {
    swapchain
        .image_views
        .iter()
        .map(|image_view| {
            unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(render_passes.tonemap)
                        .attachments(&[*image_view])
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
            }
            .map_err(|err| err.into())
        })
        .collect()
}

fn create_depthbuffer(
    width: u32,
    height: u32,
    init_resources: &mut ash_abstractions::InitResources,
) -> anyhow::Result<ash_abstractions::Image> {
    ash_abstractions::Image::new(
        &ash_abstractions::ImageDescriptor {
            width,
            height,
            name: "depthbuffer",
            mip_levels: 1,
            format: vk::Format::D32_SFLOAT,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            next_accesses: &[vk_sync::AccessType::DepthStencilAttachmentWrite],
            next_layout: vk_sync::ImageLayout::Optimal,
        },
        init_resources,
    )
}

fn create_hdr_framebuffer(
    width: u32,
    height: u32,
    mip_levels: u32,
    name: &str,
    extra_usage: vk::ImageUsageFlags,
    init_resources: &mut ash_abstractions::InitResources,
) -> anyhow::Result<ash_abstractions::Image> {
    ash_abstractions::Image::new(
        &ash_abstractions::ImageDescriptor {
            width,
            height,
            name,
            mip_levels,
            format: vk::Format::R16G16B16A16_SFLOAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | extra_usage,
            next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
            next_layout: vk_sync::ImageLayout::Optimal,
        },
        init_resources,
    )
}

struct DrawBuffer {
    buffer: ash_abstractions::Buffer,
    max_draws: u32,
}

impl DrawBuffer {
    fn new_from_max_draws(
        max_draws: u32,
        name: &str,
        init_resources: &mut ash_abstractions::InitResources,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            buffer: ash_abstractions::Buffer::new_of_size(
                max_draws.max(1) as u64
                    * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64,
                name,
                vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                init_resources,
            )?,
            max_draws,
        })
    }

    unsafe fn record(
        &self,
        device: &ash::Device,
        draw_count_buffer: &ash_abstractions::Buffer,
        draw_count_index: u64,
        command_buffer: vk::CommandBuffer,
    ) {
        device.cmd_draw_indexed_indirect_count(
            command_buffer,
            self.buffer.buffer,
            0,
            draw_count_buffer.buffer,
            draw_count_index * 4,
            self.max_draws,
            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
        )
    }
}

struct DrawBuffers {
    opaque: DrawBuffer,
    alpha_clip: DrawBuffer,
    transmission: DrawBuffer,
    transmission_alpha_clip: DrawBuffer,
    draw_counts_buffer: ash_abstractions::Buffer,
    instance_count_buffer: ash_abstractions::Buffer,
}

impl DrawBuffers {
    fn new(
        max_draw_counts: MaxDrawCounts,
        init_resources: &mut ash_abstractions::InitResources,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            draw_counts_buffer: ash_abstractions::Buffer::new_of_size(
                std::mem::size_of::<MaxDrawCounts>() as u64,
                "draw counts",
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                init_resources,
            )?,
            instance_count_buffer: ash_abstractions::Buffer::new_of_size(
                4 * (max_draw_counts.opaque
                    + max_draw_counts.alpha_clip
                    + max_draw_counts.transmission
                    + max_draw_counts.transmission_alpha_clip) as u64,
                "instance count buffer",
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                init_resources,
            )?,
            opaque: DrawBuffer::new_from_max_draws(
                max_draw_counts.opaque,
                "opaque",
                init_resources,
            )?,
            alpha_clip: DrawBuffer::new_from_max_draws(
                max_draw_counts.alpha_clip,
                "alpha clip",
                init_resources,
            )?,
            transmission: DrawBuffer::new_from_max_draws(
                max_draw_counts.transmission,
                "transmission",
                init_resources,
            )?,
            transmission_alpha_clip: DrawBuffer::new_from_max_draws(
                max_draw_counts.transmission_alpha_clip,
                "transmission alpha clip",
                init_resources,
            )?,
        })
    }

    fn cleanup(
        &self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> anyhow::Result<()> {
        self.opaque.buffer.cleanup(device, allocator)?;
        self.alpha_clip.buffer.cleanup(device, allocator)?;
        self.transmission.buffer.cleanup(device, allocator)?;
        self.transmission_alpha_clip
            .buffer
            .cleanup(device, allocator)?;
        self.draw_counts_buffer.cleanup(device, allocator)?;
        self.instance_count_buffer.cleanup(device, allocator)?;
        Ok(())
    }
}

#[derive(Default)]
struct ModelStagingBuffers {
    position: Vec<Vec3>,
    normal: Vec<Vec3>,
    uv: Vec<Vec2>,
    index: Vec<u32>,
    // storage buffers
    instances: Vec<Instance>,
    primitives: Vec<shared_structs::PrimitiveInfo>,
    materials: Vec<shared_structs::MaterialInfo>,
}

impl ModelStagingBuffers {
    fn upload(
        &self,
        init_resources: &mut ash_abstractions::InitResources,
    ) -> anyhow::Result<ModelBuffers> {
        let ray_tracing_flags = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        Ok(ModelBuffers {
            position: ash_abstractions::Buffer::new(
                unsafe { cast_slice(&self.position) },
                "position buffer",
                vk::BufferUsageFlags::VERTEX_BUFFER | ray_tracing_flags,
                init_resources,
            )?,
            normal: ash_abstractions::Buffer::new(
                unsafe { cast_slice(&self.normal) },
                "normal buffer",
                vk::BufferUsageFlags::VERTEX_BUFFER,
                init_resources,
            )?,
            uv: ash_abstractions::Buffer::new(
                unsafe { cast_slice(&self.uv) },
                "uv buffer",
                vk::BufferUsageFlags::VERTEX_BUFFER,
                init_resources,
            )?,
            index: ash_abstractions::Buffer::new(
                unsafe { cast_slice(&self.index) },
                "index buffer",
                vk::BufferUsageFlags::INDEX_BUFFER | ray_tracing_flags,
                init_resources,
            )?,
            instances: ash_abstractions::Buffer::new(
                unsafe { cast_slice(&self.instances) },
                "instances",
                vk::BufferUsageFlags::STORAGE_BUFFER,
                init_resources,
            )?,
            materials: ash_abstractions::Buffer::new(
                unsafe { cast_slice(&self.materials) },
                "materials",
                vk::BufferUsageFlags::STORAGE_BUFFER,
                init_resources,
            )?,
            primitives: ash_abstractions::Buffer::new(
                unsafe { cast_slice(&self.primitives) },
                "primitive infos",
                vk::BufferUsageFlags::STORAGE_BUFFER,
                init_resources,
            )?,
        })
    }
}

struct ModelBuffers {
    position: ash_abstractions::Buffer,
    normal: ash_abstractions::Buffer,
    uv: ash_abstractions::Buffer,
    index: ash_abstractions::Buffer,
    // storage buffers
    instances: ash_abstractions::Buffer,
    materials: ash_abstractions::Buffer,
    primitives: ash_abstractions::Buffer,
}

impl ModelBuffers {
    fn cleanup(
        &self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> anyhow::Result<()> {
        self.position.cleanup(device, allocator)?;
        self.normal.cleanup(device, allocator)?;
        self.uv.cleanup(device, allocator)?;
        self.index.cleanup(device, allocator)?;
        self.instances.cleanup(device, allocator)?;
        self.materials.cleanup(device, allocator)?;
        self.primitives.cleanup(device, allocator)?;
        Ok(())
    }
}

fn mip_levels_for_size(width: u32, height: u32) -> u32 {
    (width.min(height) as f32).log2() as u32 + 1
}

trait Castable {}

// I had a problem where I was casting a reference without realising it, so we
// use this trait as an allowlist.
impl Castable for vk::DrawIndexedIndirectCommand {}
impl Castable for u32 {}
impl Castable for Vec2 {}
impl Castable for Vec3 {}
impl Castable for shared_structs::Instance {}
impl Castable for shared_structs::Light {}
impl Castable for shared_structs::MaterialInfo {}
impl Castable for shared_structs::Uniforms {}
impl Castable for shared_structs::PushConstants {}
impl Castable for MaxDrawCounts {}
impl Castable for vk::AccelerationStructureInstanceKHR {}
impl Castable for shared_structs::CullingPushConstants {}
impl Castable for shared_structs::PrimitiveInfo {}
impl Castable for shared_structs::AccelerationStructureDebuggingUniforms {}
impl Castable for shared_structs::WriteClusterDataPushConstants {}
impl Castable for shared_structs::AssignLightsPushConstants {}
impl Castable for colstodian::tonemap::BakedLottesTonemapperParams {}

unsafe fn cast_slice<T: Castable>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        slice as *const [T] as *const u8,
        slice.len() * std::mem::size_of::<T>(),
    )
}

unsafe fn bytes_of<T: Castable>(reference: &T) -> &[u8] {
    std::slice::from_raw_parts(reference as *const T as *const u8, std::mem::size_of::<T>())
}

#[derive(Default)]
struct KeyboardState {
    forwards: bool,
    right: bool,
    backwards: bool,
    left: bool,
    sun_up: bool,
    sun_cw: bool,
    sun_ccw: bool,
    sun_down: bool,
}

const fn dispatch_count(num: u32, group_size: u32) -> u32 {
    ((num - 1) / group_size) + 1
}

#[derive(Debug, Default, Clone, Copy)]
struct MaxDrawCounts {
    opaque: u32,
    alpha_clip: u32,
    transmission: u32,
    transmission_alpha_clip: u32,
}

pub fn transpose_matrix_for_acceleration_structure_instance(matrix: Mat4) -> [f32; 12] {
    let row_0 = matrix.row(0);
    let row_1 = matrix.row(1);
    let row_2 = matrix.row(2);
    [
        row_0.x, row_0.y, row_0.z, row_0.w, row_1.x, row_1.y, row_1.z, row_1.w, row_2.x, row_2.y,
        row_2.z, row_2.w,
    ]
}

// Stolen from ash master.
#[repr(transparent)]
#[derive(Debug)]
pub struct Packed24_8(u32);

impl Packed24_8 {
    pub fn new(low_24: u32, high_8: u8) -> Self {
        Self((low_24 & 0x00ff_ffff) | (u32::from(high_8) << 24))
    }

    /// Extracts the least-significant 24 bits (3 bytes) of this integer
    pub fn low_24(&self) -> u32 {
        self.0 & 0xffffff
    }

    /// Extracts the most significant 8 bits (single byte) of this integer
    pub fn high_8(&self) -> u8 {
        (self.0 >> 24) as u8
    }
}

fn end_init_resources(
    init_resources: ash_abstractions::InitResources,
    queue: vk::Queue,
) -> anyhow::Result<()> {
    unsafe {
        init_resources
            .device
            .end_command_buffer(init_resources.command_buffer)?;

        let fence = init_resources
            .device
            .create_fence(&vk::FenceCreateInfo::builder(), None)?;

        init_resources.device.queue_submit(
            queue,
            &[*vk::SubmitInfo::builder().command_buffers(&[init_resources.command_buffer])],
            fence,
        )?;

        init_resources
            .device
            .wait_for_fences(&[fence], true, u64::MAX)?;
    }

    Ok(())
}

#[derive(Debug)]
struct Sun {
    pitch: f32,
    yaw: f32,
}

impl Sun {
    fn as_normal(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
    }
}
