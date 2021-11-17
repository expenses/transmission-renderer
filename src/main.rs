use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{Surface as SurfaceLoader, Swapchain as SwapchainLoader};
use ash::vk;
use ash_abstractions::CStrList;
use std::ffi::{CStr, CString};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Fullscreen;

use glam::{Mat3, Mat4, UVec2, Vec2, Vec3, Vec4};
use shared_structs::PointLight;

mod profiling;

fn perspective_infinite_z_vk(vertical_fov: f32, aspect_ratio: f32, z_near: f32) -> Mat4 {
    let t = (vertical_fov / 2.0).tan();
    let sy = 1.0 / t;
    let sx = sy / aspect_ratio;

    Mat4::from_cols(
        Vec4::new(sx, 0.0, 0.0, 0.0),
        Vec4::new(0.0, -sy, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -1.0, -1.0),
        Vec4::new(0.0, 0.0, -z_near, 0.0),
    )
}

pub const DRAW_COMMAND_SIZE: usize = std::mem::size_of::<vk::DrawIndexedIndirectCommand>();
pub const MAX_IMAGES: u32 = 195;

fn main() -> anyhow::Result<()> {
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
    let window = winit::window::Window::new(&event_loop)?;

    let entry = unsafe { ash::Entry::new() }?;

    let api_version = vk::API_VERSION_1_1;

    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"Nice Grass\0")?)
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

    let device_extensions = CStrList::new(vec![
        SwapchainLoader::name(),
        vk::ExtDescriptorIndexingFn::name(),
    ]);

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

        let device_features = vk::PhysicalDeviceFeatures::builder().multi_draw_indirect(true);

        let mut null_descriptor_feature =
            vk::PhysicalDeviceRobustness2FeaturesEXT::builder().null_descriptor(true);

        let mut descriptor_indexing =
            vk::PhysicalDeviceDescriptorIndexingFeatures::builder().runtime_descriptor_array(true);

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_features(&device_features)
            .enabled_extension_names(device_extensions.pointers())
            .enabled_layer_names(enabled_layers.pointers())
            .push_next(&mut null_descriptor_feature)
            .push_next(&mut descriptor_indexing);

        unsafe { instance.create_device(physical_device, &device_info, None) }?
    };

    let render_passes = RenderPasses::new(&device, surface_format.format)?;
    let descriptor_set_layouts = DescriptorSetLayouts::new(&device)?;

    let pipeline_cache =
        unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }?;

    let pipelines = Pipelines::new(
        &device,
        &render_passes,
        &descriptor_set_layouts,
        pipeline_cache,
    )?;

    let queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

    let (command_buffer, command_pool) = {
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder().queue_family_index(graphics_queue_family),
                None,
            )
        }?;

        let cmd_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }?[0];

        (command_buffer, command_pool)
    };

    let mut allocator =
        gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
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

    let filename = std::env::args().nth(1).unwrap();

    let mut materials = Vec::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let model = load_gltf(
        &filename,
        &mut init_resources,
        &mut image_manager,
        &mut buffers_to_cleanup,
        &mut materials,
        &mut vertices,
        &mut indices,
    )?;

    let filename2 = std::env::args().nth(2).unwrap();

    let model2 = load_gltf(
        &filename2,
        &mut init_resources,
        &mut image_manager,
        &mut buffers_to_cleanup,
        &mut materials,
        &mut vertices,
        &mut indices,
    )?;

    let model_materials = ash_abstractions::Buffer::new(
        unsafe { cast_slice(&materials) },
        "materials",
        vk::BufferUsageFlags::STORAGE_BUFFER,
        &mut init_resources,
    )?;

    let vertices = ash_abstractions::Buffer::new(
        unsafe { cast_slice(&vertices) },
        "vertices",
        vk::BufferUsageFlags::VERTEX_BUFFER,
        &mut init_resources,
    )?;

    let indices = ash_abstractions::Buffer::new(
        unsafe { cast_slice(&indices) },
        "indices",
        vk::BufferUsageFlags::INDEX_BUFFER,
        &mut init_resources,
    )?;

    let draw_indirect_buffer = ash_abstractions::Buffer::new(
        unsafe {
            cast_slice(&[
                // Opaque draws
                model.opaque.as_draw_indexed_indirect_command(0),
                model2.opaque.as_draw_indexed_indirect_command(1),
                // Alpha clip draws.
                model.alpha_clip.as_draw_indexed_indirect_command(0),
                model2.alpha_clip.as_draw_indexed_indirect_command(1),
                // Transmission draws
                model.transmission.as_draw_indexed_indirect_command(0),
                model2.transmission.as_draw_indexed_indirect_command(1),
                model
                    .transmission_alpha_clip
                    .as_draw_indexed_indirect_command(0),
                model2
                    .transmission_alpha_clip
                    .as_draw_indexed_indirect_command(1),
            ])
        },
        "draw indirect",
        vk::BufferUsageFlags::INDIRECT_BUFFER,
        &mut init_resources,
    )?;

    let mut depthbuffer = create_depthbuffer(extent.width, extent.height, &mut init_resources)?;
    let mut hdr_framebuffer = create_hdr_framebuffer(
        extent.width,
        extent.height,
        1,
        "hdr framebuffer",
        vk::ImageUsageFlags::TRANSFER_SRC,
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
                .format(vk::Format::R32G32B32A32_SFLOAT)
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
                next_accesses: &[
                    vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_layout: vk_sync::ImageLayout::Optimal,
                image: opaque_sampled_hdr_framebuffer.image,
                range: subresource_range,
                discard_contents: true,
                ..Default::default()
            }],
        );
    }

    let lights_buffer = ash_abstractions::Buffer::new(
        unsafe {
            cast_slice(&[
                PointLight {
                    position: Vec3::new(0.0, 10.0, 0.0).into(),
                    colour_and_intensity: Vec4::new(1.0, 0.0, 0.0, 10000.0),
                },
                PointLight {
                    position: Vec3::new(1000.0, 10.0, 0.0).into(),
                    colour_and_intensity: Vec4::new(0.0, 0.0, 1.0, 100000.0),
                },
            ])
        },
        "lights",
        vk::BufferUsageFlags::STORAGE_BUFFER,
        &mut init_resources,
    )?;

    let instance_buffer = ash_abstractions::Buffer::new(
        unsafe {
            cast_slice(&[
                Instance {
                    translation: Vec3::ZERO,
                    rotation: Mat3::IDENTITY,
                    scale: 1.0 / 0.00800000037997961,
                },
                Instance {
                    translation: Vec3::new(0.0, 250.0, 0.0),
                    rotation: Mat3::IDENTITY,
                    scale: 100.0,
                },
            ])
        },
        "instance",
        vk::BufferUsageFlags::VERTEX_BUFFER,
        &mut init_resources,
    )?;

    let tonemapping_params = colstodian::tonemap::BakedLottesTonemapperParams::from(
        colstodian::tonemap::LottesTonemapperParams {
            ..Default::default()
        },
    );

    let sun_uniform_buffer = ash_abstractions::Buffer::new(
        unsafe {
            bytes_of(&shared_structs::SunUniform {
                dir: Vec3::new(1.0, 2.0, 1.0).normalize().into(),
                intensity: Vec3::splat(3.0),
            })
        },
        "sun uniform",
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        &mut init_resources,
    )?;

    let query_pool = profiling::QueryPool::new(&mut init_resources)?;

    drop(init_resources);

    unsafe {
        device.end_command_buffer(command_buffer)?;
        let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

        device.queue_submit(
            queue,
            &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
            fence,
        )?;

        device.wait_for_fences(&[fence], true, u64::MAX)?;
    }

    for buffer in buffers_to_cleanup.drain(..) {
        buffer.cleanup_and_drop(&device, &mut allocator)?;
    }

    image_manager.fill_with_dummy_images_up_to(MAX_IMAGES as usize);

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

    let mut camera = dolly::rig::CameraRig::builder()
        .with(dolly::drivers::Position::new(Vec3::new(
            -18.0, 300.0, 100.0,
        )))
        .with(dolly::drivers::YawPitch::new().pitch_degrees(-15.0))
        .with(dolly::drivers::Smooth::new_position_rotation(0.5, 0.25))
        .build();

    let mut perspective_matrix = perspective_infinite_z_vk(
        59.0_f32.to_radians(),
        extent.width as f32 / extent.height as f32,
        0.1,
    );

    let num_tiles = UVec2::new(12, 8);

    let mut push_constants = shared_structs::PushConstants {
        // Updated every frame.
        proj_view: Default::default(),
        view_position: Default::default(),
        num_tiles,
        tile_size_in_pixels: Vec2::new(extent.width as f32, extent.height as f32)
            / num_tiles.as_vec2(),
        debug_froxels: 0,
        framebuffer_size: UVec2::new(extent.width, extent.height),
        ggx_lut_texture_index: ggx_lut_id,
    };

    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&[
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(2),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(2),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(MAX_IMAGES),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::SAMPLER)
                        .descriptor_count(2),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(2),
                ])
                .max_sets(3),
            None,
        )
    }?;

    let descriptor_sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&[
                    descriptor_set_layouts.main,
                    descriptor_set_layouts.hdr_framebuffer,
                    descriptor_set_layouts.hdr_framebuffer,
                ])
                .descriptor_pool(descriptor_pool),
        )
    }?;

    let main_ds = descriptor_sets[0];
    let hdr_framebuffer_ds = descriptor_sets[1];
    let opaque_sampled_hdr_framebuffer_ds = descriptor_sets[2];

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

    unsafe {
        device.update_descriptor_sets(
            &[
                *vk::WriteDescriptorSet::builder()
                    .dst_set(main_ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[*vk::DescriptorBufferInfo::builder()
                        .buffer(lights_buffer.buffer)
                        .range(vk::WHOLE_SIZE)]),
                *image_manager.write_descriptor_set(main_ds, 1),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(main_ds)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[*vk::DescriptorImageInfo::builder().sampler(sampler)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(main_ds)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[*vk::DescriptorBufferInfo::builder()
                        .buffer(model_materials.buffer)
                        .range(vk::WHOLE_SIZE)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(main_ds)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[*vk::DescriptorBufferInfo::builder()
                        .buffer(sun_uniform_buffer.buffer)
                        .range(vk::WHOLE_SIZE)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(main_ds)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[*vk::DescriptorImageInfo::builder().sampler(clamp_sampler)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(hdr_framebuffer_ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&[*vk::DescriptorImageInfo::builder()
                        .image_view(hdr_framebuffer.view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .sampler(sampler)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(opaque_sampled_hdr_framebuffer_ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&[*vk::DescriptorImageInfo::builder()
                        .image_view(opaque_sampled_hdr_framebuffer.view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .sampler(sampler)]),
            ],
            &[],
        )
    }

    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    let present_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };
    let render_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };
    let render_fence = unsafe { device.create_fence(&fence_info, None)? };

    let mut cursor_grab = false;

    let mut screen_center =
        winit::dpi::LogicalPosition::new(extent.width as f64 / 2.0, extent.height as f64 / 2.0);

    let mut profiling_ctx =
        query_pool.into_profiling_context(&device, physical_device_properties.limits)?;

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
                            VirtualKeyCode::F11 => {
                                if is_pressed {
                                    if window.fullscreen().is_some() {
                                        window.set_fullscreen(None);
                                    } else {
                                        window.set_fullscreen(Some(Fullscreen::Borderless(None)))
                                    }
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

                        push_constants.tile_size_in_pixels =
                            Vec2::new(extent.width as f32, extent.height as f32)
                                / num_tiles.as_vec2();

                        perspective_matrix = perspective_infinite_z_vk(
                            59.0_f32.to_radians(),
                            extent.width as f32 / extent.height as f32,
                            0.1,
                        );

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
                            vk::ImageUsageFlags::TRANSFER_SRC,
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
                                    .format(vk::Format::R32G32B32A32_SFLOAT)
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
                                    next_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
                                    next_layout: vk_sync::ImageLayout::Optimal,
                                    image: opaque_sampled_hdr_framebuffer.image,
                                    range: subresource_range,
                                    discard_contents: true,
                                    ..Default::default()
                                }],
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

                            device.update_descriptor_sets(
                                &[
                                    *vk::WriteDescriptorSet::builder()
                                        .dst_set(hdr_framebuffer_ds)
                                        .dst_binding(0)
                                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                        .image_info(&[*vk::DescriptorImageInfo::builder()
                                            .image_view(hdr_framebuffer.view)
                                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                            .sampler(sampler)]),
                                    *vk::WriteDescriptorSet::builder()
                                        .dst_set(opaque_sampled_hdr_framebuffer_ds)
                                        .dst_binding(0)
                                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                        .image_info(&[*vk::DescriptorImageInfo::builder()
                                            .image_view(opaque_sampled_hdr_framebuffer.view)
                                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                            .sampler(sampler)]),
                                ],
                                &[],
                            )
                        }

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
                            &depthbuffer
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

                    camera
                        .driver_mut::<dolly::drivers::Position>()
                        .translate(move_vec * delta_time * 400.0);

                    camera.update(delta_time);

                    push_constants.proj_view = {
                        let view = Mat4::look_at_rh(
                            camera.final_transform.position,
                            camera.final_transform.position + camera.final_transform.forward(),
                            camera.final_transform.up(),
                        );
                        perspective_matrix * view
                    };
                    push_constants.view_position = camera.final_transform.position.into();

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
                        Ok((swapchain_image_index, _suboptimal)) => {
                            swapchain_image_index
                        },
                        Err(error) => {
                            log::warn!("Next frame error: {:?}", error);
                            return Ok(());
                        }
                    };

                    let tonemap_framebuffer =
                        swapchain_image_framebuffers[swapchain_image_index as usize];

                    let draw_clear_values = [
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
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

                    {
                        let _redraw_span = tracy_client::span!("Command buffer recording");

                        device.begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        )?;

                        profiling_ctx.reset(&device, command_buffer);

                        let all_commands_profiling_zone = profiling_zone!("all commands", vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, &device, command_buffer, &mut profiling_ctx);

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
                            &[main_ds],
                            &[],
                        );

                        device.cmd_push_constants(
                            command_buffer,
                            pipelines.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytes_of(&push_constants),
                        );

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.depth_pre_pass,
                        );

                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vertices.buffer, instance_buffer.buffer],
                            &[0, 0],
                        );

                        device.cmd_bind_index_buffer(
                            command_buffer,
                            indices.buffer,
                            0,
                            vk::IndexType::UINT32,
                        );

                        {
                            let _depth_profiling_zone = profiling_zone!("depth pre pass", vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, &device, command_buffer, &mut profiling_ctx);

                            {
                                let _profiling_zone = profiling_zone!("depth pre pass opaque", vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, &device, command_buffer, &mut profiling_ctx);

                                device.cmd_draw_indexed_indirect(
                                    command_buffer,
                                    draw_indirect_buffer.buffer,
                                    0,
                                    2,
                                    DRAW_COMMAND_SIZE as u32,
                                );
                            }

                            device.cmd_bind_pipeline(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipelines.depth_pre_pass_alpha_clip,
                            );

                            {
                                let _profiling_zone = profiling_zone!("depth pre pass alpha clipped", vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, &device, command_buffer, &mut profiling_ctx);

                                device.cmd_draw_indexed_indirect(
                                    command_buffer,
                                    draw_indirect_buffer.buffer,
                                    DRAW_COMMAND_SIZE as u64 * 2,
                                    2,
                                    DRAW_COMMAND_SIZE as u32,
                                );
                            }
                        }

                        device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.normal,
                        );

                        {
                            let _profiling_zone = profiling_zone!("main opaque + alpha clipped", vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, &device, command_buffer, &mut profiling_ctx);

                            device.cmd_draw_indexed_indirect(
                                command_buffer,
                                draw_indirect_buffer.buffer,
                                0,
                                4,
                                DRAW_COMMAND_SIZE as u32,
                            );
                        }

                        device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.depth_pre_pass_transmissive,
                        );

                        {
                            let _profiling_zone = profiling_zone!("depth pre pass transmissive", vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, &device, command_buffer, &mut profiling_ctx);

                            device.cmd_draw_indexed_indirect(
                                command_buffer,
                                draw_indirect_buffer.buffer,
                                DRAW_COMMAND_SIZE as u64 * 4,
                                2,
                                DRAW_COMMAND_SIZE as u32,
                            );

                            device.cmd_bind_pipeline(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipelines.depth_pre_pass_transmissive_alpha_clip,
                            );

                            device.cmd_draw_indexed_indirect(
                                command_buffer,
                                draw_indirect_buffer.buffer,
                                DRAW_COMMAND_SIZE as u64 * 6,
                                2,
                                DRAW_COMMAND_SIZE as u32,
                            );
                        }

                        device.cmd_end_render_pass(command_buffer);

                        device.cmd_begin_render_pass(
                            command_buffer,
                            &transmission_render_pass_info,
                            vk::SubpassContents::INLINE
                        );

                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.transmission_pipeline_layout,
                            0,
                            &[main_ds, opaque_sampled_hdr_framebuffer_ds],
                            &[],
                        );

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipelines.transmission,
                        );

                        {
                            let _profiling_zone = profiling_zone!("transmissive objects", vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, &device, command_buffer, &mut profiling_ctx);

                            device.cmd_draw_indexed_indirect(
                                command_buffer,
                                draw_indirect_buffer.buffer,
                                DRAW_COMMAND_SIZE as u64 * 4,
                                4,
                                DRAW_COMMAND_SIZE as u32,
                            );
                        }

                        device.cmd_end_render_pass(command_buffer);

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
                            &[hdr_framebuffer_ds],
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
                            let _profiling_zone = profiling_zone!("tonemapping", vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, &device, command_buffer, &mut profiling_ctx);

                            device.cmd_draw(command_buffer, 3, 1, 0, 0);
                        }

                        device.cmd_end_render_pass(command_buffer);

                        drop(all_commands_profiling_zone);

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
                        lights_buffer.cleanup(&device, &mut allocator)?;
                        image_manager.cleanup(&device, &mut allocator)?;
                        model_materials.cleanup(&device, &mut allocator)?;
                        instance_buffer.cleanup(&device, &mut allocator)?;
                        sun_uniform_buffer.cleanup(&device, &mut allocator)?;
                        vertices.cleanup(&device, &mut allocator)?;
                        indices.cleanup(&device, &mut allocator)?;
                        draw_indirect_buffer.cleanup(&device, &mut allocator)?;
                        hdr_framebuffer.cleanup(&device, &mut allocator)?;
                        opaque_sampled_hdr_framebuffer.cleanup(&device, &mut allocator)?;
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

struct DescriptorSetLayouts {
    main: vk::DescriptorSetLayout,
    hdr_framebuffer: vk::DescriptorSetLayout,
}

impl DescriptorSetLayouts {
    fn new(device: &ash::Device) -> anyhow::Result<Self> {
        Ok(Self {
            main: unsafe {
                device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(MAX_IMAGES)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(2)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(3)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(4)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(5)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    ]),
                    None,
                )?
            },
            hdr_framebuffer: unsafe {
                device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                    ]),
                    None,
                )?
            },
        })
    }
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

#[derive(Default)]
struct KeyboardState {
    forwards: bool,
    right: bool,
    backwards: bool,
    left: bool,
}

struct Pipelines {
    normal: vk::Pipeline,
    depth_pre_pass: vk::Pipeline,
    depth_pre_pass_alpha_clip: vk::Pipeline,
    depth_pre_pass_transmissive: vk::Pipeline,
    depth_pre_pass_transmissive_alpha_clip: vk::Pipeline,
    transmission: vk::Pipeline,
    tonemap: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    tonemap_pipeline_layout: vk::PipelineLayout,
    transmission_pipeline_layout: vk::PipelineLayout,
}

impl Pipelines {
    fn new(
        device: &ash::Device,
        render_passes: &RenderPasses,
        descriptor_set_layouts: &DescriptorSetLayouts,
        pipeline_cache: vk::PipelineCache,
    ) -> anyhow::Result<Self> {
        let vertex_instanced_entry_point = CString::new("vertex_instanced")?;
        let fragment_entry_point = CString::new("fragment")?;
        let vertex_depth_pre_pass_entry_point = CString::new("depth_pre_pass_instanced")?;
        let vertex_depth_pre_pass_alpha_clip_entry_point =
            CString::new("depth_pre_pass_vertex_alpha_clip")?;
        let fragment_depth_pre_pass_alpha_clip_entry_point =
            CString::new("depth_pre_pass_alpha_clip")?;
        let fragment_transmission_entry_point = CString::new("fragment_transmission")?;
        let fullscreen_tri_entry_point = CString::new("fullscreen_tri")?;
        let fragment_tonemap_entry_point = CString::new("fragment_tonemap")?;

        let module = ash_abstractions::load_shader_module(include_bytes!("../shader.spv"), device)?;

        let fragment_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&fragment_entry_point);

        let fragment_transmission_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&fragment_transmission_entry_point);

        let vertex_instanced_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&vertex_instanced_entry_point);

        let vertex_depth_pre_pass_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&vertex_depth_pre_pass_entry_point);

        let vertex_depth_pre_pass_alpha_clip_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&vertex_depth_pre_pass_alpha_clip_entry_point);

        let fragment_depth_pre_pass_alpha_clip_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&fragment_depth_pre_pass_alpha_clip_entry_point);

        let fullscreen_tri_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(&fullscreen_tri_entry_point);

        let fragment_tonemap_stage = vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(&fragment_tonemap_entry_point);

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layouts.main])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .size(std::mem::size_of::<shared_structs::PushConstants>() as u32)]),
                None,
            )
        }?;

        let transmission_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.main,
                        descriptor_set_layouts.hdr_framebuffer,
                    ])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .size(std::mem::size_of::<shared_structs::PushConstants>() as u32)]),
                None,
            )
        }?;

        let tonemap_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layouts.hdr_framebuffer])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .size(
                            std::mem::size_of::<colstodian::tonemap::BakedLottesTonemapperParams>()
                                as u32,
                        )]),
                None,
            )
        }?;

        let full_vertex_attributes = &[
            ash_abstractions::VertexAttribute::Vec3,
            ash_abstractions::VertexAttribute::Vec3,
            ash_abstractions::VertexAttribute::Vec2,
            ash_abstractions::VertexAttribute::Uint,
        ];

        let instance_attributes = &[
            ash_abstractions::VertexAttribute::Vec3,
            ash_abstractions::VertexAttribute::Vec3,
            ash_abstractions::VertexAttribute::Vec3,
            ash_abstractions::VertexAttribute::Vec3,
            ash_abstractions::VertexAttribute::Float,
        ];

        let normal_pipeline_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::BACK,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: false,
                depth_compare_op: vk::CompareOp::EQUAL,
            }),
            vertex_attributes: &ash_abstractions::create_vertex_attribute_descriptions(&[
                full_vertex_attributes,
                instance_attributes,
            ]),
            vertex_bindings: &[
                *vk::VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<Vertex>() as u32),
                *vk::VertexInputBindingDescription::builder()
                    .binding(1)
                    .stride(std::mem::size_of::<Instance>() as u32)
                    .input_rate(vk::VertexInputRate::INSTANCE),
            ],
            colour_attachments: &[
                *vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::all())
                    .blend_enable(false),
                *vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::all())
                    .blend_enable(false),
            ],
        };

        let transmission_pipeline_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::BACK,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: vk::CompareOp::EQUAL,
            }),
            vertex_attributes: &ash_abstractions::create_vertex_attribute_descriptions(&[
                full_vertex_attributes,
                instance_attributes,
            ]),
            vertex_bindings: &[
                *vk::VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<Vertex>() as u32),
                *vk::VertexInputBindingDescription::builder()
                    .binding(1)
                    .stride(std::mem::size_of::<Instance>() as u32)
                    .input_rate(vk::VertexInputRate::INSTANCE),
            ],
            colour_attachments: &[*vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)],
        };

        let depth_pre_pass_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::BACK,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: vk::CompareOp::LESS,
            }),
            vertex_attributes: &ash_abstractions::create_vertex_attribute_descriptions(&[
                &[ash_abstractions::VertexAttribute::Vec3],
                instance_attributes,
            ]),
            vertex_bindings: &[
                *vk::VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<Vertex>() as u32),
                *vk::VertexInputBindingDescription::builder()
                    .binding(1)
                    .stride(std::mem::size_of::<Instance>() as u32)
                    .input_rate(vk::VertexInputRate::INSTANCE),
            ],
            colour_attachments: &[],
        };

        let depth_pre_pass_alpha_clip_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::BACK,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: Some(ash_abstractions::DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: vk::CompareOp::LESS,
            }),
            vertex_attributes: &ash_abstractions::create_vertex_attribute_descriptions(&[
                full_vertex_attributes,
                instance_attributes,
            ]),
            vertex_bindings: &[
                *vk::VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<Vertex>() as u32),
                *vk::VertexInputBindingDescription::builder()
                    .binding(1)
                    .stride(std::mem::size_of::<Instance>() as u32)
                    .input_rate(vk::VertexInputRate::INSTANCE),
            ],
            colour_attachments: &[],
        };

        let tonemap_pipeline_desc = ash_abstractions::GraphicsPipelineDescriptor {
            primitive_state: ash_abstractions::PrimitiveState {
                cull_mode: vk::CullModeFlags::NONE,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                polygon_mode: vk::PolygonMode::FILL,
            },
            depth_stencil_state: None,
            vertex_attributes: &[],
            vertex_bindings: &[],
            colour_attachments: &[*vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)],
        };

        let normal_baked = normal_pipeline_desc.as_baked();
        let depth_pre_pass_baked = depth_pre_pass_desc.as_baked();
        let depth_pre_pass_alpha_clip_baked = depth_pre_pass_alpha_clip_desc.as_baked();
        let transmission_baked = transmission_pipeline_desc.as_baked();
        let tonemap_pipeline_baked = tonemap_pipeline_desc.as_baked();

        let stages = &[*vertex_instanced_stage, *fragment_stage];

        let normal_pipeline_desc =
            normal_baked.as_pipeline_create_info(stages, pipeline_layout, render_passes.draw, 1);

        let depth_pre_pass_stage = &[*vertex_depth_pre_pass_stage];

        let depth_pre_pass_desc = depth_pre_pass_baked.as_pipeline_create_info(
            depth_pre_pass_stage,
            pipeline_layout,
            render_passes.draw,
            0,
        );

        let depth_pre_pass_alpha_clip_stages = &[
            *vertex_depth_pre_pass_alpha_clip_stage,
            *fragment_depth_pre_pass_alpha_clip_stage,
        ];

        let depth_pre_pass_alpha_clip_desc = depth_pre_pass_alpha_clip_baked
            .as_pipeline_create_info(
                depth_pre_pass_alpha_clip_stages,
                pipeline_layout,
                render_passes.draw,
                0,
            );

        let transmission_stages = &[*vertex_instanced_stage, *fragment_transmission_stage];

        let transmission_pipeline_desc = transmission_baked.as_pipeline_create_info(
            transmission_stages,
            transmission_pipeline_layout,
            render_passes.transmission,
            0,
        );

        let tonemap_stages = &[*fullscreen_tri_stage, *fragment_tonemap_stage];

        let tonemap_pipeline_desc = tonemap_pipeline_baked.as_pipeline_create_info(
            tonemap_stages,
            tonemap_pipeline_layout,
            render_passes.tonemap,
            0,
        );

        let depth_pre_pass_transmissive_desc = depth_pre_pass_baked.as_pipeline_create_info(
            depth_pre_pass_stage,
            pipeline_layout,
            render_passes.draw,
            2,
        );

        let depth_pre_pass_transmissive_alpha_clip_desc = depth_pre_pass_alpha_clip_baked
            .as_pipeline_create_info(
                depth_pre_pass_alpha_clip_stages,
                pipeline_layout,
                render_passes.draw,
                2,
            );

        let pipelines = unsafe {
            device.create_graphics_pipelines(
                pipeline_cache,
                &[
                    *normal_pipeline_desc,
                    *depth_pre_pass_desc,
                    *depth_pre_pass_alpha_clip_desc,
                    *depth_pre_pass_transmissive_desc,
                    *depth_pre_pass_transmissive_alpha_clip_desc,
                    *transmission_pipeline_desc,
                    *tonemap_pipeline_desc,
                ],
                None,
            )
        }
        .map_err(|(_, err)| err)?;

        Ok(Self {
            normal: pipelines[0],
            depth_pre_pass: pipelines[1],
            depth_pre_pass_alpha_clip: pipelines[2],
            depth_pre_pass_transmissive: pipelines[3],
            depth_pre_pass_transmissive_alpha_clip: pipelines[4],
            transmission: pipelines[5],
            tonemap: pipelines[6],
            pipeline_layout,
            tonemap_pipeline_layout,
            transmission_pipeline_layout,
        })
    }
}

struct Vertex {
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    material: u32,
}

struct Instance {
    translation: Vec3,
    rotation: Mat3,
    scale: f32,
}

struct RenderPasses {
    draw: vk::RenderPass,
    transmission: vk::RenderPass,
    tonemap: vk::RenderPass,
}

impl RenderPasses {
    fn new(device: &ash::Device, surface_format: vk::Format) -> anyhow::Result<Self> {
        let draw_attachments = [
            // Depth buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            // HDR framebuffers
            *vk::AttachmentDescription::builder()
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            *vk::AttachmentDescription::builder()
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];

        let depth_attachment_ref = *vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let hdr_framebuffer_refs = [
            *vk::AttachmentReference::builder()
                .attachment(1)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            *vk::AttachmentReference::builder()
                .attachment(2)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];

        let draw_subpasses = [
            // Depth pre-pass
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .depth_stencil_attachment(&depth_attachment_ref),
            // Colour pass
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&hdr_framebuffer_refs)
                .depth_stencil_attachment(&depth_attachment_ref),
            // Second depth pre-pass for transmissive objects.
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .depth_stencil_attachment(&depth_attachment_ref),
        ];

        let draw_subpass_dependencices = [
            *vk::SubpassDependency::builder()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(1)
                .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags::empty())
                // We use late and not early fragment tests as we handle alpha clipping in the depth pre pass.
                .dst_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            *vk::SubpassDependency::builder()
                .src_subpass(0)
                .dst_subpass(1)
                .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            *vk::SubpassDependency::builder()
                .src_subpass(1)
                .dst_subpass(2)
                .src_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                .dst_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
        ];

        let draw_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&draw_attachments)
                    .subpasses(&draw_subpasses)
                    .dependencies(&draw_subpass_dependencices),
                None,
            )
        }?;

        let tonemap_attachments = [
            // swapchain image
            *vk::AttachmentDescription::builder()
                .format(surface_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        ];

        let swapchain_image_ref = [*vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let tonemap_subpasses = [
            // Tonemapping pass
            *vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&swapchain_image_ref),
        ];

        let tonemap_subpass_dependencies = [*vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        let tonemap_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&tonemap_attachments)
                    .subpasses(&tonemap_subpasses)
                    .dependencies(&tonemap_subpass_dependencies),
                None,
            )
        }?;

        let transmission_attachments = [
            // Depth buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            // HDR framebuffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];

        let depth_attachment_ref = *vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let hdr_framebuffer_ref = [*vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let transmission_subpass = [*vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&hdr_framebuffer_ref)
            .depth_stencil_attachment(&depth_attachment_ref)];

        let transmission_subpass_dependency = [*vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        let transmission_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&transmission_attachments)
                    .subpasses(&transmission_subpass)
                    .dependencies(&transmission_subpass_dependency),
                None,
            )
        }?;

        Ok(Self {
            draw: draw_render_pass,
            tonemap: tonemap_render_pass,
            transmission: transmission_render_pass,
        })
    }
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
            format: vk::Format::R32G32B32A32_SFLOAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | extra_usage,
            next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
            next_layout: vk_sync::ImageLayout::Optimal,
        },
        init_resources,
    )
}

struct IndexBufferSlice {
    offset: u32,
    count: u32,
}

impl IndexBufferSlice {
    fn as_draw_indexed_indirect_command(
        &self,
        first_instance: u32,
    ) -> vk::DrawIndexedIndirectCommand {
        vk::DrawIndexedIndirectCommand {
            index_count: self.count,
            instance_count: 1,
            first_index: self.offset,
            vertex_offset: 0,
            first_instance,
        }
    }
}

struct Model {
    opaque: IndexBufferSlice,
    alpha_clip: IndexBufferSlice,
    transmission: IndexBufferSlice,
    transmission_alpha_clip: IndexBufferSlice,
}

fn load_gltf(
    path: &str,
    init_resources: &mut ash_abstractions::InitResources,
    image_manager: &mut ImageManager,
    buffers_to_cleanup: &mut Vec<ash_abstractions::Buffer>,
    materials: &mut Vec<shared_structs::MaterialInfo>,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) -> anyhow::Result<Model> {
    let (gltf, buffers, mut images) = gltf::import(path)?;

    for image in &mut images {
        if image.format == gltf::image::Format::R8G8B8 {
            let dynamic_image = image::DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(
                    image.width,
                    image.height,
                    std::mem::take(&mut image.pixels),
                )
                .unwrap(),
            );

            let rgba8 = dynamic_image.to_rgba8();

            image.format = gltf::image::Format::R8G8B8A8;
            image.pixels = rgba8.into_raw();
        }
    }

    let node_tree = NodeTree::new(gltf.nodes());

    let mut opaque_indices = Vec::new();
    let mut alpha_clip_indices = Vec::new();
    let mut tranmission_indices = Vec::new();
    let mut transmission_alpha_clip_indices = Vec::new();

    for (node, mesh) in gltf
        .nodes()
        .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
    {
        let transform = node_tree.transform_of(node.index());
        let normal_transform = normal_matrix(transform);

        for primitive in mesh.primitives() {
            let material = primitive.material();

            let indices = match (material.alpha_mode(), material.transmission().is_some()) {
                (gltf::material::AlphaMode::Opaque, false) => &mut opaque_indices,
                (gltf::material::AlphaMode::Mask, false) => &mut alpha_clip_indices,
                (gltf::material::AlphaMode::Opaque, true) => &mut tranmission_indices,
                (gltf::material::AlphaMode::Mask, true) => &mut transmission_alpha_clip_indices,
                (mode, _) => {
                    dbg!(mode);
                    &mut opaque_indices
                }
            };

            // We handle texture transforms, but only scaling and only on the base colour texture.
            // This is the bare minimum to render the SheenCloth correctly.
            let uv_scaling = material
                .pbr_metallic_roughness()
                .base_color_texture()
                .and_then(|info| info.texture_transform())
                .map(|transform| Vec2::from(transform.scale()))
                .unwrap_or(Vec2::ONE);

            let material_id = material.index().unwrap_or(0) + materials.len();

            let reader = primitive.reader(|i| Some(&buffers[i.index()]));

            let read_indices = reader.read_indices().unwrap().into_u32();

            let num_vertices = vertices.len() as u32;

            indices.extend(read_indices.map(|index| index + num_vertices));

            let positions = reader.read_positions().unwrap();
            let normals = reader.read_normals().unwrap();
            let uvs = reader.read_tex_coords(0).unwrap().into_f32();

            for ((position, normal), uv) in positions.zip(normals).zip(uvs) {
                vertices.push(Vertex {
                    position: (transform * Vec3::from(position).extend(1.0)).truncate(),
                    uv: uv_scaling * Vec2::from(uv),
                    normal: normal_transform * Vec3::from(normal),
                    material: material_id as u32,
                });
            }
        }
    }

    let opaque_slice = IndexBufferSlice {
        offset: indices.len() as u32,
        count: opaque_indices.len() as u32,
    };

    indices.extend_from_slice(&opaque_indices);

    let alpha_clip_slice = IndexBufferSlice {
        offset: indices.len() as u32,
        count: alpha_clip_indices.len() as u32,
    };

    indices.extend_from_slice(&alpha_clip_indices);

    let transmission_slice = IndexBufferSlice {
        offset: indices.len() as u32,
        count: tranmission_indices.len() as u32,
    };

    indices.extend_from_slice(&tranmission_indices);

    let transmission_alpha_clip_slice = IndexBufferSlice {
        offset: indices.len() as u32,
        count: transmission_alpha_clip_indices.len() as u32,
    };

    indices.extend_from_slice(&transmission_alpha_clip_indices);

    let mut image_index_to_id = std::collections::HashMap::new();

    for (i, material) in gltf.materials().enumerate() {
        let mut load_optional_texture =
            |optional_texture_info: Option<gltf::texture::Texture>, name: &str, srgb| {
                match optional_texture_info {
                    Some(info) => {
                        let image_index = info.source().index();

                        let id = match image_index_to_id.entry((image_index, srgb)) {
                            std::collections::hash_map::Entry::Occupied(occupied) => {
                                println!("reusing image {} (srgb: {})", image_index, srgb);
                                *occupied.get()
                            }
                            std::collections::hash_map::Entry::Vacant(vacancy) => {
                                let image = &images[image_index];

                                let texture = load_texture_from_gltf(
                                    image,
                                    srgb,
                                    &format!("{} {} {}", path, name, i),
                                    init_resources,
                                    buffers_to_cleanup,
                                )?;

                                let id = image_manager.push_image(texture);

                                vacancy.insert(id);

                                id
                            }
                        };

                        Ok::<_, anyhow::Error>(id as i32)
                    }
                    None => Ok(-1),
                }
            };

        let pbr = material.pbr_metallic_roughness();

        materials.push(shared_structs::MaterialInfo {
            textures: shared_structs::Textures {
                diffuse: load_optional_texture(
                    pbr.base_color_texture().map(|info| info.texture()),
                    "diffuse",
                    true,
                )?,
                metallic_roughness: load_optional_texture(
                    pbr.metallic_roughness_texture().map(|info| info.texture()),
                    "metallic roughness",
                    false,
                )?,
                normal_map: load_optional_texture(
                    material.normal_texture().map(|info| info.texture()),
                    "normal map",
                    false,
                )?,
                emissive: load_optional_texture(
                    material.emissive_texture().map(|info| info.texture()),
                    "emissive",
                    true,
                )?,
                occlusion: load_optional_texture(
                    material.occlusion_texture().map(|info| info.texture()),
                    "occlusion",
                    true,
                )?,
                transmission: load_optional_texture(
                    material
                        .transmission()
                        .and_then(|transmission| transmission.transmission_texture())
                        .map(|info| info.texture()),
                    "transmission",
                    false,
                )?,
            },
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            alpha_clipping_cutoff: material.alpha_cutoff().unwrap_or(0.5),
            diffuse_factor: pbr.base_color_factor().into(),
            emissive_factor: material.emissive_factor().into(),
            normal_map_scale: material
                .normal_texture()
                .map(|normal_texture| normal_texture.scale())
                .unwrap_or_default(),
            occlusion_strength: material
                .occlusion_texture()
                .map(|occlusion| occlusion.strength())
                .unwrap_or(1.0),
            index_of_refraction: material.ior().unwrap_or(1.5),
            transmission_factor: material
                .transmission()
                .map(|transmission| transmission.transmission_factor())
                .unwrap_or(0.0),
        });
    }

    Ok(Model {
        opaque: opaque_slice,
        alpha_clip: alpha_clip_slice,
        transmission: transmission_slice,
        transmission_alpha_clip: transmission_alpha_clip_slice,
    })
}

fn normal_matrix(transform: Mat4) -> Mat3 {
    let inverse_transpose = transform.inverse().transpose();
    Mat3::from_mat4(inverse_transpose)
}

fn mip_levels_for_size(width: u32, height: u32) -> u32 {
    (width.min(height) as f32).log2() as u32 + 1
}

fn load_texture_from_gltf(
    image: &gltf::image::Data,
    srgb: bool,
    name: &str,
    init_resources: &mut ash_abstractions::InitResources,
    buffers_to_cleanup: &mut Vec<ash_abstractions::Buffer>,
) -> anyhow::Result<ash_abstractions::Image> {
    let format = match (image.format, srgb) {
        (gltf::image::Format::R8G8B8A8, true) => vk::Format::R8G8B8A8_SRGB,
        (gltf::image::Format::R8G8B8A8, false) => vk::Format::R8G8B8A8_UNORM,
        format => panic!("unsupported format: {:?}", format),
    };

    let mip_levels = mip_levels_for_size(image.width, image.height);

    let (image, staging_buffer) = ash_abstractions::load_image_from_bytes(
        &ash_abstractions::LoadImageDescriptor {
            bytes: &image.pixels,
            extent: vk::Extent3D {
                width: image.width,
                height: image.height,
                depth: 1,
            },
            view_ty: vk::ImageViewType::TYPE_2D,
            format,
            name,
            next_accesses: &[
                vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
            ],
            next_layout: vk_sync::ImageLayout::Optimal,
            mip_levels,
        },
        init_resources,
    )?;

    buffers_to_cleanup.push(staging_buffer);

    Ok(image)
}

unsafe fn cast_slice<T>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        slice as *const [T] as *const u8,
        slice.len() * std::mem::size_of::<T>(),
    )
}

unsafe fn bytes_of<T>(reference: &T) -> &[u8] {
    std::slice::from_raw_parts(reference as *const T as *const u8, std::mem::size_of::<T>())
}

#[derive(Default)]
pub struct ImageManager {
    images: Vec<ash_abstractions::Image>,
    image_infos: Vec<vk::DescriptorImageInfo>,
}

impl ImageManager {
    pub fn push_image(&mut self, image: ash_abstractions::Image) -> u32 {
        let index = self.images.len() as u32;

        self.image_infos.push(
            *vk::DescriptorImageInfo::builder()
                .image_view(image.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        );

        self.images.push(image);

        index
    }

    pub fn fill_with_dummy_images_up_to(&mut self, items: usize) {
        while self.image_infos.len() < items {
            self.image_infos.push(self.image_infos[0]);
        }
    }

    pub fn write_descriptor_set(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
    ) -> vk::WriteDescriptorSetBuilder {
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&self.image_infos)
    }

    pub fn cleanup(
        &self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> anyhow::Result<()> {
        for image in &self.images {
            image.cleanup(device, allocator)?;
        }

        Ok(())
    }
}

pub struct NodeTree {
    inner: Vec<(Mat4, usize)>,
}

impl NodeTree {
    fn new(nodes: gltf::iter::Nodes) -> Self {
        let mut inner = vec![(Mat4::IDENTITY, usize::max_value()); nodes.clone().count()];

        for node in nodes {
            inner[node.index()].0 = Mat4::from_cols_array_2d(&node.transform().matrix());
            for child in node.children() {
                inner[child.index()].1 = node.index();
            }
        }

        Self { inner }
    }

    pub fn transform_of(&self, mut index: usize) -> Mat4 {
        let mut transform_sum = Mat4::IDENTITY;

        while index != usize::max_value() {
            let (transform, parent_index) = self.inner[index];
            transform_sum = transform * transform_sum;
            index = parent_index;
        }

        transform_sum
    }
}
