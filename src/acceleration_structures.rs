use crate::{ModelBuffers, ModelStagingBuffers};
use ash::extensions::khr::AccelerationStructure as AccelerationStructureLoader;
use ash::vk;
use glam::Vec3;

pub(crate) fn build_acceleration_structures_from_primitives(
    model_buffers: &ModelBuffers,
    model_staging_buffers: &ModelStagingBuffers,
    acceleration_structure_properties: &vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    acceleration_structure_loader: &AccelerationStructureLoader,
    init_resources: &mut ash_abstractions::InitResources,
    buffers_to_cleanup: &mut Vec<ash_abstractions::Buffer>,
) -> anyhow::Result<Vec<AccelerationStructure>> {
    let triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: model_buffers.position.device_address(init_resources.device),
        })
        .vertex_stride(std::mem::size_of::<Vec3>() as u64)
        .max_vertex(model_staging_buffers.position.len() as u32)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: model_buffers.index.device_address(init_resources.device),
        });

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: *triangles_data,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let geometries = &[*geometry];

    let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(geometries)
        .build();

    let mut build_geometry_infos = Vec::new();
    let mut build_flat_ranges = Vec::new();

    let mut acceleration_structures = Vec::new();

    for (i, primitive) in model_staging_buffers.primitives.iter().enumerate() {
        let primitive_count = primitive.index_count / 3;

        // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkAccelerationStructureBuildRangeInfoKHR.html#_description
        let range = vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(primitive_count)
            .primitive_offset(primitive.first_index * 4);

        let build_sizes = unsafe {
            acceleration_structure_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &geometry_info,
                &[primitive_count],
            )
        };

        let scratch_buffer = ash_abstractions::Buffer::new_of_size_with_alignment(
            build_sizes.build_scratch_size,
            &format!("Scratch buffer for primitive {}", i),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            acceleration_structure_properties.min_acceleration_structure_scratch_offset_alignment
                as u64,
            init_resources,
        )?;

        let acceleration_structure = AccelerationStructure::new(
            build_sizes.acceleration_structure_size,
            &format!("Acceleration structure for primitive {}", i),
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            acceleration_structure_loader,
            init_resources,
        )?;

        let mut complete_geometry_info = geometry_info;
        complete_geometry_info.dst_acceleration_structure = acceleration_structure.object;
        complete_geometry_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_address(init_resources.device),
        };

        build_geometry_infos.push(complete_geometry_info);
        build_flat_ranges.push(*range);
        acceleration_structures.push(acceleration_structure);

        buffers_to_cleanup.push(scratch_buffer);
    }

    let build_ranges: Vec<_> = (0..build_flat_ranges.len())
        .map(|i| &build_flat_ranges[i..i + 1])
        .collect();

    unsafe {
        acceleration_structure_loader.cmd_build_acceleration_structures(
            init_resources.command_buffer,
            &build_geometry_infos,
            &build_ranges,
        );
    }

    Ok(acceleration_structures)
}

pub fn build_top_level_acceleration_structure_from_instances(
    instances_buffer: &ash_abstractions::Buffer,
    num_instances: u32,
    acceleration_structure_properties: &vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    acceleration_structure_loader: &AccelerationStructureLoader,
    init_resources: &mut ash_abstractions::InitResources,
    buffers_to_cleanup: &mut Vec<ash_abstractions::Buffer>,
) -> anyhow::Result<AccelerationStructure> {
    let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::builder().data(
        vk::DeviceOrHostAddressConstKHR {
            device_address: instances_buffer.device_address(init_resources.device),
        },
    );

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: *instances_data,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let range =
        *vk::AccelerationStructureBuildRangeInfoKHR::builder().primitive_count(num_instances);

    let geometries = &[*geometry];

    let mut geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(geometries);

    let build_sizes = unsafe {
        acceleration_structure_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &[num_instances],
        )
    };

    let scratch_buffer = ash_abstractions::Buffer::new_of_size_with_alignment(
        build_sizes.build_scratch_size,
        "Scratch buffer for the instances",
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        acceleration_structure_properties.min_acceleration_structure_scratch_offset_alignment
            as u64,
        init_resources,
    )?;

    let acceleration_structure = AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        "Acceleration structure for instances",
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        &acceleration_structure_loader,
        init_resources,
    )?;

    geometry_info = geometry_info
        .dst_acceleration_structure(acceleration_structure.object)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_address(init_resources.device),
        });
    let geometry_info = *geometry_info;

    let range = &[range][..];
    let range = &[range][..];
    let geometry_infos = &[geometry_info];

    unsafe {
        acceleration_structure_loader.cmd_build_acceleration_structures(
            init_resources.command_buffer,
            geometry_infos,
            range,
        );
    }

    buffers_to_cleanup.push(scratch_buffer);

    Ok(acceleration_structure)
}

pub struct AccelerationStructure {
    pub buffer: ash_abstractions::Buffer,
    object: vk::AccelerationStructureKHR,
}

impl AccelerationStructure {
    fn new(
        size: u64,
        name: &str,
        ty: vk::AccelerationStructureTypeKHR,
        acceleration_structure_loader: &AccelerationStructureLoader,
        init_resources: &mut ash_abstractions::InitResources,
    ) -> anyhow::Result<Self> {
        let buffer = ash_abstractions::Buffer::new_of_size(
            size,
            name,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            init_resources,
        )?;

        let object = unsafe {
            acceleration_structure_loader.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::builder()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .size(size)
                    .ty(ty),
                None,
            )
        }?;

        if let Some(debug_utils_loader) = init_resources.debug_utils_loader {
            ash_abstractions::set_object_name(
                init_resources.device,
                debug_utils_loader,
                object,
                name,
            )?;
        }

        Ok(Self { object, buffer })
    }
}
