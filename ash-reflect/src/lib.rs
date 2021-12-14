use ash::vk;
use std::ffi::CString;
use rspirv_reflect::rspirv::spirv;
use std::collections::BTreeMap;

type DescriptorSet = BTreeMap<u32, rspirv_reflect::DescriptorInfo>;

pub struct ShaderReflection {
    entry_point: CString,
    name: String,
    shader_stage: vk::ShaderStageFlags,
    descriptor_sets: BTreeMap<u32, DescriptorSet>,
    push_constant_range: Option<rspirv_reflect::PushConstantInfo>,
}

impl ShaderReflection {
    pub fn new(bytes: &[u8]) -> anyhow::Result<Self> {
        let reflection = rspirv_reflect::Reflection::new_from_spirv(bytes).map_err(|err| anyhow::anyhow!("{}", err))?;

        let entry_point_inst = &reflection.0.entry_points[0];

        let execution_model = entry_point_inst.operands[0].unwrap_execution_model();
        let name = entry_point_inst.operands[2].unwrap_literal_string();

        let shader_stage = match execution_model {
            spirv::ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
            spirv::ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
            spirv::ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
            other => unimplemented!("{:?}", other)
        };

        Ok(Self {
            entry_point: CString::new(name)?,
            shader_stage,
            descriptor_sets: reflection.get_descriptor_sets().map_err(|err| anyhow::anyhow!("{}", err))?,
            push_constant_range: reflection.get_push_constant_range().map_err(|err| anyhow::anyhow!("{}", err))?,
            name: name.into(),
        })
    }
}

pub struct ShaderModule {
    vk_module: vk::ShaderModule,
    reflection: ShaderReflection,
}

impl ShaderModule {
    pub fn new(device: &ash::Device, bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(Self {
            vk_module: {
                let spv = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
                unsafe {
                    device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spv), None)
                }?
            },
            reflection: ShaderReflection::new(bytes)?,
        })
    }

    pub fn as_stage_create_info(&self) -> vk::PipelineShaderStageCreateInfoBuilder {
        vk::PipelineShaderStageCreateInfo::builder()
            .module(self.vk_module)
            .name(&self.reflection.entry_point)
            .stage(self.reflection.shader_stage)
    }
}

#[derive(Debug)]
struct DescriptorInfo {
    shader_stages: vk::ShaderStageFlags,
    inner: rspirv_reflect::DescriptorInfo,
}

#[derive(Default, Debug)]
pub struct PoolSizes {
    map: BTreeMap<vk::DescriptorType, u32>,
}

impl PoolSizes {
    pub fn add(&mut self, ty: vk::DescriptorType, count: u32) {
        *self.map.entry(ty).or_insert(0) += count;
    }

    pub fn as_vec(&self) -> Vec<vk::DescriptorPoolSize> {
        self.map.iter().map(|(ty, count)| {
            vk::DescriptorPoolSize {
                ty: *ty,
                descriptor_count: *count
            }
        }).collect()
    }
}

#[derive(Debug)]
struct DescriptorSetLayout {
    bindings: BTreeMap<u32, DescriptorInfo>,
}

impl DescriptorSetLayout {
    fn new(shader_stage: vk::ShaderStageFlags, descriptor_set: &DescriptorSet) -> Self {
        Self {
            bindings: descriptor_set.iter().map(|(id, descriptor)| {
                (*id, DescriptorInfo {
                    shader_stages: shader_stage,
                    inner: descriptor.clone(),
                })
            }).collect()
        }
    }

    fn merge(&mut self, shader_stage: vk::ShaderStageFlags, descriptor_set: &DescriptorSet) -> bool {
        let is_same = descriptor_set.iter().all(|(id, descriptor)| {
            match self.bindings.get(id) {
                Some(info) => &info.inner == descriptor,
                None => true
            }
        });

        if is_same {
            for (id, descriptor) in descriptor_set {
                let entry = self.bindings.entry(*id).or_insert_with(|| DescriptorInfo {
                    shader_stages: shader_stage,
                    inner: descriptor.clone(),
                });
                entry.shader_stages |= shader_stage;
            }
        }

        is_same
    }

    pub fn add_to_pool(&self, pool_sizes: &mut PoolSizes, max_unbounded_bindings: u32) {
        for descriptor in self.bindings.values() {
            let count = match descriptor.inner.binding_count {
                rspirv_reflect::BindingCount::One => 1,
                rspirv_reflect::BindingCount::StaticSized(size) => size as u32,
                rspirv_reflect::BindingCount::Unbounded => max_unbounded_bindings
            };

            *pool_sizes.map.entry(
                vk::DescriptorType::from_raw(descriptor.inner.ty.0 as i32)
            ).or_insert(0) += count;
        }
    }

    fn build(&self, device: &ash::Device, max_unbounded_bindings: u32, unbounded_partially_bound: bool) -> anyhow::Result<vk::DescriptorSetLayout> {
        let mut flags = vec![vk::DescriptorBindingFlags::empty(); self.bindings.len()];

        let bindings = self.bindings.iter().map(|(&binding_id, binding)| {
            vk::DescriptorSetLayoutBinding {
                binding: binding_id,
                descriptor_type: vk::DescriptorType::from_raw(binding.inner.ty.0 as i32),
                stage_flags: binding.shader_stages,
                descriptor_count: match binding.inner.binding_count {
                    rspirv_reflect::BindingCount::One => 1,
                    rspirv_reflect::BindingCount::StaticSized(size) => size as u32,
                    rspirv_reflect::BindingCount::Unbounded => {
                        if unbounded_partially_bound {
                            flags[binding_id as usize] |= vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        }

                        max_unbounded_bindings
                    }
                },
                ..Default::default()
            }
        }).collect::<Vec<_>>();

        let mut create_info =vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings);

        let mut flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&flags);

        if unbounded_partially_bound {
            create_info = create_info.push_next(&mut flags);
        }

        Ok(unsafe {
            device.create_descriptor_set_layout(&create_info, None)
        }?)
    }
}

#[derive(Debug, Default)]
pub struct DescriptorSetLayouts {
    mapping: BTreeMap<(String, u32), usize>,
    layouts: Vec<DescriptorSetLayout>,
}

impl DescriptorSetLayouts {
    pub fn passthrough(&mut self, shader_module: ShaderModule) -> ShaderModule {
        self.merge_from_module(&shader_module);
        shader_module
    }

    pub fn merge_from_reflection(&mut self, reflection: &ShaderReflection) {
        for (set_id, descriptor_set) in &reflection.descriptor_sets {
            let merged = self.layouts.iter_mut().enumerate().find_map(|(layout_index, layout)| {
                if layout.merge(reflection.shader_stage, descriptor_set) {
                    Some(layout_index)
                } else {
                    None
                }
            });

            let layout_index = match merged {
                Some(layout_index) => layout_index,
                None => {
                    let layout_index = self.layouts.len();
                    self.layouts.push(DescriptorSetLayout::new(reflection.shader_stage, descriptor_set));
                    layout_index
                }
            };

            self.mapping.insert((reflection.name.clone(), *set_id), layout_index);
        }
    }

    pub fn merge_from_module(&mut self, shader_module: &ShaderModule) {
        self.merge_from_reflection(&shader_module.reflection)
    }

    pub fn get_pool_sizes(&self, max_unbounded_bindings: u32) -> PoolSizes {
        let mut pool_sizes = PoolSizes::default();

        for layout in &self.layouts {
            layout.add_to_pool(&mut pool_sizes, max_unbounded_bindings);
        }

        pool_sizes
    }

    pub fn build(self, device: &ash::Device, max_unbounded_bindings: u32, unbounded_partially_bound: bool) -> anyhow::Result<BuiltDescriptorSetLayouts> {
        Ok(BuiltDescriptorSetLayouts {
            built_layouts: self.layouts.iter().map(|layout| layout.build(device, max_unbounded_bindings, unbounded_partially_bound)).collect::<anyhow::Result<Vec<_>>>()?,
            mapping: self.mapping
        })
    }
}

#[derive(Debug, Default)]
pub struct BuiltDescriptorSetLayouts {
    mapping: BTreeMap<(String, u32), usize>,
    built_layouts: Vec<vk::DescriptorSetLayout>,
}

impl BuiltDescriptorSetLayouts {
   pub fn layout_for_shader(&self, shader_name: &str, set_id: u32) -> vk::DescriptorSetLayout {
        match self.mapping.get(&(shader_name.into(), set_id)) {
            Some(index) => self.built_layouts[*index],
            None => panic!("Could not find descriptor set {} for shader {}", set_id, shader_name)
        }
    }
}
