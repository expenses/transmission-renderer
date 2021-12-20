use ash::vk;
use rspirv_reflect::rspirv::spirv;
use std::collections::BTreeMap;
use std::ffi::CString;

type DescriptorSet = BTreeMap<u32, rspirv_reflect::DescriptorInfo>;

pub struct ShaderReflection {
    entry_point: CString,
    name: String,
    shader_stage: vk::ShaderStageFlags,
    descriptor_sets: BTreeMap<u32, DescriptorSet>,
    push_constant_size: Option<u32>,
}

impl ShaderReflection {
    pub fn new(bytes: &[u8]) -> anyhow::Result<Self> {
        let reflection = rspirv_reflect::Reflection::new_from_spirv(bytes)
            .map_err(|err| anyhow::anyhow!("{}", err))?;

        let entry_point_inst = &reflection.0.entry_points[0];

        let execution_model = entry_point_inst.operands[0].unwrap_execution_model();
        let name = entry_point_inst.operands[2].unwrap_literal_string();

        let shader_stage = match execution_model {
            spirv::ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
            spirv::ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
            spirv::ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
            other => unimplemented!("{:?}", other),
        };

        Ok(Self {
            entry_point: CString::new(name)?,
            shader_stage,
            descriptor_sets: reflection
                .get_descriptor_sets()
                .map_err(|err| anyhow::anyhow!("{}", err))?,
            push_constant_size: reflection
                .get_push_constant_range()
                .map_err(|err| anyhow::anyhow!("{}", err))?
                .map(|range| range.size),
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
                    device.create_shader_module(
                        &vk::ShaderModuleCreateInfo::builder().code(&spv),
                        None,
                    )
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
    inner: BTreeMap<vk::DescriptorType, u32>,
}

impl PoolSizes {
    pub fn as_vec(&self) -> Vec<vk::DescriptorPoolSize> {
        self.inner
            .iter()
            .map(|(ty, count)| vk::DescriptorPoolSize {
                ty: *ty,
                descriptor_count: *count,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub max_unbounded_descriptors: u32,
    pub enable_partially_bound_unbounded_descriptors: bool,
}

#[derive(Debug)]
struct DescriptorSetLayout {
    bindings: BTreeMap<u32, DescriptorInfo>,
}

impl DescriptorSetLayout {
    fn new(shader_stage: vk::ShaderStageFlags, descriptor_set: &DescriptorSet) -> Self {
        Self {
            bindings: descriptor_set
                .iter()
                .map(|(id, descriptor)| {
                    (
                        *id,
                        DescriptorInfo {
                            shader_stages: shader_stage,
                            inner: descriptor.clone(),
                        },
                    )
                })
                .collect(),
        }
    }

    fn merge_priority(&self, descriptor_set: &DescriptorSet) -> Option<u32> {
        let mut matching_descriptors = 0;

        for (id, descriptor) in descriptor_set {
            match self.bindings.get(id) {
                Some(info) => {
                    if &info.inner == descriptor {
                        matching_descriptors += 1;
                    } else {
                        return None;
                    }
                }
                None => {}
            }
        }

        Some(matching_descriptors)
    }

    fn merge(&mut self, shader_stage: vk::ShaderStageFlags, descriptor_set: &DescriptorSet) {
        for (id, descriptor) in descriptor_set {
            let entry = self.bindings.entry(*id).or_insert_with(|| DescriptorInfo {
                shader_stages: shader_stage,
                inner: descriptor.clone(),
            });
            entry.shader_stages |= shader_stage;
        }
    }

    pub fn add_to_pool(&self, pool_sizes: &mut PoolSizes, settings: Settings) {
        for descriptor in self.bindings.values() {
            let count = match descriptor.inner.binding_count {
                rspirv_reflect::BindingCount::One => 1,
                rspirv_reflect::BindingCount::StaticSized(size) => size as u32,
                rspirv_reflect::BindingCount::Unbounded => settings.max_unbounded_descriptors,
            };

            *pool_sizes
                .inner
                .entry(vk::DescriptorType::from_raw(descriptor.inner.ty.0 as i32))
                .or_insert(0) += count;
        }
    }

    fn build(
        &self,
        device: &ash::Device,
        settings: Settings,
    ) -> anyhow::Result<vk::DescriptorSetLayout> {
        let mut flags = vec![vk::DescriptorBindingFlags::empty(); self.bindings.len()];

        let bindings = self
            .bindings
            .iter()
            .map(|(&binding_id, binding)| vk::DescriptorSetLayoutBinding {
                binding: binding_id,
                descriptor_type: vk::DescriptorType::from_raw(binding.inner.ty.0 as i32),
                stage_flags: binding.shader_stages,
                descriptor_count: match binding.inner.binding_count {
                    rspirv_reflect::BindingCount::One => 1,
                    rspirv_reflect::BindingCount::StaticSized(size) => size as u32,
                    rspirv_reflect::BindingCount::Unbounded => {
                        if settings.enable_partially_bound_unbounded_descriptors {
                            flags[binding_id as usize] |=
                                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        }

                        settings.max_unbounded_descriptors
                    }
                },
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let mut create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

        let mut flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&flags);

        if settings.enable_partially_bound_unbounded_descriptors {
            create_info = create_info.push_next(&mut flags);
        }

        Ok(unsafe { device.create_descriptor_set_layout(&create_info, None) }?)
    }
}

#[derive(Debug)]
struct ShaderInfo {
    set_id_to_index: BTreeMap<u32, usize>,
    shader_stage: vk::ShaderStageFlags,
    push_constant_size: Option<u32>,
}

#[derive(Debug, Default)]
pub struct DescriptorSetLayouts {
    mapping: BTreeMap<String, ShaderInfo>,
    layouts: Vec<DescriptorSetLayout>,
}

impl DescriptorSetLayouts {
    pub fn load_and_merge_module(
        &mut self,
        device: &ash::Device,
        bytes: &[u8],
    ) -> anyhow::Result<ShaderModule> {
        let module = ShaderModule::new(device, bytes)?;

        self.merge_from_module(&module);

        Ok(module)
    }

    pub fn merge_from_reflection(&mut self, reflection: &ShaderReflection) {
        let mut shader_info = ShaderInfo {
            set_id_to_index: BTreeMap::new(),
            shader_stage: reflection.shader_stage,
            push_constant_size: reflection.push_constant_size,
        };

        for (set_id, descriptor_set) in &reflection.descriptor_sets {
            let layout_to_merge = self
                .layouts
                .iter_mut()
                .enumerate()
                .filter_map(|(layout_index, layout)| {
                    layout
                        .merge_priority(descriptor_set)
                        .map(|merge_priority| (merge_priority, layout, layout_index))
                })
                .max_by_key(|(merge_priority, ..)| *merge_priority);

            let layout_index = match layout_to_merge {
                Some((_, layout, layout_index)) => {
                    layout.merge(reflection.shader_stage, descriptor_set);
                    layout_index
                }
                None => {
                    let layout_index = self.layouts.len();
                    self.layouts.push(DescriptorSetLayout::new(
                        reflection.shader_stage,
                        descriptor_set,
                    ));
                    layout_index
                }
            };

            shader_info.set_id_to_index.insert(*set_id, layout_index);
        }

        self.mapping.insert(reflection.name.clone(), shader_info);
    }

    pub fn merge_from_module(&mut self, shader_module: &ShaderModule) {
        self.merge_from_reflection(&shader_module.reflection)
    }

    pub fn build(
        self,
        device: &ash::Device,
        settings: Settings,
    ) -> anyhow::Result<BuiltDescriptorSetLayouts> {
        Ok(BuiltDescriptorSetLayouts {
            built_layouts: self
                .layouts
                .iter()
                .map(|layout| layout.build(device, settings))
                .collect::<anyhow::Result<Vec<_>>>()?,
            layouts: self.layouts,
            mapping: self.mapping,
            build_settings: settings,
        })
    }
}

pub struct OwnedPipelineLayoutCreateInfo {
    set_layouts: Vec<vk::DescriptorSetLayout>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
}

impl OwnedPipelineLayoutCreateInfo {
    pub fn as_ref(&self) -> vk::PipelineLayoutCreateInfoBuilder {
        vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&self.set_layouts)
            .push_constant_ranges(&self.push_constant_ranges)
    }
}

#[derive(Debug)]
pub struct BuiltDescriptorSetLayouts {
    mapping: BTreeMap<String, ShaderInfo>,
    layouts: Vec<DescriptorSetLayout>,
    built_layouts: Vec<vk::DescriptorSetLayout>,
    build_settings: Settings,
}

impl BuiltDescriptorSetLayouts {
    pub fn pipeline_layout_for_shaders(
        &self,
        shader_names: &[&str],
    ) -> anyhow::Result<OwnedPipelineLayoutCreateInfo> {
        let mut set_layouts = Vec::new();
        let mut push_constant_range = None;

        for shader_name in shader_names {
            let info = self
                .mapping
                .get(*shader_name)
                .ok_or_else(|| anyhow::anyhow!("Could not find shader {}", shader_name))?;

            for (&set_id, &layout_index) in &info.set_id_to_index {
                while set_layouts.len() <= set_id as usize {
                    set_layouts.push(vk::DescriptorSetLayout::null());
                }
                set_layouts[set_id as usize] = self.built_layouts[layout_index];
            }

            push_constant_range = match (info.push_constant_size, push_constant_range) {
                (None, None) => None,
                (Some(size), None) => Some(vk::PushConstantRange {
                    stage_flags: info.shader_stage,
                    size,
                    offset: 0,
                }),
                (None, Some(range)) => Some(range),
                (Some(size), Some(range)) => Some(vk::PushConstantRange {
                    stage_flags: range.stage_flags | info.shader_stage,
                    size: range.size.max(size),
                    offset: 0,
                }),
            };
        }

        for (i, set_layout) in set_layouts.iter().enumerate() {
            if *set_layout == vk::DescriptorSetLayout::null() {
                panic!("Set layout {} is missing", i);
            }
        }

        let mut push_constant_ranges = Vec::new();

        if let Some(push_constant_range) = push_constant_range {
            push_constant_ranges.push(push_constant_range);
        }

        Ok(OwnedPipelineLayoutCreateInfo {
            set_layouts,
            push_constant_ranges,
        })
    }

    pub fn layout_for_shader(
        &self,
        shader_name: &str,
        set_id: u32,
    ) -> anyhow::Result<FetchedDescriptorSetLayout> {
        match self.mapping.get(shader_name) {
            Some(shader_info) => match shader_info.set_id_to_index.get(&set_id) {
                Some(index) => Ok(FetchedDescriptorSetLayout {
                    inner: self.built_layouts[*index],
                    index: *index,
                }),
                None => Err(anyhow::anyhow!(
                    "Could not find set id {} for shader {}",
                    set_id,
                    shader_name
                )),
            },
            None => Err(anyhow::anyhow!("Could not find shader {}", shader_name)),
        }
    }

    pub fn get_pool_sizes(&self, fetched_layouts: &[FetchedDescriptorSetLayout]) -> PoolSizes {
        let mut pool_sizes = PoolSizes::default();

        for fetched_layout in fetched_layouts {
            let layout = &self.layouts[fetched_layout.index];
            layout.add_to_pool(&mut pool_sizes, self.build_settings);
        }

        pool_sizes
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FetchedDescriptorSetLayout {
    inner: vk::DescriptorSetLayout,
    index: usize,
}

impl std::ops::Deref for FetchedDescriptorSetLayout {
    type Target = vk::DescriptorSetLayout;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
