use ash::vk;

struct TimestampBuffer<const N: usize> {
    timestamps: [i64; N],
    len: usize,
}

impl<const N: usize> Default for TimestampBuffer<N> {
    fn default() -> Self {
        Self {
            timestamps: [0; N],
            len: 0,
        }
    }
}

impl<const N: usize> TimestampBuffer<N> {
    fn emit(&mut self) {
        for i in 0..self.len {
            tracy_client::emit_gpu_time(self.timestamps[i], 0, i as u16);
        }
    }

    fn next_id(&mut self) -> usize {
        let next_id = self.len;
        self.len += 1;
        next_id
    }
}

pub struct QueryPool {
    pool: vk::QueryPool,
}

impl QueryPool {
    pub fn new(init_resources: &mut ash_abstractions::InitResources) -> anyhow::Result<Self> {
        let pool = unsafe {
            init_resources.device.create_query_pool(
                &vk::QueryPoolCreateInfo::builder()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2048),
                None,
            )
        }?;

        unsafe {
            init_resources.device.cmd_reset_query_pool(
                init_resources.command_buffer,
                pool,
                0,
                2048,
            );
        }

        vk_sync::cmd::pipeline_barrier(
            init_resources.device,
            init_resources.command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::HostWrite],
                next_accesses: &[vk_sync::AccessType::HostWrite],
            }),
            &[],
            &[],
        );

        unsafe {
            init_resources.device.cmd_write_timestamp(
                init_resources.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                pool,
                0,
            )
        }

        Ok(Self { pool })
    }

    pub fn into_profiling_context(
        self,
        device: &ash::Device,
        limits: vk::PhysicalDeviceLimits,
    ) -> anyhow::Result<ProfilingContext> {
        let mut initial_timestamp = [0_i64];

        unsafe {
            device.get_query_pool_results(
                self.pool,
                0,
                1,
                &mut initial_timestamp,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::TYPE_64,
            )?;
        }

        tracy_client::emit_new_gpu_context(
            initial_timestamp[0],
            limits.timestamp_period,
            0,
            tracy_client::GpuContextType::Vulkan,
            None,
        );

        Ok(ProfilingContext {
            buffer: TimestampBuffer {
                timestamps: [0; 4096],
                // As it contains the initial timestamp.
                len: 1,
            },
            pool: self.pool,
            can_reset: true,
        })
    }
}

pub struct ProfilingContext {
    buffer: TimestampBuffer<4096>,
    pool: vk::QueryPool,
    can_reset: bool,
}

impl ProfilingContext {
    /// Must be called before using for the first time and between collecting and recording zones. Ideally you call this immediately after starting a command buffer each frame.
    pub fn reset(&mut self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        if !self.can_reset {
            return;
        }

        unsafe {
            device.cmd_reset_query_pool(command_buffer, self.pool, 0, self.buffer.len as u32);
        }

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::HostWrite],
                next_accesses: &[vk_sync::AccessType::HostWrite],
            }),
            &[],
            &[],
        );

        self.buffer.len = 0;
    }

    // Must be called once per frame outside of a command buffer. Ideally as the last thing per frame.
    pub fn collect(&mut self, device: &ash::Device) -> anyhow::Result<()> {
        let res = unsafe {
            device.get_query_pool_results(
                self.pool,
                0,
                self.buffer.len as u32,
                &mut self.buffer.timestamps[..self.buffer.len],
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::TYPE_64,
            )
        };

        // Even though we're setting the WAIT flag, get_query_pool_results still seems
        // to sometimes return NOT_READY occasionally. In this case we just ignore it.
        self.can_reset = if res.is_ok() {
            self.buffer.emit();
            true
        } else {
            println!("vkGetQueryResults just returned NOT_READY illegally");
            false
        };

        Ok(())
    }
}

#[macro_export]
macro_rules! profiling_zone {
    ($name:expr, $device:expr, $command_buffer:expr, $context:expr) => {
        profiling_zone!(
            $name,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            $device,
            $command_buffer,
            $context
        )
    };
    ($name:expr, $start_stage:expr, $end_stage:expr, $device:expr, $command_buffer:expr, $context:expr) => {{
        struct S;
        let func_name = std::any::type_name::<S>();
        $crate::profiling::ProfilingZone::new(
            &$crate::profiling::SourceContext {
                name: $name,
                function: &func_name[..func_name.len() - 3],
                file: file!(),
                line: line!(),
            },
            $start_stage,
            $end_stage,
            $device,
            $command_buffer,
            $context,
        )
    }};
}

pub struct SourceContext<'a> {
    pub name: &'a str,
    pub function: &'a str,
    pub file: &'a str,
    pub line: u32,
}

pub struct ProfilingZone {
    _gpu_zone: tracy_client::GpuZone,
    pool: vk::QueryPool,
    device: ash::Device,
    command_buffer: vk::CommandBuffer,
    end_query_id: u16,
    end_stage: vk::PipelineStageFlags,
}

impl ProfilingZone {
    pub fn new(
        source_context: &SourceContext,
        start_stage: vk::PipelineStageFlags,
        end_stage: vk::PipelineStageFlags,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        context: &mut ProfilingContext,
    ) -> Self {
        let start_query_id = context.buffer.next_id() as u16;
        let end_query_id = context.buffer.next_id() as u16;

        unsafe {
            device.cmd_write_timestamp(
                command_buffer,
                start_stage,
                context.pool,
                start_query_id as u32,
            );
        }

        Self {
            _gpu_zone: tracy_client::GpuZone::new(
                source_context.name,
                source_context.function,
                source_context.file,
                source_context.line,
                start_query_id,
                end_query_id,
                0,
            ),
            pool: context.pool,
            device: device.clone(),
            command_buffer,
            end_query_id,
            end_stage,
        }
    }
}

impl Drop for ProfilingZone {
    fn drop(&mut self) {
        unsafe {
            self.device.cmd_write_timestamp(
                self.command_buffer,
                self.end_stage,
                self.pool,
                self.end_query_id as u32,
            );
        }
    }
}
