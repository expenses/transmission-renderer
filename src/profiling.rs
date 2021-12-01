use ash::vk;
use std::sync::atomic::{AtomicU16, Ordering};

const POOL_SIZE: u32 = 4096;

fn emit(timestamps: &[i64]) {
    for (i, timestamp) in timestamps.iter().enumerate() {
        tracy_client::emit_gpu_time(*timestamp, 0, i as u16);
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
                    .query_count(POOL_SIZE),
                None,
            )
        }?;

        unsafe {
            init_resources.device.cmd_reset_query_pool(
                init_resources.command_buffer,
                pool,
                0,
                POOL_SIZE,
            );

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
            num_written_timestamps: Default::default(),
            pool: self.pool,
            // As it contains the initial timestamp.
            timestamps_to_reset: AtomicU16::new(1),
        })
    }
}

pub struct ProfilingContext {
    num_written_timestamps: AtomicU16,
    pool: vk::QueryPool,
    timestamps_to_reset: AtomicU16,
}

impl ProfilingContext {
    /// Must be called before using for the first time and between collecting and recording zones. Ideally you call this immediately after starting a command buffer each frame.
    pub fn reset(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        let timestamps_to_reset = self.timestamps_to_reset.load(Ordering::Relaxed);

        if timestamps_to_reset == 0 {
            return;
        }

        unsafe {
            device.cmd_reset_query_pool(command_buffer, self.pool, 0, timestamps_to_reset as u32);
        }
    }

    // Must be called once per frame outside of a command buffer. Ideally as the last thing per frame.
    pub fn collect(&self, device: &ash::Device) -> anyhow::Result<()> {
        let num_timestamps = self.num_written_timestamps.load(Ordering::Acquire);

        let mut buffer = [0; POOL_SIZE as usize];
        let timestamps = &mut buffer[..num_timestamps as usize];

        let res = unsafe {
            device.get_query_pool_results(
                self.pool,
                0,
                num_timestamps as u32,
                timestamps,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::TYPE_64,
            )
        };

        // Even though we're setting the WAIT flag, get_query_pool_results still seems
        // to sometimes return NOT_READY occasionally. In this case we just ignore it.
        if res.is_ok() {
            emit(timestamps);

            self.num_written_timestamps.store(0, Ordering::Release);
            self.timestamps_to_reset
                .store(num_timestamps, Ordering::Relaxed);
        } else {
            println!("vkGetQueryResults just returned NOT_READY illegally");
            self.timestamps_to_reset.store(0, Ordering::Relaxed);
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
            tracy_client::span!($name),
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
    _span: tracy_client::Span,
}

impl ProfilingZone {
    pub fn new(
        source_context: &SourceContext,
        start_stage: vk::PipelineStageFlags,
        end_stage: vk::PipelineStageFlags,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        context: &ProfilingContext,
        span: tracy_client::Span,
    ) -> Self {
        let start_query_id = context
            .num_written_timestamps
            .fetch_add(2, Ordering::Relaxed);
        let end_query_id = start_query_id + 1;

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
            _span: span,
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
