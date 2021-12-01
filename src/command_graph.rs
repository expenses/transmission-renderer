use ash::vk;
use thread_pool::{Handle, ThreadPool};

#[derive(Copy, Clone)]
struct CommandPoolBufferPair {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
}

pub struct CommandGraph {
    buffers: Vec<CommandPoolBufferPair>,
    buffer_index: usize,
    nodes: petgraph::Graph<CommandBufferId, ()>,
    handles: Vec<Handle>,
    thread_pool: ThreadPool,
    device: ash::Device,
    graphics_queue_family: u32,
}

impl CommandGraph {
    pub fn new(
        device: &ash::Device,
        thread_pool: ThreadPool,
        graphics_queue_family: u32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            buffers: Vec::new(),
            buffer_index: 0,
            nodes: Default::default(),
            handles: Default::default(),
            device: device.clone(),
            thread_pool,
            graphics_queue_family,
        })
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.buffer_index = 0;
    }

    fn next_command_pair(&mut self) -> anyhow::Result<(usize, CommandPoolBufferPair)> {
        let buffer_id = self.buffer_index;
        self.buffer_index += 1;

        let pair = match self.buffers.get(buffer_id) {
            Some(command_pair) => *command_pair,
            None => {
                let pool = unsafe {
                    self.device.create_command_pool(
                        &vk::CommandPoolCreateInfo::builder()
                            .queue_family_index(self.graphics_queue_family),
                        None,
                    )
                }?;

                let pair = CommandPoolBufferPair {
                    pool,
                    buffer: unsafe {
                        self.device.allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::builder()
                                .command_pool(pool)
                                .level(vk::CommandBufferLevel::PRIMARY)
                                .command_buffer_count(1),
                        )
                    }?[0],
                };

                self.buffers.push(pair);

                pair
            }
        };

        Ok((buffer_id, pair))
    }

    pub fn register_commands<FN: Fn(vk::CommandBuffer) + Send + Sync>(
        &mut self,
        name: &str,
        parents: &[(NodeId, Option<BarrierId>)],
        record_function: FN,
    ) -> anyhow::Result<NodeId> {
        let _span = tracy_client::span!(&format!("Registering commands for {}", name));

        let (buffer_id, command_pair) = self.next_command_pair()?;

        let span = tracy_client::span!("spawning");

        let device = &self.device;

        let task = move || {
            let _span = tracy_client::span!(&format!("Recording commands for {}", name));

            unsafe {
                device.reset_command_pool(command_pair.pool, vk::CommandPoolResetFlags::empty())?;

                device.begin_command_buffer(
                    command_pair.buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;

                record_function(command_pair.buffer);

                device.end_command_buffer(command_pair.buffer)?;
            }

            Ok::<_, anyhow::Error>(())
        };

        let handle = unsafe { self.thread_pool.spawn(task) };

        self.handles.push(handle);

        drop(span);

        let node_id = self.nodes.add_node(CommandBufferId(buffer_id));

        for (parent_node_id, barrier_id) in parents {
            match barrier_id {
                Some(barrier_id) => {
                    self.nodes.update_edge(parent_node_id.0, barrier_id.0, ());
                    self.nodes.update_edge(barrier_id.0, node_id, ());
                }
                None => {
                    self.nodes.update_edge(parent_node_id.0, node_id, ());
                }
            }
        }

        Ok(NodeId(node_id))
    }

    pub fn register_global_barrier(
        &mut self,
        barrier: vk_sync::GlobalBarrier,
    ) -> anyhow::Result<BarrierId> {
        let (buffer_id, command_pair) = self.next_command_pair()?;

        let device = &self.device;
        let barrier = barrier.clone();

        let task = move || {
            let _span = tracy_client::span!("Recording global barrier");

            unsafe {
                device.reset_command_pool(command_pair.pool, vk::CommandPoolResetFlags::empty())?;

                device.begin_command_buffer(
                    command_pair.buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;

                vk_sync::cmd::pipeline_barrier(
                    device,
                    command_pair.buffer,
                    Some(barrier),
                    &[],
                    &[],
                );

                device.end_command_buffer(command_pair.buffer)?;
            }

            Ok::<_, anyhow::Error>(())
        };

        let handle = unsafe { self.thread_pool.spawn(task) };

        self.handles.push(handle);

        let node_id = self.nodes.add_node(CommandBufferId(buffer_id));

        Ok(BarrierId(node_id))
    }

    pub fn get_buffers(&mut self) -> Vec<vk::CommandBuffer> {
        let span = tracy_client::span!("toposort");

        let sort = petgraph::algo::toposort(&self.nodes, None).unwrap();

        let buffers = sort
            .iter()
            .map(|id| self.buffers[self.nodes[*id].0].buffer)
            .collect();

        drop(span);

        for handle in self.handles.drain(..) {
            if let Err(err) = handle.block_on_result() {
                eprintln!("{}", err);
            }
        }

        buffers
    }
}

struct CommandBufferId(usize);
#[derive(Clone, Copy, Debug)]
pub struct NodeId(petgraph::graph::NodeIndex);
#[derive(Clone, Copy)]
pub struct BarrierId(petgraph::graph::NodeIndex);
