use ash::vk;
use std::sync::atomic::{AtomicUsize, Ordering};
use thread_pool::{Handle, ThreadPool};

#[derive(Copy, Clone)]
struct CommandPoolBufferPair {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
}

pub struct CommandGraph {
    buffers: Vec<CommandPoolBufferPair>,
    buffer_index: AtomicUsize,
    nodes: petgraph::Graph<CommandBufferId, ()>,
    handles: Vec<Handle>,
}

impl CommandGraph {
    pub fn new(device: &ash::Device, num_command_buffers: u32, graphics_queue_family: u32) -> anyhow::Result<Self> {
        Ok(Self {
            buffers: (0 .. num_command_buffers).map(|_| {
                let pool = unsafe {
                    device.create_command_pool(
                        &vk::CommandPoolCreateInfo::builder().queue_family_index(graphics_queue_family),
                        None,
                    )
                }?;

                Ok(CommandPoolBufferPair {
                    pool,
                    buffer: unsafe {
                        device.allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::builder()
                                .command_pool(pool)
                                .level(vk::CommandBufferLevel::PRIMARY)
                                .command_buffer_count(1),
                        )
                    }?[0]
                })
            }).collect::<anyhow::Result<_>>()?,
            buffer_index: Default::default(),
            nodes: Default::default(),
            handles: Default::default(),
        })
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.buffer_index.store(0, Ordering::Relaxed);
    }

    pub fn register_commands<'a, FN: Fn(vk::CommandBuffer) + Send + Sync + 'a>(&mut self, name: &str, parents: &[(NodeId, Option<BarrierId>)], device: &'a ash::Device, record_function: FN, thread_pool: &ThreadPool) -> anyhow::Result<NodeId> {
        let _span = tracy_client::span!(&format!("Registering commands for {}", name));

        let buffer_id = self.buffer_index.fetch_add(1, Ordering::Relaxed);

        let command_pair = self.buffers[buffer_id];

        let span = tracy_client::span!("spawning");

        self.handles.push(thread_pool.spawn(move || {
            let _span = tracy_client::span!(&format!("Recording commands for {}", name));

            unsafe {
                device.reset_command_pool(
                    command_pair.pool,
                    vk::CommandPoolResetFlags::empty(),
                )?;

                device.begin_command_buffer(
                    command_pair.buffer,
                    &vk::CommandBufferBeginInfo::builder()
                                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                )?;

                record_function(command_pair.buffer);

                device.end_command_buffer(command_pair.buffer)?;
            }

            Ok::<_, anyhow::Error>(())
        }));

        drop(span);

        let node_id = self.nodes.add_node(CommandBufferId(buffer_id));

        for (parent_node_id, barrier_id) in parents {
            match barrier_id {
                Some(barrier_id) => {
                    self.nodes.update_edge(parent_node_id.0, barrier_id.0, ());
                    self.nodes.update_edge(barrier_id.0, node_id, ());
                },
                None => {
                    self.nodes.update_edge(parent_node_id.0, node_id, ());
                }
            }
        }

        Ok(NodeId(node_id))
    }

    pub fn register_global_barrier(&mut self, device: &ash::Device, barrier: vk_sync::GlobalBarrier) -> anyhow::Result<BarrierId> {
        let buffer_id = self.buffer_index.fetch_add(1, Ordering::Relaxed);

        let command_pair = self.buffers[buffer_id];

        unsafe {
            device.reset_command_pool(
                command_pair.pool,
                vk::CommandPoolResetFlags::empty(),
            )?;

            device.begin_command_buffer(
                command_pair.buffer,
                &vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
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

        let node_id = self.nodes.add_node(CommandBufferId(buffer_id));

        Ok(BarrierId(node_id))
    }

    pub fn get_buffers(&mut self) -> Vec<vk::CommandBuffer> {
        let span = tracy_client::span!("toposort");

        let sort = petgraph::algo::toposort(&self.nodes, None).unwrap();

        let buffers = sort.iter().map(|id| self.buffers[self.nodes[*id].0].buffer).collect();

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
