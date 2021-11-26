use ash::vk;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;
use std::sync::Arc;
use crate::profiling::ProfilingContext;

#[derive(Copy, Clone)]
struct CommandPoolBufferPair {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
}

pub struct RecordContext<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub profiling_ctx: &'a ProfilingContext,
}

pub struct CommandGraph {
    buffers: Vec<CommandPoolBufferPair>,
    buffer_index: AtomicUsize,
    device: ash::Device,
    nodes: Arc<Mutex<petgraph::Graph<CommandBufferId, ()>>>,
    root: AtomicUsize,
    handles: Mutex<Vec<async_std::task::JoinHandle<anyhow::Result<()>>>>,
    profiling_context: Arc<ProfilingContext>,
}

impl CommandGraph {
    pub fn new(device: &ash::Device, num_command_buffers: u32, graphics_queue_family: u32, profiling_context: Arc<ProfilingContext>) -> anyhow::Result<Self> {
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
            device: device.clone(),
            buffer_index: Default::default(),
            nodes: Default::default(),
            root: Default::default(),
            handles: Default::default(),
            profiling_context
        })
    }

    pub fn reset(&self) {
        self.nodes.lock().clear();
        self.buffer_index.store(0, Ordering::Relaxed);
    }

    pub fn register_commands<FN: FnOnce(&RecordContext) + Send + Sync + 'static>(&self, parents: &[(NodeId, Option<BarrierId>)], record_function: FN) -> anyhow::Result<NodeId> {
        let _span = tracy_client::span!("Registering commands");

        let buffer_id = self.buffer_index.fetch_add(1, Ordering::Relaxed);

        let command_pair = self.buffers[buffer_id];

        let device = self.device.clone();
        let profiling_ctx = self.profiling_context.clone();

        let handle = async_std::task::spawn(async move {
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

                let record_context = RecordContext {
                    device: &device,
                    command_buffer: command_pair.buffer,
                    profiling_ctx: &profiling_ctx,
                };

                record_function(&record_context);

                device.end_command_buffer(command_pair.buffer)?;
            }

            Ok::<_, anyhow::Error>(())
        });

        self.handles.lock().push(handle);

        let mut nodes = self.nodes.lock();

        let node_id = nodes.add_node(CommandBufferId(buffer_id));

        if parents.is_empty() {
            self.root.store(node_id.index(), Ordering::Relaxed);
        }

        for (parent_node_id, barrier_id) in parents {
            match barrier_id {
                Some(barrier_id) => {
                    nodes.update_edge(parent_node_id.0, barrier_id.0, ());
                    nodes.update_edge(barrier_id.0, node_id, ());
                },
                None => {
                    nodes.update_edge(parent_node_id.0, node_id, ());
                }
            }
        }

        Ok(NodeId(node_id))
    }

    pub fn register_global_barrier(&self, barrier: vk_sync::GlobalBarrier) -> anyhow::Result<BarrierId> {
        let buffer_id = self.buffer_index.fetch_add(1, Ordering::Relaxed);

        let command_pair = self.buffers[buffer_id];

        unsafe {
            self.device.reset_command_pool(
                command_pair.pool,
                vk::CommandPoolResetFlags::empty(),
            )?;

            self.device.begin_command_buffer(
                command_pair.buffer,
                &vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            )?;

            vk_sync::cmd::pipeline_barrier(
                &self.device,
                command_pair.buffer,
                Some(barrier),
                &[],
                &[],
            );

            self.device.end_command_buffer(command_pair.buffer)?;
        }

        let node_id = self.nodes.lock().add_node(CommandBufferId(buffer_id));

        Ok(BarrierId(node_id))
    }

    pub fn get_buffers(&self) -> anyhow::Result<Vec<vk::CommandBuffer>> {
        let _span = tracy_client::span!("get buffers");

        async_std::task::block_on(async move {
            let nodes = self.nodes.clone();

            let sort_handle = async_std::task::spawn(async move {
                let _span = tracy_client::span!("toposort");

                let nodes = nodes.lock();

                petgraph::algo::toposort(&*nodes, None)
            });

            futures::future::try_join_all(
                self.handles.lock().drain(..)
            ).await?;

            let sort = sort_handle.await.unwrap();

            let nodes = self.nodes.lock();

            let buffers = sort.iter().map(|id| self.buffers[nodes[*id].0].buffer).collect();

            Ok(buffers)
        })
    }
}

struct CommandBufferId(usize);
#[derive(Clone, Copy, Debug)]
pub struct NodeId(petgraph::graph::NodeIndex);
#[derive(Clone, Copy)]
pub struct BarrierId(petgraph::graph::NodeIndex);

#[test]
fn hmmm() {
    let command_graph = CommandGraph {
        buffers: (0 .. 10).map(|_| DummyCommandBuffer(Default::default())).collect(),
        buffer_index: Default::default(),
        nodes: Default::default(),
        root: Default::default()
    };

    let barrier = command_graph.register_global_barrier(vk_sync::GlobalBarrier::default()).unwrap();

    let a = command_graph.register_commands(vec![], |buffer| {
        buffer.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }).unwrap();

    let b = command_graph.register_commands(vec![(a, None)], |buffer| {
        buffer.0.fetch_add(2, Ordering::Relaxed);
        Ok(())
    }).unwrap();

    let c = command_graph.register_commands(vec![(a, Some(barrier))], |buffer| {
        buffer.0.fetch_add(3, Ordering::Relaxed);
        Ok(())
    }).unwrap();

    let d = command_graph.register_commands(vec![(b, None), (c, None)], |buffer| {
        buffer.0.fetch_add(99, Ordering::Relaxed);
        Ok(())
    }).unwrap();

    panic!("{:?}", command_graph.get_buffers());

}
