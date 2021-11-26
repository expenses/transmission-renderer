use ash::vk;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Copy, Clone)]
struct CommandPoolBufferPair {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
}

pub struct CommandGraph {
    buffers: Vec<CommandPoolBufferPair>,
    buffer_index: AtomicUsize,
    nodes: petgraph::Graph<CommandBufferId, ()>,
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
        })
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.buffer_index.store(0, Ordering::Relaxed);
    }

    pub fn register_commands<'a, FN: FnOnce(vk::CommandBuffer) + Send + Sync + 'a>(&mut self, name: &str, parents: &[(NodeId, Option<BarrierId>)], device: &'a ash::Device, record_function: FN, scope: &mut bevy_tasks::Scope<'a, anyhow::Result<()>>) -> anyhow::Result<NodeId> {
        let _span = tracy_client::span!(&format!("Registering commands for {}", name));

        let buffer_id = self.buffer_index.fetch_add(1, Ordering::Relaxed);

        let command_pair = self.buffers[buffer_id];

        let span = tracy_client::span!("spawning");

        //let record_function = Box::new(record_function);

        scope.spawn(async move {
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
        });

        drop(span);

        let timing_test = tracy_client::span!("timing test");

        scope.spawn(async move {
            Ok::<_, anyhow::Error>(())
        });

        let node_id = self.nodes.add_node(CommandBufferId(buffer_id));

        let span = tracy_client::span!("Adding edges");

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

        drop(span);

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

    pub fn get_buffers(&self) -> Vec<vk::CommandBuffer> {
        let _span = tracy_client::span!("toposort");

        let sort = petgraph::algo::toposort(&self.nodes, None).unwrap();

        sort.iter().map(|id| self.buffers[self.nodes[*id].0].buffer).collect()
    }

    pub fn scoped<'a>(&'a mut self, scope: &'a mut bevy_tasks::Scope<'a, anyhow::Result<()>>) -> ScopedCommandGraph<'a, 'a> {
        ScopedCommandGraph {
            inner: self,
            scope
        }
    }
}

pub struct ScopedCommandGraph<'a, 'b> {
    pub inner: &'a mut CommandGraph,
    pub scope: &'b mut bevy_tasks::Scope<'a, anyhow::Result<()>>
}

impl<'a, 'b> ScopedCommandGraph<'a, 'b> {
    pub fn register_commands<FN: FnOnce(vk::CommandBuffer) + Send + Sync + 'a>(&mut self, name: &str, parents: &[(NodeId, Option<BarrierId>)], device: &'a ash::Device, record_function: FN) -> anyhow::Result<NodeId> {
        self.inner.register_commands(name, parents, device, record_function, self.scope)
    }

    pub fn register_global_barrier(&mut self, device: &ash::Device, barrier: vk_sync::GlobalBarrier) -> anyhow::Result<BarrierId> {
        self.inner.register_global_barrier(device, barrier)
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
