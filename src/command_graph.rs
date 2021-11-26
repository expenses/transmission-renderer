use ash::vk;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;
use std::collections::{HashSet, VecDeque};

pub struct CommandGraph {
    buffers: Vec<vk::CommandBuffer>,
    buffer_index: AtomicUsize,
    device: ash::Device,
    nodes: Mutex<petgraph::Graph<CommandBufferId, ()>>,
    root: AtomicUsize,
}

impl CommandGraph {
    pub fn new(device: &ash::Device, pool: vk::CommandPool, num_command_buffers: u32) -> anyhow::Result<Self> {
        Ok(Self {
            buffers: unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(num_command_buffers),
                )
            }?,
            device: device.clone(),
            buffer_index: Default::default(),
            nodes: Default::default(),
            root: Default::default()
        })
    }

    pub fn reset(&self) {
        self.nodes.lock().clear();
        self.buffer_index.store(0, Ordering::Relaxed);
    }

    pub fn register_commands<FN: Fn(vk::CommandBuffer)>(&self, parents: Vec<(NodeId, Option<BarrierId>)>, record_function: FN) -> anyhow::Result<NodeId> {
        let _span = tracy_client::span!("Registering commands");

        let buffer_id = self.buffer_index.fetch_add(1, Ordering::Relaxed);

        let buffer = self.buffers[buffer_id];

        unsafe {
            self.device.begin_command_buffer(
                buffer,
                &vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            )?;

            record_function(buffer);

            self.device.end_command_buffer(buffer)?;
        }

        let command_buffer_id = CommandBufferId(buffer_id);

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

        let buffer = self.buffers[buffer_id];

        unsafe {
            self.device.begin_command_buffer(
                buffer,
                &vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            )?;

            vk_sync::cmd::pipeline_barrier(
                &self.device,
                buffer,
                Some(barrier),
                &[],
                &[],
            );

            self.device.end_command_buffer(buffer)?;
        }

        let node_id = self.nodes.lock().add_node(CommandBufferId(buffer_id));

        Ok(BarrierId(node_id))
    }

    pub fn get_buffers(&self) -> Vec<vk::CommandBuffer> {
        let _span = tracy_client::span!("toposort");

        let nodes = self.nodes.lock();

        let sort = petgraph::algo::toposort(&*nodes, None).unwrap();

        sort.iter().map(|id| self.buffers[nodes[*id].0]).collect()
    }
}

struct CommandBufferId(usize);
#[derive(Clone, Copy)]
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
