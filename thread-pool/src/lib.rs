use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

type TaskInner<'a> = Box<dyn FnOnce() -> anyhow::Result<()> + Send + 'a>;

struct Task<'a> {
    task: TaskInner<'a>,
    sender: Sender<anyhow::Result<()>>,
}

pub struct Handle {
    receiver: Receiver<anyhow::Result<()>>,
}

impl Handle {
    pub fn block_on_result(&self) -> anyhow::Result<()> {
        let result = self.receiver.recv()?;
        result
    }
}

pub struct ThreadPool {
    _threads: Vec<std::thread::JoinHandle<()>>,
    sender: Sender<Task<'static>>,
}

impl ThreadPool {
    pub fn new() -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();

        let receiver = Arc::new(parking_lot::Mutex::new(receiver));

        Self {
            _threads: (0..num_cpus::get())
                .map(|_| {
                    let receiver = receiver.clone();

                    std::thread::spawn(move || loop {
                        let Task { task, sender } = receiver.lock().recv().unwrap();
                        let _ = sender.send(task());
                    })
                })
                .collect(),
            sender,
        }
    }

    /// This function is marked unsafe as it doesn't require that tasks have a static lifetime.
    ///
    /// Safe usage requires that you wait on the task to complete before leaving the scope.
    pub unsafe fn spawn<FN: FnOnce() -> anyhow::Result<()> + Send>(&self, task: FN) -> Handle {
        let (sender, receiver) = std::sync::mpsc::channel();

        let task: TaskInner<'_> = Box::new(task);

        let _ = self.sender.send(Task {
            task: std::mem::transmute::<TaskInner<'_>, TaskInner<'static>>(task),
            sender,
        });

        Handle { receiver }
    }
}
