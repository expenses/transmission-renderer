use std::sync::Arc;
use std::sync::mpsc::{Sender, Receiver};

type TaskInner<'a> = Box<dyn Fn() -> anyhow::Result<()> + Send + 'a>;

struct Task<'a> {
    task: TaskInner<'a>,
    sender: Sender<anyhow::Result<()>>,
}

pub struct Handle {
    receiver: Receiver<anyhow::Result<()>>
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
            _threads: (0 .. num_cpus::get()).map(|_| {
                let receiver = receiver.clone();

                std::thread::spawn(move || {
                    loop {
                        let Task { task, sender } = receiver.lock().recv().unwrap();
                        let _ = sender.send(task());
                    }
                })
            }).collect(),
            sender,
        }
    }

    pub fn spawn<'a, FN: Fn() -> anyhow::Result<()> + Send + 'a>(&self, task: FN) -> Handle {
        let (sender, receiver) = std::sync::mpsc::channel();

        let task: TaskInner<'a> = Box::new(task);

        let _ = self.sender.send(Task {
            task: unsafe {
                std::mem::transmute::<TaskInner<'a>, TaskInner<'static>>(task)
            },
            sender
        });

        Handle {
            receiver
        }
    }
}
