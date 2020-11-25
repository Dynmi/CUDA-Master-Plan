# <center>Cuda Master Plan</center>

This is an unofficial **Tutorial** for CUDA programming.

</br>

## What is cuda and why you need it?

CUDA stands for "Compute Unified Device Architecture". 
CUDA, served as a C/C++ toolkit, is a parallel computing platform and application programming interface (API) model created by NVIDIA, which **only support PC with NVIDIA GPU**. Now, CUDA is popular all over the world, especially in the field of Computer Vision and Machine Learning. 
If you are a practitioner in CV or ML, CUDA ought to be a required skill for you. 

Sometimes there will be tasks that contains lots of simple computation operation. 
As our CPU only support low stage paralleling, maybe 12 threads or 32 threads, it seems disgusting time-consuming when it comes to such tasks. 
Well, CUDA could make such tasks time-friendly to us, by carrying out thousands of threads-paralleling. So, the huge repetitive computation task was transported to and finished on your GPU. 
By doing so, it usually **boost the speed** by 8x or 32x. 

</br>

## How to master cuda programming in one week?

I think the SOTA approach can be divided into three stages:
- stage 1: You get to know what is cuda, and be able to use cuda in a **NAIVE** way to accelerate your computation.
- stage 2: You are **PROFICIENT** to use all kinds of techniques to optimize the naive cuda programming, such as shared memories, ILP.
- stage 3: You can **FLUENTLY** use cuda in your project to do whatever you want without restraint. For example, use cuda in a nerual network project.

Standing on the first stage needs several hours, which is just a piece of cake. 
Then it will cost you about three days to join the community of those belonging to stage 2. 
There may be some of hardness to you, but don't be flinch. 
After that, you should try to stand on stage 3, which needs you to exercise your CUDA skill in practice.  

</br>

## List of examples

```
.
├── LEVEL_1
│   ├── vector add 
│   └── matrix multiply
│   └── matrix transpose
...
├── LEVEL_2
│   ├── ...
...
├── LEVEL_3
│   ├── ...
...
└───
```

</br>

## Streaming Multiprocessor(SM),  Streaming Process(SP)

SM is consist of 
CUDA cores(SP), 
Shared Memory/L1Cache, 
Register File,
Load/Store Units,
Special Function Units,
Warp Scheduler.

SM是GPU的基本控制指令执行单元，它拥有独立的指令调度电路。一个SM下所有的SP共享同一组控制指令。因此每个独立的计算任务至少要用一个SM执行，如果计算任务的规模无法让SM及其下所有的SP“吃饱”，就会浪费该SM下部分SP的算力。此外，每个SM还拥有一套独立的共享内存。

SP是GPU的基本算术指令执行单元，它没有指令调度电路，但拥有独立的算术电路，包括1个ALU（Arithmetic logic unit）和1个FPU（Float Point Unit）。每个SP负责处理固定数量的线程，注意这里的“线程”与CPU上的线程不同，它们共享同一组算术指令，处理不同的数据，这种并行方式又叫作SIMD（Single Instruction Multiple Data）。SP内部没有任何除寄存器和缓冲队列外的独立存储系统。

</br>

## Grid,  Block,  Thread,  Wrap

...

## Global Memory,  Shared Memory,  Register,  L1 cache,  L2 cache

...


## Metrics and Evaluation Tools

...


## Contributing

- [Dynmi](https://github.com/Dynmi)
