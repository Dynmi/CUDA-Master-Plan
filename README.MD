# <center>Cuda Master Plan</center>

This is an unofficial **Tutorial** for CUDA programming.

</br>

## What is cuda and why you need it?

CUDA stands for "Compute Unified Device Architecture". 
CUDA, served as a C/C++ toolkit, is a parallel computing platform and application programming interface (API) model created by NVIDIA, which **only support PC with NVIDIA GPU**. Now, CUDA is popular all over the world, especially in the field of Computer Vision and Machine Learning. 
If you are a practitioner in CV or ML, CUDA ought to be a required skill for you. 

Sometimes there will be tasks that contains lots of simple computation operation. 
As our CPU only support low level paralleling, maybe 12 threads or 32 threads, it seems disgusting time-consuming when it comes to such tasks. 
Well, CUDA could make such tasks time-friendly to us, by carrying out thousands of threads-paralleling. So, the huge repetitive computation task was transported to and finished on your GPU. 
By doing so, it usually **boost the speed** by 8x or 32x. 

</br>

## How to master cuda programming in one week?

I think the SOTA approach can be divided into three levels:
- Level 1: You get to know what is cuda, and be able to use cuda in a **NAIVE** way to accelerate your computation.
- Level 2: You are **PROFICIENT** to use all kinds of techniques to optimize the naive cuda programming, such as shared memories, ILP.
- Level 3: You can **FLUENTLY** use cuda in your project to do whatever you want without restraint. For example, use cuda in a nerual network project.

Standing on the first level just needs several hours, and it's a piece of cake. 
Then it will cost you about three days to join the community of those belonging to level 2. 
There may be some of hardness to you, but don't be flinch. 
After that, you should try to stand on level 3, which needs you to exercise your CUDA skill in practice.  

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

...

## Grid,  Block,  Thread,  Wrap

...

## Global Memory,  Shared Memory,  Register,  L1 cache,  L2 cache

...


## Metrics and Evaluation Tools

...


## Contributing

- [Dynmi](https://github.com/Dynmi)
