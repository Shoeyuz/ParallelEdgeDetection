# Parallel Edge Detection


## Part 1: MISD Parallelization using two processors
In this we split up the Gx Gy kernel applications to the image assinging the function to a processor. This means we have two processors carrying out work in tandem, after which the serial bottleneck of combining them takes over. This will prevent us from achieving a perfect x2 speedup. 
```
//Serial filtering finished in 13.6863068 sec
//Parallel filtering finished in 6.588868300000001 sec
```

Looking at the formulation, we find that we don't obtain the optimal speedup, but that theoretical speedup is impossible anyway because of the need to combine the two filters. However, it is pretty close.
6.588868300000001*2 = 13.1777366



## Part 2: Parallelizing using GPU speedup with CUDA JIT
Using the compiler infrastructure we can take advantage of CUDA by applying process allocation onto the GPU through the compiler. This grants us quite the speedup with our code managing to complete in a mere 2 seconds.

What's nice about this approach compared to previous is that we aren't splitting it and combining the two filters later. Thus, we in a way are able to obtain a faster architecture here. Even if we assign half our cores to one, and half our cores to the other filter - Gx, Gy - we still run into the problem of serialzed portion having to combine. By doing them both in one function, we are able to allow GPU cores to combine them in the function itself and prevent the serial portion of the program to be big!


```
//Serial filtering finished in 13.6863068 sec
//Parallel filtering finished in 2.1004436 sec
```

## Part 3: Most effective parallelization
We don't use CUDA in this case, but what we do is we split up the image into a bunch of sections and then we pass those sections to some cores - these processors then run in parallel and carry out their work on these chunks themselves and at the end we serially combine the multiple chunks back into the image together. You can actually see in the image how the image is split up if you run the file: parallelBreakdown.py

```
//Serial filtering finished in 13.6863068 sec
//Parallel filtering finished in 1.5849926 sec
```
