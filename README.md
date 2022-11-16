# COMP-5704-Project



## Part 1: Basic Parallelization using two processors
In this we split up the Gx Gy kernel applications to the image assinging the function to a processor. This means we have two processors carrying out work in tandem, after which the serial bottleneck of combining them takes over. This will prevent us from achieving a perfect x2 speedup. 
```
//Serial filtering finished in 13.6863068 sec
//Parallel filtering finished in 6.588868300000001 sec
```

Looking at the formulation, we find that we don't obtain the optimal speedup, but that theoretical speedup is impossible anyway because of the need to combine the two filters. However, it is pretty close.
6.588868300000001*2 = 13.1777366
