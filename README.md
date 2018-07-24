# GDataStructures

Data structures made for CUDA

## Purpose

This library was designed to propose some data structures for the GPU.

It provides those following dictionaries:
 - A non-concurrent B+-tree
 - A "GPU LSM: A Dynamic Dictionary Data Structure for the GPU" [LSM](https://escholarship.org/uc/item/65t741zg)
 - A (fixed depth) Hash Array Mapped Trie
 - A X-fast trie

## Instructions to compile

It relies on the [GSTL](https://github.com/Gawaboumga/GSTL).

 - mkdir build
 - cd build
 - cmake .. -DCMAKE_GENERATOR_PLATFORM=x64

## Thanks

A special thanks to the [Catch](https://github.com/catchorg/Catch2), [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers) and [CUB](https://github.com/NVlabs/cub) libraries.