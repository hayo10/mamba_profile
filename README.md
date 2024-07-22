This is the repository for profiling the inference of Mamba1.
There's a code generating tokens with beam search (size 2)

prompt = "My cat wrote all this CUDA code for a new language model and"

1. selective search with cuda kernel
   run nsys profile cudas.py
   selective search is implemented in the hugMamba.py file.

2. selective search with pytorch
   run nsys profile selective.py
   selective search ref is implemented in the hugMamba.py file.
   If you edit selective_scan_fn to selective_scan_ref in the cuda_kernels_forward function(in the MambaMixer class),
   you can run selective search without using selective scan cuda kernel.

3. recurrent update
   run nsys profile script.py
   recurrently updating is implemented in the recurrent_mamba.py file.
   For now, this code has not syntex errors but makes wrong result.

