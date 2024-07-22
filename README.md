This is the repository for profiling the inference of Mamba1.<br>
There's a code generating tokens with beam search (size 2)<br>

prompt = "My cat wrote all this CUDA code for a new language model and"<br>

1. selective search with cuda kernel<br>
   run nsys profile cudas.py<br>
   selective search is implemented in the hugMamba.py file.<br>

2. selective search with pytorch<br>
   run nsys profile selective.py<br>
   selective search ref is implemented in the hugMamba.py file.<br>
   If you edit selective_scan_fn to selective_scan_ref in the cuda_kernels_forward function(in the MambaMixer class),<br>
   you can run selective search without using selective scan cuda kernel.<br>

3. recurrent update<br>
   run nsys profile script.py<br>
   recurrently updating is implemented in the recurrent_mamba.py file.<br>
   For now, this code has not syntex errors but makes wrong result.<br>

