import faiss
import numpy as np
import h5py




# Create a GPU index
gpu_res = faiss.StandardGpuResources()
gpu_index = faiss.index_factory(768, "IVF1024,Flat")
gpu_index_ivf = faiss.extract_index_ivf(gpu_index)
gpu_index_ivf.quantizer_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, gpu_index_ivf.quantizer)

# Train the index
faiss.index_gpu_to_cpu(gpu_index).train(embeddings)

# Add the embeddings to the index
faiss.index_cpu_to_gpu(gpu_res, 0, faiss.index_gpu_to_cpu(gpu_index)).add(embeddings)
