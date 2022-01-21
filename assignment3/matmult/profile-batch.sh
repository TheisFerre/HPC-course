#BSUB -J proftest 
#BSUB -q hpcintrogpu 
#BSUB -n 4 
#BSUB -R "span[hosts=1]"  
#BSUB -gpu "num=1:mode=exclusive_process"  
#BSUB -W 10 
#BSUB -R "rusage[mem=2048]"  
 
module load cuda/11.5.1
GPU_VERSION=gpulib
export TMPDIR=$__LSF_JOB_TMPDIR__
SIZE=2048
 
export MFLOPS_MAX_IT=1  
 
nv-nsight-cu-cli -o profile_$GPU_VERSION \ 
    --section MemoryWorkloadAnalysis \ 
    --section MemoryWorkloadAnalysis_Chart \ 
    --section ComputeWorkloadAnalysis \ 
./matmult_f.nvcc $GPU_VERSION $SIZE $SIZE $SIZE