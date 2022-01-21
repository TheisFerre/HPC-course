#BSUB -J proftest 
#BSUB -q hpcintrogpu 
#BSUB -n 4 
#BSUB -R "span[hosts=1]"  
#BSUB -gpu "num=1:mode=exclusive_process"  
#BSUB -W 10 
#BSUB -R "rusage[mem=2048]"  
 
module load cuda/11.5.1

export TMPDIR=$__LSF_JOB_TMPDIR__
export MFLOPS_MAX_IT=1

SIZE=2048
GPU_VERSION=gpulib
 
nv-nsight-cu-cli -f -o profile_$GPU_VERSION --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section ComputeWorkloadAnalysis --section SpeedOfLight ./matmult_f.nvcc $GPU_VERSION $SIZE $SIZE $SIZE