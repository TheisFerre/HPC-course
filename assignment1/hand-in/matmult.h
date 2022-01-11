void matmult_nat(int M, int N, int K, double **A, double **B, double **C);

void matmult_lib(int M, int N, int K, double **A, double **B, double **C);

void matmult_mkn(int M, int N, int K, double **A, double **B, double **C);

void matmult_mnk(int M, int N, int K, double **A, double **B, double **C);

void matmult_kmn(int M, int N, int K, double **A, double **B, double **C);

void matmult_knm(int M, int N, int K, double **A, double **B, double **C);

void matmult_nkm(int M, int N, int K, double **A, double **B, double **C);

void matmult_nmk(int M, int N, int K, double **A, double **B, double **C);

void matmult_blk(int M, int N, int K, double **A, double **B, double **C, int bs);
