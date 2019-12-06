#ifndef KNNRING_H
#define KNNRING_H



typedef struct knnresult {
	int* nidx; //!< Indices (0-based) of nearest neighbors [m-by-k]
	double* ndist; //!< Distance of nearest neighbors [m-by-k]
	int m; //!< Number of query points [scalar]
	int k; //!< Number of nearest neighbors [scalar]
} knnresult;

knnresult kNN(double* X, double* Y, int n, int m, int d, int k);

knnresult distrAllkNN(double* Y, int n, int d, int k); // reversed symbols (Y == X) for clarity of code

#endif
