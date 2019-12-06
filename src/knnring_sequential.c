
#include <stdio.h>
#include <cblas.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "knnring.h"


double* sum(double* points, int dim1, int dim2);

void quicksort(double* X, int* ind, int left, int right, int step);

int qsortpartition(double* X, int* ind, int left, int right, int step);

void swap(double* X, int* ind, int a, int b);

double QS(double* X, int* ind, int left, int right, int k, int step);

int partition(double* X, int* ind, int left, int right, int pivotIndex, int step);



knnresult kNN(double* X, double* Y, int n, int m, int d, int k)
{

	knnresult knn;
	knn.k = k;
	knn.m = m;
	knn.ndist = (double*)malloc(n * m * sizeof(double));
	knn.nidx = (int*)malloc(n * m * sizeof(double));
	int j;
	int i;
	double* sumY = sum(Y, m, d);
	double* sumX = sum(X, n, d);
	clock_t begin = clock();

	int counter = 0;
	double alpha = 1.0;
	double beta = 0.0;



	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
		m, n, d, alpha, Y, m, X, n, beta, knn.ndist, m);
	clock_t end = clock();

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			knn.ndist[j * m + i] = sumY[i] - 2 * knn.ndist[j * m + i] + sumX[j];//the distance matrix is n-by-m,so next step, we wanna keep up until the k row

			knn.nidx[j * m + i] = j;
			if (knn.ndist[j * m + i] < 1e-6)
				counter++;
		}

	}
	if (counter > 1)
		for (i = 0; i < n; i++)
		{
			knn.ndist[i + i * n] = 0.0;
		}


	for (i = 0; i < m; i++)
	{
		QS(knn.ndist, knn.nidx, i, (n - 1) * m + i, (k - 1) * m + i, m); // we modify quickselect to run on rows instead of columns
	}

	knn.ndist = (double*)realloc(knn.ndist, m * k * sizeof(double)); // cutting the extra rows
	knn.nidx = (int*)realloc(knn.nidx, m * k * sizeof(int));

	for (i = 0; i < m; i++)
	{
		quicksort(knn.ndist, knn.nidx, i, (k - 1) * m + i, m);
	}

	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("time spent on BLAS : %f \n", time_spent);
	for (i = 0; i < m * k; i++)
		knn.ndist[i] = sqrt(knn.ndist[i]);
	return knn;
}


void swap(double* X, int* ind, int a, int b)
{
	double temp = X[a];
	int temp2 = ind[a];
	X[a] = X[b];
	X[b] = temp;
	ind[a] = ind[b];
	ind[b] = temp2;

}

double QS(double* X, int* ind, int left, int right, int k, int step)
{
	if (left == right) {
		return X[left];
	}
	int pivotIndex = right;
	pivotIndex = partition(X, ind, left, right, pivotIndex, step);
	if (k == pivotIndex) {
		return X[k];
	}
	else if (k < pivotIndex) {
		return QS(X, ind, left, pivotIndex - step, k, step);
	}
	else {
		return QS(X, ind, pivotIndex + step, right, k, step);
	}

}

int partition(double* X, int* ind, int left, int right, int pivotIndex, int step)
{

	double pivotValue = X[pivotIndex];
	swap(X, ind, pivotIndex, right);
	int storeIndex = left;
	for (int i = left; i < right; i += step) {
		if (X[i] < pivotValue) {
			swap(X, ind, storeIndex, i);
			storeIndex += step;
		}
	}
	swap(X, ind, right, storeIndex);
	return storeIndex;
}

int qsortpartition(double* X, int* ind, int left, int right, int step)
{
	double pvalue = X[right];
	int i = left;
	for (int j = left; j < right; j += step)
	{
		if (X[j] < pvalue)
		{
			swap(X, ind, i, j);
			i += step;
		}
	}
	swap(X, ind, i, right);
	return i;
}

void quicksort(double* X, int* ind, int left, int right, int step)
{
	if (left < right)
	{
		int p = qsortpartition(X, ind, left, right, step);
		quicksort(X, ind, left, p - step, step);
		quicksort(X, ind, p + step, right, step);
	}
}

double* sum(double* points, int dim1, int dim2)
{
	double* sums = (double*)malloc(dim1 * sizeof(double));
	if (sums == NULL) return NULL;
	double sum;
	for (int i = 0; i < dim1; i++)
	{
		sum = 0;

		for (int j = 0; j < dim2; j++)
		{
			sum += points[dim1 * j + i] * points[dim1 * j + i];
		}
		sums[i] = sum;
	}
	return sums;
}


