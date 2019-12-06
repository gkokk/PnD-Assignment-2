
#include <stdio.h>
#include <cblas.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <string.h>
#include <float.h> // used to initialize the global minimum to maximum double number
#include "knnring.h"



double* sum(double* points, int dim1, int dim2);

void swappointers(double** a, double** b);

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
	//printf("time spent on BLAS : %f \n", time_spent);
	for (i = 0; i < m * k; i++)
		knn.ndist[i] = sqrt(knn.ndist[i]);
	return knn;
}


knnresult distrAllkNN(double* Y, int n, int d, int k) // reversed symbols (Y == X) for clarity of code
{
	double start = MPI_Wtime();
	double avgtime, time, end;
	double globmax;
	double globmin;
	double minDist = DBL_MAX;
	double maxDist = -1;
	int numprocs, rank, block;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	block = n;

	double* disthelper = (double*)malloc(k * sizeof(double));
	int* indhelper = (int*)malloc(k * sizeof(int));
	knnresult tempknn;
	knnresult knnres = kNN(Y, Y, block, block, d, k); // knn init
	if (rank > 1)
	{
		for (int i = 0; i < block * k; i++)
		{
			knnres.nidx[i] += (rank - 1) * block;
		}
	}
	else if (rank == 0)
	{
		for (int i = 0; i < block * k; i++)
		{
			knnres.nidx[i] += (numprocs - 1) * block;
		}

	}

	//counters
	int z;
	int x;
	int y;

	double* X = malloc(block * d * sizeof(double));
	if (rank % 2 == 0)
	{

		double* temp = malloc(block * d * sizeof(double));

		for (int i = 1; i < numprocs; i++)
		{

			if (i > 1)
			{
				if (rank != 0)
					MPI_Recv(temp, block * d, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				else
					MPI_Recv(temp, block * d, MPI_DOUBLE, numprocs - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				if (rank != numprocs - 1)
					MPI_Send(X, block * d, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
				else
					MPI_Send(X, block * d, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
				swappointers(&X, &temp);

			}
			else
			{
				if (rank != 0)
					MPI_Recv(X, block * d, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				else
					MPI_Recv(X, block * d, MPI_DOUBLE, numprocs - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (rank != numprocs - 1)
					MPI_Send(Y, block * d, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
				else
					MPI_Send(Y, block * d, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

			}


			tempknn = kNN(X, Y, block, block, d, k);
			if (rank - i > 0)
				for (int j = 0; j < block; j++)
				{
					z = 0;
					x = 0;
					y = 0;


					while (z < k)
					{
						if (knnres.ndist[j + x * block] < tempknn.ndist[j + y * block])
						{
							disthelper[z] = knnres.ndist[j + x * block];
							indhelper[z] = knnres.nidx[j + x * block];
							x++;
						}
						else
						{

							disthelper[z] = tempknn.ndist[j + y * block];
							indhelper[z] = tempknn.nidx[j + y * block] + block * (rank - 1 - i);
							y++;
						}
						z++;

					}
					for (z = 0; z < k; z++)
					{
						knnres.ndist[j + block * z] = disthelper[z];
						knnres.nidx[j + block * z] = indhelper[z];
					}
				}
			else if (rank - i < 0)
				for (int j = 0; j < block; j++)
				{
					z = 0;
					x = 0;
					y = 0;


					while (z < k)
					{
						if (knnres.ndist[j + x * block] < tempknn.ndist[j + y * block])
						{
							disthelper[z] = knnres.ndist[j + x * block];
							indhelper[z] = knnres.nidx[j + x * block];
							x++;
						}
						else
						{
							disthelper[z] = tempknn.ndist[j + y * block];
							indhelper[z] = tempknn.nidx[j + y * block] + block * (numprocs + (rank - i) - 1);
							y++;
						}
						z++;

					}
					for (int z = 0; z < k; z++)
					{
						knnres.ndist[j + block * z] = disthelper[z];
						knnres.nidx[j + block * z] = indhelper[z];
					}
				}
			else
				for (int j = 0; j < block; j++)
				{
					z = 0;
					x = 0;
					y = 0;


					while (z < k)
					{
						if (knnres.ndist[j + x * block] < tempknn.ndist[j + y * block])
						{
							disthelper[z] = knnres.ndist[j + x * block];
							indhelper[z] = knnres.nidx[j + x * block];
							x++;
						}
						else
						{
							disthelper[z] = tempknn.ndist[j + y * block];
							indhelper[z] = tempknn.nidx[j + y * block] + block * (numprocs - 1);
							y++;
						}
						z++;

					}
					for (z = 0; z < k; z++)
					{
						knnres.ndist[j + block * z] = disthelper[z];
						knnres.nidx[j + block * z] = indhelper[z];
					}
				}


		}

		free(temp);
	}
	else
	{
		for (int i = 1; i < numprocs; i++)
		{

			if (i > 1)
			{
				if (rank != numprocs - 1)
					MPI_Send(X, block * d, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
				else
					MPI_Send(X, block * d, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
				if (rank != 0)
					MPI_Recv(X, block * d, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				else
					MPI_Recv(X, block * d, MPI_DOUBLE, numprocs - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			else
			{
				if (rank != numprocs - 1)
					MPI_Send(Y, block * d, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
				else
					MPI_Send(Y, block * d, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
				if (rank != 0)
					MPI_Recv(X, block * d, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				else
					MPI_Recv(X, block * d, MPI_DOUBLE, numprocs - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			tempknn = kNN(X, Y, block, block, d, k);
			if (rank - i > 0)
				for (int j = 0; j < block; j++)
				{
					z = 0;
					x = 0;
					y = 0;


					while (z < k)
					{
						if (knnres.ndist[j + x * block] < tempknn.ndist[j + y * block])
						{
							disthelper[z] = knnres.ndist[j + x * block];
							indhelper[z] = knnres.nidx[j + x * block];
							x++;
						}
						else
						{

							disthelper[z] = tempknn.ndist[j + y * block];
							indhelper[z] = tempknn.nidx[j + y * block] + block * (rank - 1 - i);
							y++;
						}
						z++;

					}
					for (z = 0; z < k; z++)
					{
						knnres.ndist[j + block * z] = disthelper[z];
						knnres.nidx[j + block * z] = indhelper[z];
					}
				}
			else if (rank - i < 0)
				for (int j = 0; j < block; j++)
				{
					z = 0;
					x = 0;
					y = 0;


					while (z < k)
					{
						if (knnres.ndist[j + x * block] < tempknn.ndist[j + y * block])
						{
							disthelper[z] = knnres.ndist[j + x * block];
							indhelper[z] = knnres.nidx[j + x * block];
							x++;
						}
						else
						{
							disthelper[z] = tempknn.ndist[j + y * block];
							indhelper[z] = tempknn.nidx[j + y * block] + block * (numprocs + (rank - i) - 1);
							y++;
						}
						z++;

					}
					for (int z = 0; z < k; z++)
					{
						knnres.ndist[j + block * z] = disthelper[z];
						knnres.nidx[j + block * z] = indhelper[z];
					}
				}
			else
				for (int j = 0; j < block; j++)
				{
					z = 0;
					x = 0;
					y = 0;


					while (z < k)
					{
						if (knnres.ndist[j + x * block] < tempknn.ndist[j + y * block])
						{
							disthelper[z] = knnres.ndist[j + x * block];
							indhelper[z] = knnres.nidx[j + x * block];
							x++;
						}
						else
						{
							disthelper[z] = tempknn.ndist[j + y * block];
							indhelper[z] = tempknn.nidx[j + y * block] + block * (numprocs - 1);
							y++;
						}
						z++;

					}
					for (z = 0; z < k; z++)
					{
						knnres.ndist[j + block * z] = disthelper[z];
						knnres.nidx[j + block * z] = indhelper[z];
					}
				}

		}

	}
	for (int i = 0; i < n * k; i++)
	{
		if (knnres.ndist[i] > maxDist)
			maxDist = knnres.ndist[i];
		if (knnres.ndist[i] < minDist && knnres.ndist[i] >1e-6)
			minDist = knnres.ndist[i];
	}
	MPI_Reduce(&maxDist, &globmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&minDist, &globmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	end = MPI_Wtime();
	time = end - start;
	MPI_Reduce(&time, &avgtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("average time spent is %.3f secs in sychronous \n", avgtime / numprocs);
		printf("max is %f  and min is %f\n", globmax, globmin);
	}
	free(disthelper);
	free(indhelper);
	free(tempknn.ndist);
	free(tempknn.nidx);
	return knnres;
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

void swappointers(double** a, double** b)
{

	double* temp = *a;
	*a = *b;
	*b = temp;

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
