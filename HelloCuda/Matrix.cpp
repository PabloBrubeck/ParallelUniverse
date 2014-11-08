#include<stdio.h>
#include<math.h>

template <size_t n, size_t m>
void printMatrix(double(&M)[n][m]){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < m; j++){
			printf("%f\t", M[i][j]);
		}
		printf("\n");
	}
}

template <size_t n, size_t m>
void GaussJordan(double(&M)[n][m]){
	int p;
	double t;
	for (int i = 0; i<n; i++){
		t = M[i][i];
		for (int j = i; j<m; j++){
			M[i][j] /= t;
		}
		for (int k = 1; k<n; k++){
			p = (k + i) % n;
			t = M[p][i];
			for (int h = 0; h<m; h++){
				M[p][h] -= t*M[i][h];
			}
		}
	}
}

template <size_t n, size_t m>
void matAdd(double(&A)[n][m], double(&B)[n][m], double(&C)[n][m]){
	for (int i = 0; i<n; i++){
		for (int j = 0; j<r; j++){
			C[i][j] = A[i][j] + B[i][j];
		}
	}
}

template <size_t n, size_t m, size_t r>
void matMult(double(&A)[n][m], double(&B)[m][r], double(&C)[n][r]){
	for (int i = 0; i<n; i++){
		for (int j = 0; j<r; j++){
			C[i][j] = 0;
			for (int k = 0; k<m; k++){
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void eulerRotation(double alpha, double beta, double gamma, double(&R)[4][4]){
	double c = cos(-alpha);
	double s = sin(-alpha);
	double Rx[4][4] = { { 1, 0, 0, 0 }, { 0, c, -s, 0 }, { 0, s, c, 0 }, { 0, 0, 0, 1 } };
	c = cos(-beta);
	s = sin(-beta);
	double Ry[4][4] = { { c, 0, s, 0 }, { 0, 1, 0, 0 }, { -s, 0, c, 0 }, { 0, 0, 0, 1 } };
	c = cos(-gamma);
	s = sin(-gamma);
	double Rz[4][4] = { { c, -s, 0, 0 }, { s, c, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
	double temp[4][4];
	matMult(Rx, Ry, temp);
	matMult(temp, Rz, R);
}

template <size_t n>
void toColumn(double arr[n], double(&M)[n][1]){
	for (int i = 0; i < n; i++){
		M[i][0] = arr[i];
	}
}

int notmain(){
	
	double A[4][1] = { {1}, {-2}, {-3}, {1} };
	printMatrix(A);
	printf("\n");

	double R[4][4];
	eulerRotation(1.5, 3.2, 2.7, R);
	printMatrix(R);
	printf("\n");

	double B[4][1];
	matMult(R, A, B);
	printMatrix(B);

	return 0;
}