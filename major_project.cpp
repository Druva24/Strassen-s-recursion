#include <omp.h>
#include <stdio.h>   
#include <stdlib.h>  
#include <time.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <ctime>
#include <time.h>
#include <vector>
#include <limits.h>


using namespace std;


int** initialize_mat(int r, int c)
{
    
    int** mat1 = new int*[r];
    
    int* mat2 = new int[r * c];

    
    for (int counter1 = 0; counter1 < r; counter1++)
    {
        mat1[counter1] = mat2 + (c * counter1);
    }

    return mat1;
}

void update_indices(int** mat1, int m, int** mat2, int ind_1, int ind_2, bool update = true)
{
    for (int counter1 = 0; counter1 < m; counter1++)
    {   
        if(update) {
            mat1[counter1] = mat2[ind_1 + counter1] + ind_2;
        }
        
    }
}

void clear_memory(int** mat)
{
    delete[] (*mat);
    delete[] mat;
}




void traditional_mat_mul(int size, int** mat1, int** mat2, int** res_mat, bool default_cal = true)
{    
	for (int counter1 = 0; counter1 < size; counter1++)   
    {
        for (int counter2 = 0; counter2 < size; counter2++)    
        {          
            int out = 0;

            for (int k = 0; k < size; k++)
            {
                out += mat1[counter1][k] * mat2[k][counter2];
            }

            if(default_cal) {
                res_mat[counter1][counter2] = out;
            }
            
        }   
    }   
}  





void add_matrices(int** res_mat, int size1, int size2, int** mat1, int** mat2)
{
    for (int counter1 = 0; counter1 < size1; counter1++)
    {	
        for (int counter2 = 0; counter2 < size2; counter2++)
        {
            res_mat[counter1][counter2] = mat1[counter1][counter2] + mat2[counter1][counter2];
        }
    }
}



void diff_matrices(int** res_mat, int r, int columns, int** mat1, int** mat2)
{
    for (int counter1 = 0; counter1 < r; counter1++) {
        for (int counter2 = 0; counter2 < columns; counter2++) {
            *(*(res_mat + counter1) + counter2) = *(*(mat1 + counter1) + counter2) - *(*(mat2 + counter1) + counter2);
        }
    }
}



void strassen_mat_mul_algorithm (int matrix_size, int **first_matrix, int **second_matrix, int **res_mat, int &limit, bool is_parallel = true)
{
	if (matrix_size<=limit){
		for (int counter1 = 0; counter1 < matrix_size; counter1++)   
        {
            for (int counter2 = 0; counter2 < matrix_size; counter2++)    
            {          
                int sum = 0;
                for (int k = 0; k < matrix_size; k++)
                {
                    sum += first_matrix[counter1][k] * second_matrix[k][counter2];
                }

                if(is_parallel) {
                    res_mat[counter1][counter2] = sum;
                }
                
            }   
        }   
		
	}
    
	else {
		int n = matrix_size/2;

		int **mat1 = initialize_mat(n, n);
        int **mat2 = initialize_mat(n, n);
        int **mat3 = initialize_mat(n, n);
        int **mat4 = initialize_mat(n, n);
        int **mat5 = initialize_mat(n, n);
        int **mat6 = initialize_mat(n, n);
		int **mat7 = initialize_mat(n, n);

        
        int **mat_a1 = initialize_mat(n, n); 
        int **mat_a2 = initialize_mat(n, n); 
        int **mat_a5 = initialize_mat(n, n);
        int **mat_a6 = initialize_mat(n, n);
        int **mat_a7 = initialize_mat(n, n);

        
		int **mat_b1 = initialize_mat(n, n);
        int **mat_b3 = initialize_mat(n, n);
        int **mat_b4 = initialize_mat(n, n);
        int **mat_b6 = initialize_mat(n, n);
        int **mat_b7 = initialize_mat(n, n);


		int **mat_one_11 = new int*[n];
        int **mat_two_11 = new int*[n];
        int **res_11 = new int*[n]; 

        int **mat_one_12 = new int*[n];
        int **mat_two_12 = new int*[n];
        int **res_12= new int*[n];
       
		int **mat_one_21 = new int*[n]; 
        int **mat_two_21 = new int*[n]; 
        int **res_21 = new int*[n];

        int **mat_one_22 = new int*[n];
        int **mat_two_22 = new int*[n];
		int **res_22 = new int*[n]; 
        
       

		update_indices(mat_one_11, n, first_matrix,  0,  0); 
		update_indices(mat_two_11, n, second_matrix,  0,  0); 
		update_indices(res_11, n, res_mat,  0,  0);
		
		update_indices(mat_one_12, n, first_matrix,  0, n);
		update_indices(mat_two_12, n, second_matrix,  0, n);
		update_indices(res_12, n, res_mat,  0, n);
		
		update_indices(mat_one_21, n, first_matrix, n,  0);
		update_indices(mat_two_21, n, second_matrix, n,  0);
		update_indices(res_21, n, res_mat, n,  0);
		
		
		
		update_indices(mat_one_22, n, first_matrix, n, n);
		update_indices(mat_two_22, n, second_matrix, n, n);
	    update_indices(res_22, n, res_mat, n, n);

		#pragma omp task
		{
			add_matrices(mat_a1, n, n, mat_one_11, mat_one_22);
			add_matrices(mat_b1, n, n, mat_two_11, mat_two_22);
			strassen_mat_mul_algorithm(n, mat_a1, mat_b1, mat1, limit);
		}

		#pragma omp task
		{
			add_matrices(mat_a2, n, n, mat_one_21, mat_one_22);
			strassen_mat_mul_algorithm(n, mat_a2, mat_two_11, mat2, limit);
		}

		#pragma omp task
		{
			diff_matrices(mat_b3, n, n, mat_two_12, mat_two_22);
			strassen_mat_mul_algorithm(n, mat_one_11, mat_b3, mat3, limit);
		}

		#pragma omp task
		{
			diff_matrices(mat_b4, n, n, mat_two_21, mat_two_11);
			strassen_mat_mul_algorithm(n, mat_one_22, mat_b4, mat4, limit);
		}

		#pragma omp task
		{
			add_matrices(mat_a5, n, n, mat_one_11, mat_one_12);
			strassen_mat_mul_algorithm(n, mat_a5, mat_two_22, mat5, limit);
		}

		#pragma omp task
		{
			diff_matrices(mat_a6, n, n, mat_one_21, mat_one_11);
			add_matrices(mat_b6, n, n, mat_two_11, mat_two_12);
			strassen_mat_mul_algorithm(n, mat_a6, mat_b6, mat6, limit);
		}

		#pragma omp task
		{
			diff_matrices(mat_a7, n, n, mat_one_12, mat_one_22);
			add_matrices(mat_b7, n, n, mat_two_21, mat_two_22);
			strassen_mat_mul_algorithm(n, mat_a7, mat_b7, mat7, limit);
		}

		#pragma omp taskwait
		#pragma omp parallel for
		for (int counter1 = 0; counter1 < n; counter1++)
			for (int counter2 = 0; counter2 < n; counter2++) {
				res_11[counter1][counter2] = mat1[counter1][counter2] + mat4[counter1][counter2] - mat5[counter1][counter2] + mat7[counter1][counter2];
				res_21[counter1][counter2] = mat2[counter1][counter2] + mat4[counter1][counter2];
				res_12[counter1][counter2] = mat3[counter1][counter2] + mat5[counter1][counter2];
				res_22[counter1][counter2] = mat1[counter1][counter2] - mat2[counter1][counter2] + mat3[counter1][counter2] + mat6[counter1][counter2];
			}

		

            clear_memory(mat1);
            clear_memory(mat2);
            clear_memory(mat3);
            clear_memory(mat4);
            clear_memory(mat5);
            clear_memory(mat6);
            clear_memory(mat7);

            clear_memory(mat_a1);
            clear_memory(mat_a2);
            clear_memory(mat_a5);
            clear_memory(mat_a6);
            clear_memory(mat_a7);

            clear_memory(mat_b1);
            clear_memory(mat_b3);
            clear_memory(mat_b4);
            clear_memory(mat_b6);
            clear_memory(mat_b7);


			delete[] mat_one_11; 
			delete[] mat_two_11; 
            delete[] res_11; 

            delete[] mat_one_12;
            delete[] mat_two_12; 
            delete[] res_12; 

            delete[] mat_one_21;
            delete[] mat_two_21;
            delete[] res_21; 
            
            delete[] mat_one_22; 
            delete[] mat_two_22;     
			delete[] res_22;  

		}
}

int compute_err(int** mat1, int** mat2, int size) {
    
	int count =0;
	
	
	for (int counter1 = 0; counter1 < size; ++counter1){
		for (int counter2 = 0; counter2 < size; ++counter2){
			if(mat1[counter1][counter2] != mat2[counter1][counter2]){
				count++;
			} 
		}
	}  
	return count;
}



void print_log(int flag, int size, int k1, int max_threads, double time) {
    const char *result = (flag == 0) ? "Correct" : "Incorrect";

    printf("%s: matrix size = %d, K' = %d, # of proc = %d, Exec time = %lf sec, Errors = %d \n",
           result, size, k1, max_threads, time, flag);
}

  
int main(int argc, char* argv[])   
{      
	int k = atoi(argv[1]);
    int k1 = atoi(argv[2]);
    int proc = atoi(argv[3]);
	int limit = pow(2,(k-k1));
	int mat_size = pow(2, k);
	
	int **first_matrix = initialize_mat(mat_size, mat_size);
	int **second_matrix = initialize_mat(mat_size, mat_size);
	int **res_matrix = initialize_mat(mat_size, mat_size);
	int **test_matrix = initialize_mat(mat_size, mat_size); 
	
	srand((unsigned)time(NULL));

	for (int counter1 = 0; counter1 < mat_size; ++counter1)
		for (int counter2 = 0; counter2 < mat_size; ++counter2){
		first_matrix[counter1][counter2] = rand()%100;
		second_matrix[counter1][counter2] = rand()%100;

	}

	omp_set_dynamic(0);    
	omp_set_num_threads(proc);

	double start_time = omp_get_wtime();
	
	#pragma omp parallel
    {
        #pragma omp single
        {
            strassen_mat_mul_algorithm(mat_size, first_matrix, second_matrix, res_matrix, limit);
        }
    }
	
	auto e_time = omp_get_wtime() - start_time;

	traditional_mat_mul(mat_size, first_matrix, second_matrix, test_matrix); 
	
	print_log(compute_err(res_matrix, test_matrix, mat_size), mat_size, k1, omp_get_max_threads(), e_time);


    clear_memory(first_matrix);
    clear_memory(second_matrix);
    clear_memory(res_matrix);
    clear_memory(test_matrix);



	return 0;   
}