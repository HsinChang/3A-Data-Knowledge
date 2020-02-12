#include <mpi.h>

#include <sstream>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#pragma GCC diagnostic ignored "-Wwrite-strings"



//to compile: mpicxx compute_rmse.cpp -Wall -I/usr/local/include -L/usr/local/lib  -lgsl -lgslcblas -lm -o compute_rmse
//to run: compute_rmse -n "numProcesses" ./compute_rmse 3883 6040 "rank" "maxiter" "numberOfBlocks"
//eg: mpirun -n 10 ./compute_rmse 3883 6040 10 20 4 

typedef struct mtxElm
{
    int i;
    int j;
    double val;

} mtxElm;

int loadDataSparse(char * fileName, mtxElm * data, int * size1, int * size2);
void loadFactorBin(char * fName, gsl_matrix * M);
double sparseRmse(mtxElm * X, int dataBlockSize, gsl_matrix * Z1, gsl_matrix * Z2);

int main(int argc, char ** argv)
{
	
	
    int s1 = atoi(argv[1]);
    int s2 = atoi(argv[2]);
    int s3 = atoi(argv[3]);
    int MaxIter   = atoi(argv[4]);
    int numBlocks   = atoi(argv[5]);

  
	mtxElm * X = (mtxElm *)malloc(21000000 * sizeof(mtxElm));
	


	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

  	// Get the number of processes
	int numProcesses;
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	

	// Load the whole data 
	char * dataPath = "./data/";
	char * outPath  = "./output/"; //where the output of the algorithm is
    char * outFilePath = "./rmse/";

 	char fileName[1024];
 	
 	sprintf(fileName,"%s/X_%d_%d.dat",dataPath,s1,s2);
 	
	
	int temp1,temp2;
	int dataSize = loadDataSparse(fileName,X, &temp1, &temp2);



	printf("Data loaded\n");

	// Get the rank of the process
	int processId;
	MPI_Comm_rank(MPI_COMM_WORLD, &processId);

	
	// Allocate factors
	gsl_matrix * Z1 = gsl_matrix_calloc (s1, s3);
	gsl_matrix * Z2 = gsl_matrix_calloc (s3, s2);


	int num_iter_per_process = MaxIter/numProcesses;
	int start_ix = processId*num_iter_per_process;
	int end_ix   = start_ix + num_iter_per_process;


	FILE * fOut; 
	char outFileName[1024];
	sprintf(outFileName,"%s/%d.txt",outFilePath,processId);
	fOut = fopen(outFileName, "w");


	int * z1sizes = (int *)malloc(numBlocks*sizeof(int));
	int * z2sizes = (int *)malloc(numBlocks*sizeof(int));	


	for(int i = 0; i<numBlocks; i++)
	{
		int tempS1, tempS2, tempVal;

		FILE * fTemp;
		char tempFileName[1024];
		
		sprintf(tempFileName,"%s/blocks_%d_%d_%d/X0_%d.dat",dataPath,s1,s2,numBlocks,i);

		

		fTemp = fopen(tempFileName, "r");
		fscanf(fTemp,"%d %d %d",&tempS1,&tempS2,&tempVal);

		z2sizes[i] = tempS2;


		fclose(fTemp);

		
		sprintf(tempFileName,"%s/blocks_%d_%d_%d/X%d_0.dat",dataPath,s1,s2,numBlocks,i);

		fTemp = fopen(tempFileName, "r");
		fscanf(fTemp,"%d %d %d",&tempS1,&tempS2,&tempVal);

		z1sizes[i] = tempS1;

		fclose(fTemp);

	}



 
	for(int ee = start_ix; ee<end_ix; ee++)
	{
		//read Z1 Z2
 		
 		int i1ix = 0;
 		int i2ix = 0;

		for(int bb = 0; bb<numBlocks; bb++)
		{
			char factFileName [1024];

			sprintf(factFileName,"%s/%d_%d_%d/Z1_%d_iter%d.dat",outPath,s1,s2,s3,bb,ee);
		
			gsl_matrix_view Z1part =  gsl_matrix_submatrix (Z1, i1ix, 0, z1sizes[bb], s3);
			loadFactorBin(factFileName, &(Z1part.matrix));
			i1ix += z1sizes[bb]; 

			sprintf(factFileName,"%s/%d_%d_%d/Z2_%d_iter%d.dat",outPath,s1,s2,s3,bb,ee);
			gsl_matrix_view Z2part =  gsl_matrix_submatrix (Z2,  0, i2ix,  s3, z2sizes[bb]);
			loadFactorBin(factFileName, &(Z2part.matrix));
			i2ix += z2sizes[bb];

	
		}

		double rmse = sparseRmse(X, dataSize, Z1, Z2);    	

		fprintf(fOut,"%20.20lf\n",rmse);


		//if((ee%100) == 0)
			printf("Iteration: %d\n",ee);

	}




	// Finalize the MPI environment.
  	MPI_Finalize();

  	fclose (fOut);
  	free(z1sizes);
  	free(z2sizes);
	

  	gsl_matrix_free(Z1);
	gsl_matrix_free(Z2);
	free(X);

	// if(processId == 0)
	// {	
	// 	//create the folder if it doesn't exist
 //    	char cmd[1024];
 //    	sprintf(cmd,"rmdir %s/%d_%d_%d",outPath,s1,s2,s3);
 //    	system(cmd);
 //    }


	return 0;
}

double sparseRmse(mtxElm * X, int dataBlockSize, gsl_matrix * Z1, gsl_matrix * Z2)
{
  int s3 = Z1->size2;

  double rmse = 0;

  for(int i =0; i<dataBlockSize; i++)
  {
    int curI1   = X[i].i;
    int curI2   = X[i].j;
    double curX = X[i].val;

    double curXhat = 0;
    for(int k = 0; k<s3; k++)
    {
        curXhat = curXhat +  Z1->data[curI1 * Z1->tda + k ] * Z2->data[k* Z2->tda + curI2];
    }
    curXhat = fmax(curXhat,1e-10);

    double curdiv = curXhat - curX;
    rmse = rmse + curdiv*curdiv;
    
  }

  rmse = sqrt(rmse/dataBlockSize);

  return rmse;

}







void loadFactorBin(char * fName, gsl_matrix * M)
{
    FILE *myfile;
    myfile=fopen(fName, "rb");

    
    for(unsigned int j = 0; j< M->size2; j ++)
    {
        for(unsigned int i = 0; i < M->size1; i++)
        {
            double tempNum;
            fread(&tempNum, sizeof(double), 1, myfile);
            M->data[i * M->tda + j] = tempNum;

        }

    }
  

    fclose(myfile);

}


int loadDataSparse(char * fileName, mtxElm * data, int * size1, int * size2)
{
    // INDEXING STARTS AT 0!!!
    
    FILE *myfile;
    myfile=fopen(fileName, "r");

    int s1,s2,numel;
    
    fscanf(myfile,"%d %d %d",&s1,&s2,&numel);

    for(int i = 0; i < numel; i++)
    {
        double tempi, tempj, tempval;
        fscanf(myfile,"%lf %lf %lf",&tempi,&tempj,&tempval);
        
        data[i].i   = tempi-1;
        data[i].j   = tempj-1;
        data[i].val = tempval;
        
    }

    fclose(myfile);

    *size1 = s1;
    *size2 = s2;

    return numel;

}
