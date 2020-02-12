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

#define DSGD_Z2BLOCK 0
#define DSGD_Z2SIZE  1

#pragma GCC diagnostic ignored "-Wwrite-strings"


//to compile: mpicxx dsgd_mf.cpp -Wall -I/usr/local/include -L/usr/local/lib  -lgsl -lgslcblas -lm -o dsgd_mf
//to run: mpirun -n 4 ./dsgd_mf 3883 6040 "rank" "maxiter" "eta" "writeOutput"
//eg: mpirun -n 4 ./dsgd_mf 3883 6040 10 20 0.00001 0

//We define a structure to represent a single element of the observed 'sparse' matrix
typedef struct mtxElm
{
    int i;
    int j;
    double val;

} mtxElm;



//Function signatures (no need to modify)
int loadDataSparse(char * fileName, mtxElm * data, int * size1, int * size2);
void customGSLmatrixAlloc(gsl_matrix * M,int siz1, int siz2, int tda, double * data, int siz);
void writeDataGSLBin(char* fName, gsl_matrix * M);



//Functions to be completed (see below)
void initalizeGSLMatrix(gsl_matrix * M, double mult);
void computeDivDiff(mtxElm * X, int dataBlockSize, mtxElm * div_term, gsl_matrix * Z1, gsl_matrix * Z2);
void computeZ1update(mtxElm * div_term, int dataBlockSize, gsl_matrix * Z2, gsl_matrix * Z1update);
void computeZ2update(mtxElm * div_term, int dataBlockSize, gsl_matrix * Z1, gsl_matrix * Z2update);


int main(int argc, char ** argv)
{
	
	//char * dataPath = argv[1];
    int s1 = atoi(argv[1]); 
    int s2 = atoi(argv[2]);
    int s3 = atoi(argv[3]);
    int MaxIter   = atoi(argv[4]);
    double eta = atof(argv[5]);

    double writeOutput = false;
    if(atoi(argv[6])==1)
        writeOutput = true;

    //don't need to modify here
    const int MaxBlockSize = 5000000;


	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

  	// Get the number of processes
	int numBlocks;
	MPI_Comm_size(MPI_COMM_WORLD, &numBlocks);

	// Get the rank of the process
	int processId;
	MPI_Comm_rank(MPI_COMM_WORLD, &processId);

	// Random seed
	srand(numBlocks * time(NULL));
	

	// Load the initial data block 
	char * dataPath = "./data/";
	


	char outPath[1024];
	sprintf(outPath,"./output/%d_%d_%d/",s1,s2,s3);
	
	if(processId == 0)
	{	
		//create the folder if it doesn't exist
    	char cmd[1024];
    	sprintf(cmd,"mkdir %s",outPath);
    	system(cmd);
    }
	
	//This data structure will keep the data block as a list of entries
	mtxElm * tempX = (mtxElm *)malloc(MaxBlockSize * sizeof(mtxElm));


	//Determine which data block will be used
	int blockI1 = processId; //the index in the first dimension
	int blockI2 = processId; //the index in the second dimension


	char fileName[1024];
	sprintf(fileName,"%s/blocks_%d_%d_%d/X%d_%d.dat",dataPath,s1,s2,numBlocks,blockI1,blockI2);
	int curS1, curS2;
	int dataBlockSize = loadDataSparse(fileName,tempX, &curS1, &curS2);

	printf("Process%d: Initial data loaded\n",processId);


	// Allocate the matrices, it is a bit different than before, no need to change
	int fact2BlockSize = 2* ceil(s2/numBlocks) * s3; //paranoid size

	//initialize the first factor block
	gsl_matrix * Z1block = gsl_matrix_calloc(curS1,s3);
	
	//initialize the second factor block
	gsl_matrix * Z2block = (gsl_matrix *)malloc(sizeof(gsl_matrix));
	Z2block->block = (gsl_block*)malloc(sizeof(gsl_block));
	double * Z2data = (double*)malloc(fact2BlockSize*sizeof(double));

	customGSLmatrixAlloc(Z2block,s3,curS2,curS2, Z2data, fact2BlockSize );



	//Initialize the factor blocks
	double mult = 1.5;
	initalizeGSLMatrix(Z1block, mult);
	initalizeGSLMatrix(Z2block, mult);

	//Allocate the temporary buffer, required during the communications
	double * tempBuf = (double*)malloc(fact2BlockSize*sizeof(double));

	//Allocate the elements that will be used for computing the gradients
	mtxElm * div_term = (mtxElm *)malloc(MaxBlockSize * sizeof(mtxElm));

	gsl_matrix * Z1update = gsl_matrix_calloc (curS1, s3);    
	
	gsl_matrix * Z2update = (gsl_matrix *)malloc(sizeof(gsl_matrix));
	Z2update->block = (gsl_block*)malloc(sizeof(gsl_block));
	double * Z2UpdateData = (double*)malloc(fact2BlockSize*sizeof(double));

	customGSLmatrixAlloc(Z2update,s3,curS2,curS2, Z2UpdateData, fact2BlockSize ); 

	//For computing computation times
	double t1, t2; 
	double tElapsed = 0;
	
	//Start the gradient descent!
	for(int ee = 0; ee<MaxIter; ee++)
	{
		t1 = MPI_Wtime(); 

		if(ee>0)
		{	
			//Read the appropriate data block
			sprintf(fileName,"%s/blocks_%d_%d_%d/X%d_%d.dat",dataPath,s1,s2,numBlocks,blockI1,blockI2);
			dataBlockSize = loadDataSparse(fileName,tempX, &curS1, &curS2);
		}

		//Zero out the gradients of Z1 and Z2
		gsl_matrix_set_all (Z1update, 0);
		gsl_matrix_set_all (Z2update, 0);

		//Compute the gradients
		computeDivDiff(tempX,dataBlockSize,div_term,Z1block,Z2block); //Fill this in

		computeZ1update(div_term, dataBlockSize, Z2block, Z1update); //Fill this in
		computeZ2update(div_term, dataBlockSize, Z1block, Z2update); //Fill this in


		//Update Z1 and Z2 with Gradient Descent

		// TODO
		for (int i = 0; i < curS1; i++) {
			for (int j = 0; j < s3; j++) {
				Z1block->data[i * Z1block->tda + j] += eta * Z1update->data[i * Z1update->tda + j];
			}
		}
		for (int i = 0; i < s3; i++) {
			for (int j = 0; j < curS2; j++){
				Z2block->data[i * Z2block->tda + j] += eta * Z2update->data[i * Z2update->tda + j];
			}
		}


    	t2 = MPI_Wtime();
		tElapsed += t2-t1;

    	if(writeOutput)
		{
			//save outputs
			char outFileName1[1024];
			char outFileName2[1024];

			sprintf(outFileName1,"%s/Z1_%d_iter%d.dat",outPath,blockI1,ee);
			sprintf(outFileName2,"%s/Z2_%d_iter%d.dat",outPath,blockI2,ee);


			writeDataGSLBin(outFileName1, Z1block);
			writeDataGSLBin(outFileName2, Z2block);

		}

		t1 = MPI_Wtime();

		//Communicate the blocks of Z2
		if(ee < (MaxIter-1))
		{
			// synchronize the updated variables
			int dest = (processId - 1 + numBlocks) % numBlocks; //the process id of the destination: send it to the previous process
			int src  = (processId + 1) % numBlocks; //the process id of the source: recieve it from the next process
			

			// first send/receive some information -- no need to modify
			MPI_Status st;

			int Z2info [3];
			Z2info[0] = (int)Z2block->size1;
			Z2info[1] = (int)Z2block->size2;
			Z2info[2] = (int)Z2block->tda;
			
			int Z2infoInc [3];

			MPI_Sendrecv(&(Z2info[0]), 3, MPI_INT,
	                dest, DSGD_Z2SIZE,
	                &(Z2infoInc[0]), 3, MPI_INT,
	                src, DSGD_Z2SIZE,
	                MPI_COMM_WORLD, &st);


			//Now send/receive the Z2 block -- fill in the ??? parts
			MPI_Sendrecv(Z2block->data, fact2BlockSize, MPI_DOUBLE,
	                dest, DSGD_Z2BLOCK,
	                tempBuf, fact2BlockSize, MPI_DOUBLE,
	                src, DSGD_Z2BLOCK,
	                MPI_COMM_WORLD, &st);

			//update the meta info -- no need to modify
			Z2block->size1 = Z2infoInc[0];
			Z2block->size2 = Z2infoInc[1];
			Z2block->tda   = Z2infoInc[2];

			Z2update->size1 = Z2infoInc[0];
			Z2update->size2 = Z2infoInc[1];
			Z2update->tda   = Z2infoInc[2];
			
			memcpy(Z2block->data, tempBuf, fact2BlockSize*sizeof(double));
			
			//form a new part
			blockI1 = processId;
			blockI2 = (processId + 1) % numBlocks;
		}
		
		t2 = MPI_Wtime();
		tElapsed += t2-t1;

		if((ee%10) == 0)
			printf("Iteration: %d\n",ee);


	}
	

	// Finalize the MPI environment.
  	MPI_Finalize();

  	t2 = MPI_Wtime(); 
	long int msec = (long int)(1000*tElapsed);
	printf("*%ld\n",msec);


	//Clean up
	gsl_matrix_free(Z1block);
	gsl_matrix_free(Z2block);
  

	free(tempX);
	free(tempBuf);
    gsl_matrix_free(Z1update);
    gsl_matrix_free(Z2update);
    free(div_term);

 

	return 0;
}

/// Functions to be filled

void initalizeGSLMatrix(gsl_matrix * M, double mult)
{
	//((double)rand() / (((double)RAND_MAX)) ) generates a number in between 0 and 1
	for (int i = 0; i < (int) M->size1; i++) {
		for (int j = 0; j < (int) M->size2; j++) {
			M->data[i * M->tda + j] = ((double)rand() / (((double)RAND_MAX)))*mult;
		}
	}
}


void computeDivDiff(mtxElm * X, int dataBlockSize, mtxElm * div_term, gsl_matrix * Z1, gsl_matrix * Z2)
{
	int s3 = Z1->size2;

	for(int i =0; i<dataBlockSize; i++)
	{
		int curI1   = X[i].i;
		int curI2   = X[i].j;
		double curX = X[i].val;

		double curXhat = 0;
		//Compute the corresponding curXhat
        for(int k=0;k<s3;k++)
        {
        	curXhat-=(Z1->data[curI1 * Z1->tda + k])*(Z2->data[k * Z2->tda + curI2]);
        }
        div_term[i].i = curI1;
        div_term[i].j = curI2;
        div_term[i].val = (curXhat-curX);

	}

}


void computeZ2update(mtxElm * div_term, int dataBlockSize, gsl_matrix * Z1, gsl_matrix * Z2update)
{
	int s3 = Z1->size2;

	for(int i =0; i<dataBlockSize; i++)
	{
		int curI1   = div_term[i].i;
		int curI2   = div_term[i].j;
		double curVal = div_term[i].val;

		//Compute Z2update (the gradient wrt Z2)
		for(int k=0;k<s3;k++){
			Z2update->data[k * Z2update->tda + curI2] += curVal*Z1->data[curI1 * Z1->tda + k];
		}
	}


}


void computeZ1update(mtxElm * div_term, int dataBlockSize, gsl_matrix * Z2, gsl_matrix * Z1update)
{
	int s3 = Z2->size1;

	for(int i =0; i<dataBlockSize; i++)
	{
		int curI1   = div_term[i].i;
		int curI2   = div_term[i].j;
		double curVal = div_term[i].val;

		//Compute Z1update (the gradient wrt Z1)
		for(int k=0;k<s3;k++){
			Z1update->data[curI1 * Z1update->tda + k]+=curVal*Z2->data[k * Z2->tda + curI2];
		}
	}


}


/// Provided Functions

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


void customGSLmatrixAlloc(gsl_matrix * M,int siz1, int siz2, int tda, double * data, int siz)
{

  M->size1 = siz1;
  M->size2 = siz2;
  M->tda   = tda;
  M->data  = data;
  M->block->size = siz;
  M->block->data = data;
  M->owner = 1;

}

void writeDataGSLBin(char* fName, gsl_matrix * M)
{
  FILE *myfile;
  myfile = fopen(fName, "wb");

  int s1 = M->size1;
  int s2 = M->size2;

  for (int j = 0; j<s2; j++)
    {
      for (int i = 0; i<s1; i++)
      {
         //fprintf(myfile,"%20.20lf \n",M->data[i * M->tda + j]);
         fwrite(&(M->data[i * M->tda + j]), sizeof(double), 1, myfile);
      }
    }
    fclose(myfile);
}

