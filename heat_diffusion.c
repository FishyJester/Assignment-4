#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>	//is in /usr/include as probably expected
			//libOpenCL.so is in /usr/lib64, does this need special handling?
int main(int argc, char *argv[]){
	
	//Get Platform, stolen from lecture slide
	cl_int error;
	cl_platform_id platform_id;
	cl_uint nmb_platforms;
	if ( clGetPlatformIDs (1, &platform_id, &nmb_platforms) != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot get platform\n" );
		return 1;
	}
	
	//Get device, stolen likewise
	cl_device_id device_id;
	cl_uint nmb_devices;
	if ( clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &nmb_devices) != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot get device\n" );
		return 1;
	}

	//Get context, by thievery
	cl_context context;
	cl_context_properties properties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties) platform_id,
		0
	};
	context = clCreateContext ( properties, 1, &device_id, NULL, NULL, &error);
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create context\n" );
		return 1;
	}


	//TODO: clReleaseContext(context); //somewhere
	
	//Creating command queue, copied
	cl_command_queue command_queue;
	command_queue = clCreateCommandQueue ( context, device_id, 0, &error );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create command queue\n" );
		return 1;
	}

	//TODO: clReleaseCommandQueue ( command_queue ); //at the end of the program?
	
	//Enqueue write buffer = ?
	//error = clEnqueueWriteBuffer ( command_queue, input_buffer_a, CL_TRUE, 0, ix_m*sizeof(float), a, 0, NULL, NULL);
	//input_buffer_a = clCreateBuffer ( context, CL_MEM_READ_ONLY, sizeof(float) * ix_m, NULL, &error);

	unsigned int n;	//number of iterations to run
	unsigned int width, height;	//dimensions of work area
	double central;	//central value at start

	//read arguments:
	//first two are width, height in that order
	//-i gives central
	//-d gives c
	//-n gives n
	
	//Building OpenCL program
	char * run_heat_diff_src;
	//TODO: Read program from run_heat_diff.cl
	cl_program program;
	program = clCreateProgramWithSource( context, 1, (const char **) &run_heat_diff_src, NULL, &error );
	//TODO: Error handling
	
	clBuildProgram ( program, 0, NULL, NULL, NULL, NULL );
	//TODO: Error handling
	
	cl_kernel kernel;
	kernel = clCreateKernel ( program, "run_heat_diff", &error );
	//TODO: Error handling
	
	//TODO: clReleaseProgram(program); clReleaseKernel(kernel); //later
	
	error = clBuildProgram ( program, 0, NULL, NULL, NULL, NULL );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot build program. Log:\n" );

		size_t log_size = 0;
		clGetProgramBuildInfo ( program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size );

		char * log = calloc ( log_size, sizeof(char) );
		if (log == NULL) {
			fprintf( stderr, "Could not allocate memory\n" );
			return 1;
		}
		clGetProgramBuildInfo ( program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL );
		fprintf ( stderr, "%s\n", log );

		free (log);
		return 1;
	}
	heatmap_buffer_in = clCreateBuffer ( context, CL_MEM_READ_ONLY, sizeof(float) * width*height, NULL, &error );
	//TODO: Error handling
	heatmap_buffer_out = clCreateBuffer ( context, CL_MEM_WRITE, sizeof(float) * width*height, NULL, &error );
	//TODO: Error handling
	
	//TODO: clReleaseMemObject ( heatmap_buffer_in ); clReleaseMemObject ( heatmap_buffer_out );

	// Initialize: TODO: Fix
	double heatmap[width][height]={0.0}; //[width+2][height+2] for padding with zeroes? [width*height] for 1-dimensional? malloc?
	heatmap[width/2]    [height/2]     += central/4;	//does this work? Is it the right way to do it?
	heatmap[(width-1)/2][height/2]     += central/4;
	heatmap[width/2]    [(height-1)/2] += central/4;
	heatmap[(width-1)/2][(height-1)/2] += central/4;

	//TODO: Everything from slide ENQUEUEING COMMANDS and forward


	return 0;
}
