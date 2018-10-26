#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>	//is in /usr/include as probably expected
			//libOpenCL.so is in /usr/lib64, does this need special handling?
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS


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


	
	//Creating command queue, copied
	cl_command_queue command_queue;
	command_queue = clCreateCommandQueue ( context, device_id, 0, &error );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create command queue\n" );
		return 1;
	}

	

	unsigned int n;	//number of iterations to run
	unsigned int width, height;	//dimensions of work area
	float central;	//central value at start
	float c;	//diffusion constant

	//TODO: read arguments:
	//first two are width, height in that order
	//-i gives central
	//-d gives c
	//-n gives n
	central = 1000000;
	c = 0.3;
	width = 5;
	height = 6;

	
	//Building OpenCL program
	char * run_heat_diff_src;
	//TODO: Read program from run_heat_diff.cl
	FILE * file = fopen ("run_heat_diff.cl", "r");
	fseek ( file, 0, SEEK_END );
	size_t fS = ftell ( file );
	rewind ( file );
	run_heat_diff_src = (char*) malloc ( 0x100000 );
	fread ( run_heat_diff_src, sizeof(char), fS, file );

	cl_program program;
	program = clCreateProgramWithSource( context, 1, (const char **) &run_heat_diff_src, NULL, &error );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create program\n" );
		return 1;
	}

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

	cl_kernel kernel;
	kernel = clCreateKernel ( program, "run_heat_diff", &error );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create kernel\n" );
		return 1;
	}
	
	
	cl_mem heatmap_buffer_in = clCreateBuffer ( context, CL_MEM_READ_ONLY, sizeof(float) * width*height, NULL, &error );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create input buffer\n" );
		return 1;
	}
	cl_mem heatmap_buffer_out = clCreateBuffer ( context, CL_MEM_WRITE_ONLY, sizeof(float) * width*height, NULL, &error );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create output buffer\n" );
		return 1;
	}
	cl_mem c_buffer = clCreateBuffer ( context, CL_MEM_READ_ONLY, sizeof(float), NULL, &error );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create buffer\n" );
		return 1;
	}
	cl_mem height_buffer = clCreateBuffer ( context, CL_MEM_READ_ONLY, sizeof(int), NULL, &error );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create buffer\n" );
		return 1;
	}
	cl_mem width_buffer = clCreateBuffer ( context, CL_MEM_READ_ONLY, sizeof(int), NULL, &error );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot create buffer\n" );
		return 1;
	}
	

	// Initialize
	float * heatmap = (float*)calloc ((width+2)*(height+2), sizeof(float)); //[width+2][height+2] for padding with zeroes? [width*height] for 1-dimensional? malloc?
	heatmap[(width/2+1)    *height + height/2+1]     += central/4;	//does this work? Is it the right way to do it?
	heatmap[((width-1)/2+1)*height + height/2+1]     += central/4;
	heatmap[(width/2+1)    *height + (height-1)/2+1] += central/4;
	heatmap[((width-1)/2+1)*height + (height-1)/2+1] += central/4;


	//Enqueue write buffer = ?
	//error = clEnqueueWriteBuffer ( command_queue, input_buffer_a, CL_TRUE, 0, ix_m*sizeof(float), a, 0, NULL, NULL);
	//input_buffer_a = clCreateBuffer ( context, CL_MEM_READ_ONLY, sizeof(float) * ix_m, NULL, &error);
	
	error = clEnqueueWriteBuffer ( command_queue, heatmap_buffer_in, CL_TRUE, 0, sizeof(float) * width*height, heatmap, 0, NULL, NULL );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot enqueue write buffer\n" );
		return 1;
	}
	error = clEnqueueWriteBuffer ( command_queue, c_buffer, CL_TRUE, 0, sizeof(float), &c, 0, NULL, NULL );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot enqueue write buffer\n" );
		return 1;
	}
	error = clEnqueueWriteBuffer ( command_queue, height_buffer, CL_TRUE, 0, sizeof(int), &height, 0, NULL, NULL );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot enqueue write buffer\n" );
		return 1;
	}
	error = clEnqueueWriteBuffer ( command_queue, width_buffer, CL_TRUE, 0, sizeof(int), &width, 0, NULL, NULL );
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot enqueue write buffer\n" );
		return 1;
	}

	error = clSetKernelArg ( kernel, 0, sizeof(cl_mem), &heatmap_buffer_in );	//TODO: Error handling?
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot set input args\n" );
		return 1;
	}
	error = clSetKernelArg ( kernel, 1, sizeof(cl_mem), &heatmap_buffer_out);
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot set output args\n" );
		return 1;
	}
	error = clSetKernelArg ( kernel, 2, sizeof(cl_mem), &c_buffer);
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot set output args\n" );
		return 1;
	}
	error = clSetKernelArg ( kernel, 3, sizeof(cl_mem), &height_buffer);
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot set output args\n" );
		return 1;
	}
	error = clSetKernelArg ( kernel, 4, sizeof(cl_mem), &width_buffer);
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Cannot set output args\n" );
		return 1;
	}

	const size_t offset[2] = {1,1};
	const size_t work_size[2] = {height, width};

//ACTUALLY RUN THE PROGRAM!!!!

	clEnqueueNDRangeKernel ( command_queue, kernel, 2, offset, work_size, NULL, 0, NULL, NULL);

//Done

	float * heatmap_out = malloc ( width*height*sizeof(float) );
	clEnqueueReadBuffer ( command_queue, heatmap_buffer_out, CL_TRUE, 0, width*height*sizeof(float), heatmap_out, 0, NULL, NULL );//TODO: Error handling, ordering

	clFlush ( command_queue );
	clFinish ( command_queue );//TODO: Error handling
	if ( error != CL_SUCCESS ) {
		fprintf ( stderr, "Error from clFinish\n" );
		return 1;
	}


	clReleaseKernel(kernel); //later
	clReleaseProgram(program);

	clReleaseMemObject ( heatmap_buffer_in );
	clReleaseMemObject ( heatmap_buffer_out );
	clReleaseMemObject ( c_buffer );
	clReleaseMemObject ( height_buffer );
	clReleaseMemObject ( width_buffer );
	clReleaseCommandQueue ( command_queue );
	clReleaseContext(context);

	for ( size_t ix=0; ix < height; ++ix ) {
		for ( size_t jx=0; jx < width; ++jx )
			printf("%8f ", heatmap_out[ix+height*jx]);
		printf("\n");
	}

	free (heatmap);
	free (heatmap_out);


	return 0;
}
