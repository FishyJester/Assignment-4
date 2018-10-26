#Heat Diffusion, hpcgroup058
This report will describe our implementation of the heat diffusion problem with openCL parallelization. We will describe in broad terms how our code works and our thoughts behind doing what we do. Only critical blocks of code will be discussed and as such we will not discuss the more standard overhead of openCL, such as `clGetPlatformIDs`, unless we've specifically modified it from it's standard form.

##Program description
Note that our program contains a lot of error handling that will be left out in the following description! References to errors in code segments that appear are realted to this and is the standard openCL error checking with nothing special added by our device.

We start by standard openCL overhead: finding platform, device and create a context and a command queue. This is followed by parsing of command line arguments. Here we include padding in our weight and height argument and combine them into a total size:

~~~c
unsigned int width, height;

width = atol(argv[1])+2;
height = atol(argv[2])+2;
unsigned int size = width*height; 
~~~ 

We've made the assumption that arguments are given in correct order in command line.

Continuing we build our openCL program, put in file `run_heat_diff.cl`. We use `fseek` to seek to end of file and `ftell` to return file size. Then we read the file to buffer and create program:

~~~c
run_heat_diff_src = (char*) malloc(0x100000);
cl_program program;
program = clCreateProgramWithSource(context, 1, (const char **) &run_heat_diff_src, NULL, &error);
clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
~~~

The use of the hex in malloc allows us to allocate for maximum file size.

Next we create our kernels. We use two kernels that will run essentially the same code, but write to and read from two different heatmap buffers, `heatmap_buffer_0` and `heatmap_buffer_1`. This will allow us to run the iterations efficiently by writing initial data into one buffer, do the diffusion calculations and store them in the other buffer, which is then used to do the next iteration of calculations and store it into the first buffer, and so on. We also create buffers for the height, width and diffusion constant to be used in the diffusion calculations:

~~~c
cl_kernel kernel_0, kernel_1;
kernel_0 = clCreateKernel (program, "run_heat_diff_0", &error);
kernel_1 = clCreateKernel (program, "run_heat_diff_1", &error);

cl_mem heatmap_buffer_0 = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &error);
cl_mem heatmap_buffer_1 = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &error);

// Create buffers for diffusion constant c, height and weight
~~~

`run_heat_diff_1` and `run_heat_diff_0` are defined inside `run_heat_diff.cl` which we will look at later.

After creating the buffers we initialize the heatmap to be written to `heatmap_buffer_0`. Here `central` is the central start temperature parsed from command line:

~~~c
float * heatmap = calloc(size, sizeof(float));
heatmap[(width/2) * height + height/2] += central/4;
heatmap[(width-1)/2 * height + height/2] += central/4;
heatmap[(width/2) * height + (height-1)/2] += central/4;
heatmap[(width-1)/2 * height + (height-1)/2] += central/4;
~~~

Initializing in this way allow us to spread out the value amongst the centerpoints in the case of even widths and heights. Notice also that we traverse down columns and not along rows if thinking matrix-wise. As we are only working with a single array this has no impact on row major order however.

Next we enqueue the write buffers, writing the above initialized heatmap to `heatmap_buffer_0` which will start the iteration. We also enqueue the height, width and diffusion constant buffers. We continue by setting kernel arguments, giving both `kernel_0` and `kernel_1` access to both heatmap buffers and weight, height and diffusion constant.

Next we actually run the program, `iter` is the number of iterations, parsed from the command line:

~~~c
const size_t offset[2] = {1,1};
const size_t work_size[2] = {height-2, width-2};

for (size_t ix = 0; ix < iter-1; ix += 2){
	clEnqueueNDRangeKernel(command_queue, kernel_0, 2, offset, work_size, NULL, 0, NULL, NULL);
	clEnqueueNDRangeKernel(command_queue, kernel_1, 2, offset, work_size, NULL, 0, NULL, NULL);
}
float *heatmap_out = malloc(size * sizeof(float));
if(iter % 2 == 1){
	clEnqueueNDRangeKernel(command_queue, kernel_0, 2, offset, work_size, NULL, 0, NULL, NULL);
	clEnqueueReadBuffer(command_queue, heatmap_buffer_1, CL_TRUE, 0, size * sizeof(float), heatmap_out, 0, NULL, NULL);
} else {
	clEnqueueReadBuffer(command_queue, heatmap_buffer_0, CL_TRUE, 0, size * sizeof(float), heatmap_out, 0, NULL, NULL);
}
~~~

As seen we set offset to pad the borders and work size to give the amount of coordinates that needs to be treated, adjusted for our initial padding of weight and height. As described earlier kernel 0 will use data stored in heatmap 0 and store the resulting calculations in heatmap 1 which will then repeat the same procedure in the opposite direction. If we have an even number of iterations the for loop will stop one iteration short, having stored data in heatmap 0. If this is the case we perform the last iteration (which will then be conducted by kernel 0, storing data in heatmap 1) and write data from heatmap 1 to heatmap_out. Otherwise all iterations are accounted for inside the loop, finishing with kernel 1 (storing data in heatmap 0). So we read buffer from heatmap 0 to heatmap_out.

Now we'll take a closer look at `run_heat_diff.cl` where `run_heat_diff_0` and `run_heat_diff_1` are defined. As both of these are essentially the same except for which heatmap is read from and which is written to, we will present the code for `run_heat_diff_0`, to get `run_heat_diff_1` it's enough to swap the index `i` in `heatmap_i`:


~~~c
__kernel void
run_heat_diff_0(
	__global float * heatmap_0,
	__global float * heatmap_1,
	__global float * c, //Diffusion constant
	__global int * height
	)
{
	int ix = get_global_id(0);
	int jx = get_global_id(1);
	int kx = ix +*height * jx;
	heatmap_1[kx] = heatmap_0[kx] +
			*c * ((heatmap_0[kx - 1] +
				heatmap_0[kx + 1] +
				heatmap_0[kx - *height] +
				heatmap_0[kx + *height])
				/4
				-heatmap_0[kx]
				);
}
~~~

The calculations are straight forward. Notice again that we consider it as down columns when thinking matrix-wise, `get_global_id(0)` being height-wise index and `get_global_id(1)` being width-wise.

When calculations are done and data has been written to `heatmap_out` we release kernels, program, buffers, queue and context. Then we loop through the `heatmap_out` buffer and calculate mean heat and the mean absolute difference between the individual points' heat and the mean heat:

~~~c
float total_heat = 0;
float abs_diff_m = 0;

for(size_t ix = 1; ix < height-1; ++ix){
	for(size_t jx = 1; jx < width-1; ++jx)
		total_heat += heatmap_out[ix+height*jx];

}
float temp_mean = total_heat/((height-2)*(width-2));

for(size_t ix = 1; ix < height-1; ++ix)
	for(size_t jx = 1; jx < width-1; ++jx){
		float diff = heatmap_out[ix+height*jx] - temp_mean;
	
		if(diff > 0)
			abs_diff_m += diff;
		else
			abs_diff_m -= diff;
}

abs_diff_m /= (width-2)*(height-2);
~~~

We use the if statement to avoid having to include `math.h` to use `abs`. It is possible this will also add performance over using `abs`, although any performance hit from using one over the other in this block of code should be negligible.
		
