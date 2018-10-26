__kernel void
run_heat_diff_0(
	      __global float * heatmap_0,	//float or double, * or **?
	      __global float * heatmap_1,
	      __global float * c,
	      __global int * height,
	      __global int * width
	      )
{
	int ix = get_global_id(0);	//heightwise
	int jx = get_global_id(1);	//widthwise
	int kx = ix + *height * jx;
	heatmap_1[kx] = heatmap_0[kx] +
		  *c * ((heatmap_0[kx-1] +
			heatmap_0[kx+1] +
			heatmap_0[kx-*height] +
			heatmap_0[kx+*height])
			/ 4
			- heatmap_0[kx]
			);
}

__kernel void
run_heat_diff_1(
	      __global float * heatmap_0,	//float or double, * or **?
	      __global float * heatmap_1,
	      __global float * c,
	      __global int * height,
	      __global int * width
	      )
{
	int ix = get_global_id(0);	//heightwise
	int jx = get_global_id(1);	//widthwise
	int kx = ix + *height * jx;
	heatmap_0[kx] = heatmap_1[kx] +
		  *c * ((heatmap_1[kx-1] +
			heatmap_1[kx+1] +
			heatmap_1[kx-*height] +
			heatmap_1[kx+*height])
			/ 4
			- heatmap_1[kx]
			);
}

