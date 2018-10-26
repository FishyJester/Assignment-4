__kernel void
run_heat_diff(
	      __global const float * heatmap_in,	//float or double, * or **?
	      __global float * heatmap_out,
	      __global float * c,
	      __global int * height,
	      __global int * width
	      )
{
	int ix = get_global_id(0);	//heightwise
	int jx = get_global_id(1);	//widthwise
	int kx = ix + *height * jx;
	heatmap_out[kx-*height-1] = heatmap_in[kx] +
			  *c * (heatmap_in[kx-1] +
				heatmap_in[kx+1] +
				heatmap_in[kx-*height] +
				heatmap_in[kx+*height] -
				heatmap_in[kx]
				);

}

