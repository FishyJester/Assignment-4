__kernel void
run_heat_diff(
	      __global const float * heatmap_in,	//float or double, * or **?
	      __global float * heatmap_out
	      )
{
	int ix = get_global_id(0);
	heatmap_out[ix] = heatmap_in[ix] +
			  c * (	heatmap_in[ix-1] +
				heatmap_in[ix+1] +
				heatmap_in[ix-height] +
				heatmap_in[ix+height] -
				heatmap_in[ix]
				);

}

