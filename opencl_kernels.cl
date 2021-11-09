/*

CUDA <-> OpenCL

//////////////////////////////////////////////////
    typedef struct cuda_id_struct
    {
        int x;
        int y;
    } cuda_id;

    cuda_id threadIdx, blockIdx, blockDim;
    threadIdx.x = get_local_id(0);
    threadIdx.y = get_local_id(1);
    blockIdx.x  = get_group_id(0);
    blockIdx.y  = get_group_id(1);
    blockDim.x  = get_local_size(0);
    blockDim.y  = get_local_size(1);


    __syncthreads() <->  barrier(CLK_LOCAL_MEM_FENCE);

*/


__kernel void kernel_copy(__global unsigned char* gInput,
                                 __global unsigned char* gOutput,
                                 __constant int *filter_coeffs,
								 int imgWidth,
								 int imgWidthF)
{
    int PX = get_group_id(0)*get_local_size(0)+get_local_id(0);
    int PY = get_group_id(1) * get_local_size(1) + get_local_id(1);
  
    
    gOutput[3 * PY * imgWidth + 3 * PX] = gInput[3 * (PY + 2) * imgWidthF + 3 * (PX + 2)];
    gOutput[3 * PY * imgWidth + 3 * PX + 1] = gInput[3 * (PY + 2) * imgWidthF + 3 * (PX + 2) + 1];
    gOutput[3 * PY * imgWidth + 3 * PX + 2] = gInput[3 * (PY + 2) * imgWidthF + 3 * (PX + 2) + 2];

}


//----------------------------------------------------------------------------------------------------------------------------

__kernel void kernel_conv_global(__global unsigned char* gInput,
                                 __global unsigned char* gOutput,
                                 __constant int *filter_coeffs,
								 int imgWidth,
								 int imgWidthF)
{
    int acc[3] = { 0,0,0 };

    int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int PY = get_group_id(1) * get_local_size(1) + get_local_id(1);

    for (int fy = 0; fy < 5; fy++)
    {
        for (int fx = 0; fx < 5; fx++)
        {
            for (int rgb = 0; rgb < 3; rgb++)
            {
                acc[rgb] += gInput[3 * (PY + fy) * imgWidthF + 3 * (PX + fx) + rgb] * filter_coeffs[fy * 5 + fx];
            };
        };
    };
    for (int sat_controller = 0; sat_controller < 3; sat_controller++)
    {
        if (acc[sat_controller] < 0)
            acc[sat_controller] = 0;
        if (acc[sat_controller] > 255)
            acc[sat_controller] = 255;
    };
    gOutput[3 * PY * imgWidth + 3 * PX] = acc[0];
    gOutput[3 * PY * imgWidth + 3 * PX + 1] = acc[1];
    gOutput[3 * PY * imgWidth + 3 * PX + 2] = acc[2];
}


//----------------------------------------------------------------------------------------------------------------------------


__kernel void kernel_conv_sh_uchar_int(__global unsigned char* gInput,
	__global unsigned char* gOutput,
	__constant int* filter_coeffs,
	int imgWidth,
	int imgWidthF)
{
	__local int sh_mem[20][20][3];
	int th_load_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int start_id = 3 * (get_group_id(1) * get_local_size(1) * imgWidthF) + 3 * (get_group_id(0) * get_local_size(0));

	for (int loader = 0; loader < 5; loader++)
	{
		if (th_load_id < 240)
		{
			sh_mem[th_load_id / 60 + loader * 4][(th_load_id / 3) % 20][th_load_id % 3] = gInput[start_id + 4 * loader * 3 * imgWidthF + (th_load_id / 60) * 3 * imgWidthF + th_load_id % 60];
		};
	};
	barrier(CLK_LOCAL_MEM_FENCE);


	int X = get_local_id(0);
	int Y = get_local_id(1);

	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = get_group_id(1) * get_local_size(1) + get_local_id(1);
	
	for (int rgb = 0; rgb < 3; rgb++)
	{
		int acc = 0;
		for (int fy = 0; fy < 5; fy++)
		{
			for (int fx = 0; fx < 5; fx++)
			{
				acc += sh_mem[Y + fy][X + fx][rgb] * filter_coeffs[fy * 5 + fx];
			};
		};
		if (acc > 255)
			acc = 255;
		if (acc < 0)
			acc = 0;
		gOutput[3 * PY * imgWidth + 3 * PX + rgb] = acc;
	};
};



//----------------------------------------------------------------------------------------------------------------------------
__kernel void kernel_conv_sh_uchar_float(__global unsigned char* gInput,
                                         __global unsigned char* gOutput,
                                         __constant float *filter_coeffs,
										 int imgWidth,
                                         int imgWidthF)
{
	__local int sh_mem[20][20][3];
	int th_load_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int start_id = 3 * (get_group_id(1) * get_local_size(1) * imgWidthF) + 3 * (get_group_id(0) * get_local_size(0));

	for (int loader = 0; loader < 5; loader++)
	{
		if (th_load_id < 240)
		{
			sh_mem[th_load_id / 60 + loader * 4][(th_load_id / 3) % 20][th_load_id % 3] = gInput[start_id + 4 * loader * 3 * imgWidthF + (th_load_id / 60) * 3 * imgWidthF + th_load_id % 60];
		};
	};
	barrier(CLK_LOCAL_MEM_FENCE);


	int X = get_local_id(0);
	int Y = get_local_id(1);

	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = get_group_id(1) * get_local_size(1) + get_local_id(1);

	for (int rgb = 0; rgb < 3; rgb++)
	{
		float acc = 0;
		for (int fy = 0; fy < 5; fy++)
		{
			for (int fx = 0; fx < 5; fx++)
			{
				acc += (float)sh_mem[Y + fy][X + fx][rgb] * filter_coeffs[fy * 5 + fx];
			};
		};
		if (acc > 255)
			acc = 255;
		if (acc < 0)
			acc = 0;
		gOutput[3 * PY * imgWidth + 3 * PX + rgb] = acc;
	};
};



//----------------------------------------------------------------------------------------------------------------------------
__kernel void kernel_conv_sh_float_float(__global unsigned char* gInput,
                                         __global unsigned char* gOutput,
                                         __constant float *filter_coeffs,
										 int imgWidth,
                                         int imgWidthF)
{
	__local float sh_mem[20][20][3];
	int th_load_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int start_id = 3 * (get_group_id(1) * get_local_size(1) * imgWidthF) + 3 * (get_group_id(0) * get_local_size(0));

	for (int loader = 0; loader < 5; loader++)
	{
		if (th_load_id < 240)
		{
			sh_mem[th_load_id / 60 + loader * 4][(th_load_id / 3) % 20][th_load_id % 3] = gInput[start_id + 4 * loader * 3 * imgWidthF + (th_load_id / 60) * 3 * imgWidthF + th_load_id % 60];
		};
	};
	barrier(CLK_LOCAL_MEM_FENCE);


	int X = get_local_id(0);
	int Y = get_local_id(1);

	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = get_group_id(1) * get_local_size(1) + get_local_id(1);

	for (int rgb = 0; rgb < 3; rgb++)
	{
		float acc = 0;
		for (int fy = 0; fy < 5; fy++)
		{
			for (int fx = 0; fx < 5; fx++)
			{
				acc += sh_mem[Y + fy][X + fx][rgb] * filter_coeffs[fy * 5 + fx];
			};
		};
		if (acc > 255)
			acc = 255;
		if (acc < 0)
			acc = 0;
		gOutput[3 * PY * imgWidth + 3 * PX + rgb] = acc;
	};
};



//----------------------------------------------------------------------------------------------------------------------------
//32*8 a felettiek 16*16
__kernel void kernel_conv_sh_float_float_nbc(__global unsigned char* gInput,
                                             __global unsigned char* gOutput,
                                             __constant float *filter_coeffs,
											 int imgWidth,
                                             int imgWidthF)

{
	__local float sh_mem[12][36][3];
	int th_load_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int start_id = 3 * (get_group_id(1) * get_local_size(1) * imgWidthF) + 3 * (get_group_id(0) * get_local_size(0));

	for (int loader = 0; loader < 6; loader++)
	{
		if (th_load_id < 216)
		{
			sh_mem[th_load_id / (36*3) + loader * 2][(th_load_id / 3) % 36][th_load_id % 3] = gInput[start_id + 2 * loader * 3 * imgWidthF + (th_load_id / (36*3)) * 3 * imgWidthF + th_load_id % (36*3)];
		};
	};
	barrier(CLK_LOCAL_MEM_FENCE);


	int X = get_local_id(0);
	int Y = get_local_id(1);

	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = get_group_id(1) * get_local_size(1) + get_local_id(1);

	for (int rgb = 0; rgb < 3; rgb++)
	{
		float acc = 0;
		for (int fy = 0; fy < 5; fy++)
		{
			for (int fx = 0; fx < 5; fx++)
			{
				acc += sh_mem[Y + fy][X + fx][rgb] * filter_coeffs[fy * 5 + fx];
			};
		};
		if (acc > 255)
			acc = 255;
		if (acc < 0)
			acc = 0;
		gOutput[3 * PY * imgWidth + 3 * PX + rgb] = acc;
	};
};




//ez mûködik de nem optimalizált
__kernel void kernel_median_c_like(__global unsigned char* gInput,
								   __global unsigned char* gOutput,
								   __constant float* filter_coeffs,
								   int imgWidth,
								   int imgWidthF)

{
	int out[3][25];
	int tmp;

	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = get_group_id(1) * get_local_size(1) + get_local_id(1);

	for(int rgb = 0;rgb < 3; rgb++)
	{
		for (int fy = 0; fy < 5; fy++)
		{
			for (int fx = 0; fx < 5; fx++)
			{
			
					out[rgb][5 * fy + fx] = gInput[3 * ((PY + fy) * imgWidthF + (fx + PX)) + rgb];
			
			};
		};

	
		for (int p = 1; p < 25; p += p)
		{
			for (int k = p; k > 0; k = k / 2)
			{
				for (int j = k % p; j + k < 25; j += k + k)
				{
					for (int i = 0; i < k; i++)
					{
						if (((i + j) / (p + p) == (i + j + k) / (p + p)) && ((i + j + k) < 25))
						{
							/*if (out[rgb][i + j] > out[rgb][i + j + k]) //így mûködik
							{
								tmp = out[rgb][i + j + k];
								out[rgb][i + j + k] = out[rgb][i + j];
								out[rgb][i + j] = tmp;
							};
							*/ 
							tmp = max(out[rgb][i + j], out[rgb][i + j + k]);
							out[rgb][i + j]=min(out[rgb][i + j], out[rgb][i + j + k]);
							out[rgb][i + j + k] = tmp;
						};
					};
				};
			};
		};
		gOutput[3 * (PY * imgWidth + PX) + rgb] = (unsigned char)out[rgb][12];
	};

	
}

//Funkcionálisan jó és kb jó is 90%. Gyorsabb mint a bms2-es shared elõtt
__kernel void kernel_median_global(__global unsigned char* gInput,
								__global unsigned char* gOutput,
								__constant float* filter_coeffs,
								int imgWidth,
								int imgWidthF)
{
	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = get_group_id(1) * get_local_size(1) + get_local_id(1);
	
	#pragma unroll
	for (int rgb = 0; rgb < 3; rgb++)
	{
		int loaded_value0 = gInput[3 * ((PY ) * imgWidthF + (PX)) + rgb];
		int loaded_value1 = gInput[3 * ((PY ) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value2 = gInput[3 * ((PY ) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value3 = gInput[3 * ((PY ) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value4 = gInput[3 * ((PY ) * imgWidthF + (4 + PX)) + rgb];

		int loaded_value5 = gInput[3 * ((PY + 1) * imgWidthF + (PX)) + rgb];
		int loaded_value6 = gInput[3 * ((PY + 1) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value7 = gInput[3 * ((PY + 1) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value8 = gInput[3 * ((PY + 1) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value9 = gInput[3 * ((PY + 1) * imgWidthF + (4 + PX)) + rgb];

		int loaded_value10 = gInput[3 * ((PY + 2) * imgWidthF + (PX)) + rgb];
		int loaded_value11 = gInput[3 * ((PY + 2) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value12 = gInput[3 * ((PY + 2) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value13 = gInput[3 * ((PY + 2) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value14 = gInput[3 * ((PY + 2) * imgWidthF + (4 + PX)) + rgb];

		int loaded_value15 = gInput[3 * ((PY + 3) * imgWidthF + (PX)) + rgb];
		int loaded_value16 = gInput[3 * ((PY + 3) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value17 = gInput[3 * ((PY + 3) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value18 = gInput[3 * ((PY + 3) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value19 = gInput[3 * ((PY + 3) * imgWidthF + (4 + PX)) + rgb];

		int loaded_value20 = gInput[3 * ((PY + 4) * imgWidthF + (PX)) + rgb];
		int loaded_value21 = gInput[3 * ((PY + 4) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value22 = gInput[3 * ((PY + 4) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value23 = gInput[3 * ((PY + 4) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value24 = gInput[3 * ((PY + 4) * imgWidthF + (4 + PX)) + rgb];

		//Section 1
		int result0 = min(loaded_value0, loaded_value1);
		int result1 = max(loaded_value0, loaded_value1);

		int result2 = min(loaded_value2, loaded_value3);
		int result3 = max(loaded_value2, loaded_value3);

		int result4 = min(loaded_value4, loaded_value5);
		int result5 = max(loaded_value4, loaded_value5);

		int result6 = min(loaded_value6, loaded_value7);
		int result7 = max(loaded_value6, loaded_value7);

		int result8 = min(loaded_value8, loaded_value9);
		int result9 = max(loaded_value8, loaded_value9);

		int result10 = min(loaded_value10, loaded_value11);
		int result11 = max(loaded_value10, loaded_value11);

		int result12 = min(loaded_value12, loaded_value13);
		int result13 = max(loaded_value12, loaded_value13);

		int result14 = min(loaded_value14, loaded_value15);
		int result15 = max(loaded_value14, loaded_value15);

		int result16 = min(loaded_value16, loaded_value17);
		int result17 = max(loaded_value16, loaded_value17);

		int result18 = min(loaded_value18, loaded_value19);
		int result19 = max(loaded_value18, loaded_value19);

		int result20 = min(loaded_value20, loaded_value21);
		int result21 = max(loaded_value20, loaded_value21);

		int result22 = min(loaded_value22, loaded_value23);
		int result23 = max(loaded_value22, loaded_value23);


		//Section 2
		int tmp = max(result0, result2);
		result0 = min(result0, result2);
		result2 = tmp;

		tmp = max(result1, result3);
		result1 = min(result1, result3);
		result3 = tmp;

		tmp = max(result4, result6);
		result4 = min(result4, result6);
		result6 = tmp;

		tmp = max(result5, result7);
		result5 = min(result5, result7);
		result7 = tmp;

		tmp = max(result8, result10);
		result8 = min(result8, result10);
		result10 = tmp;

		tmp = max(result9, result11);
		result9 = min(result9, result11);
		result11 = tmp;

		tmp = max(result12, result14);
		result12 = min(result12, result14);
		result14 = tmp;

		tmp = max(result13, result15);
		result13 = min(result13, result15);
		result15 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		tmp = max(result17, result19);
		result17 = min(result17, result19);
		result19 = tmp;

		tmp = max(result20, result22);
		result20 = min(result20, result22);
		result22 = tmp;

		tmp = max(result21, result23);
		result21 = min(result21, result23);
		result23 = tmp;



		//Section 3
		tmp = max(result1, result2);
		result1 = min(result1, result2);
		result2 = tmp;

		tmp = max(result5, result6);
		result5 = min(result5, result6);
		result6 = tmp;

		tmp = max(result9, result10);
		result9 = min(result9, result10);
		result10 = tmp;

		tmp = max(result13, result14);
		result13 = min(result13, result14);
		result14 = tmp;

		tmp = max(result17, result18);
		result17 = min(result17, result18);
		result18 = tmp;

		tmp = max(result21, result22);
		result21 = min(result21, result22);
		result22 = tmp;

		//Section 4
		tmp = max(result0, result4);
		result0 = min(result0, result4);
		result4 = tmp;

		tmp = max(result1, result5);
		result1 = min(result1, result5);
		result5 = tmp;

		tmp = max(result2, result6);
		result2 = min(result2, result6);
		result6 = tmp;

		tmp = max(result3, result7);
		result3 = min(result3, result7);
		result7 = tmp;

		tmp = max(result8, result12);
		result8 = min(result8, result12);
		result12 = tmp;

		tmp = max(result9, result13);
		result9 = min(result9, result13);
		result13 = tmp;

		tmp = max(result10, result14);
		result10 = min(result10, result14);
		result14 = tmp;

		tmp = max(result11, result15);
		result11 = min(result11, result15);
		result15 = tmp;

		tmp = max(result16, result20);
		result16 = min(result16, result20);
		result20 = tmp;

		tmp = max(result17, result21);
		result17 = min(result17, result21);
		result21 = tmp;

		tmp = max(result18, result22);
		result18 = min(result18, result22);
		result22 = tmp;

		tmp = max(result19, result23);
		result19 = min(result19, result23);
		result23 = tmp;

		//Section 5
		tmp = max(result2, result4);
		result2 = min(result2, result4);
		result4 = tmp;

		tmp = max(result3, result5);
		result3 = min(result3, result5);
		result5 = tmp;

		tmp = max(result10, result12);
		result10 = min(result10, result12);
		result12 = tmp;

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result18, result20);
		result18 = min(result18, result20);
		result20 = tmp;

		tmp = max(result19, result21);
		result19 = min(result19, result21);
		result21 = tmp;


		//Section 6
		tmp = max(result1, result2);
		result1 = min(result1, result2);
		result2 = tmp;

		tmp = max(result3, result4);
		result3 = min(result3, result4);
		result4 = tmp;

		tmp = max(result5, result6);
		result5 = min(result5, result6);
		result6 = tmp;

		tmp = max(result9, result10);
		result9 = min(result9, result10);
		result10 = tmp;

		tmp = max(result11, result12);
		result11 = min(result11, result12);
		result12 = tmp;

		tmp = max(result13, result14);
		result13 = min(result13, result14);
		result14 = tmp;

		tmp = max(result17, result18);
		result17 = min(result17, result18);
		result18 = tmp;

		tmp = max(result19, result20);
		result19 = min(result19, result20);
		result20 = tmp;

		tmp = max(result21, result22);
		result21 = min(result21, result22);
		result22 = tmp;


		//Section 7
		tmp = max(result0, result8);
		result0 = min(result0, result8);
		result8 = tmp;

		tmp = max(result1, result9);
		result1 = min(result1, result9);
		result9 = tmp;

		tmp = max(result2, result10);
		result2 = min(result2, result10);
		result10 = tmp;

		tmp = max(result3, result11);
		result3 = min(result3, result11);
		result11 = tmp;

		tmp = max(result4, result12);
		result4 = min(result4, result12);
		result12 = tmp;

		tmp = max(result5, result13);
		result5 = min(result5, result13);
		result13 = tmp;

		tmp = max(result6, result14);
		result6 = min(result6, result14);
		result14 = tmp;

		tmp = max(result7, result15);
		result7 = min(result7, result15);
		result15 = tmp;

		tmp = max(result16, loaded_value24);
		result16 = min(result16, loaded_value24);
		int result24 = tmp;

		//Section 8
		tmp = max(result4, result8);
		result4 = min(result4, result8);
		result8 = tmp;

		tmp = max(result5, result9);
		result5 = min(result5, result9);
		result9 = tmp;

		tmp = max(result6, result10);
		result6 = min(result6, result10);
		result10 = tmp;

		tmp = max(result7, result11);
		result7 = min(result7, result11);
		result11 = tmp;

		tmp = max(result20, result24);
		result20 = min(result20, result24);
		result24 = tmp;

		//Section 9
		tmp = max(result2, result4);
		result2 = min(result2, result4);
		result4 = tmp;

		tmp = max(result3, result5);
		result3 = min(result3, result5);
		result5 = tmp;

		tmp = max(result6, result8);
		result6 = min(result6, result8);
		result8 = tmp;

		tmp = max(result7, result9);
		result7 = min(result7, result9);
		result9 = tmp;

		tmp = max(result10, result12);
		result10 = min(result10, result12);
		result12 = tmp;

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result18, result20);
		result18 = min(result18, result20);
		result20 = tmp;

		tmp = max(result19, result21);
		result19 = min(result19, result21);
		result21 = tmp;

		tmp = max(result22, result24);
		result22 = min(result22, result24);
		result24 = tmp;

		//Section 10
		tmp = max(result1, result2);
		result1 = min(result1, result2);
		result2 = tmp;

		tmp = max(result3, result4);
		result3 = min(result3, result4);
		result4 = tmp;

		tmp = max(result5, result6);
		result5 = min(result5, result6);
		result6 = tmp;

		tmp = max(result7, result8);
		result7 = min(result7, result8);
		result8 = tmp;

		tmp = max(result9, result10);
		result9 = min(result9, result10);
		result10 = tmp;

		tmp = max(result11, result12);
		result11 = min(result11, result12);
		result12 = tmp;

		tmp = max(result13, result14);
		result13 = min(result13, result14);
		result14 = tmp;

		tmp = max(result17, result18);
		result17 = min(result17, result18);
		result18 = tmp;

		tmp = max(result19, result20);
		result19 = min(result19, result20);
		result20 = tmp;

		tmp = max(result21, result22);
		result21 = min(result21, result22);
		result22 = tmp;

		tmp = max(result23, result24);
		result23 = min(result23, result24);
		result24 = tmp;

		//Section 11
		tmp = max(result0, result16);
		result0 = min(result0, result16);
		result16 = tmp;

		tmp = max(result1, result17);
		result1 = min(result1, result17);
		result17 = tmp;

		tmp = max(result2, result18);
		result2 = min(result2, result18);
		result18 = tmp;

		tmp = max(result3, result19);
		result3 = min(result3, result19);
		result19 = tmp;

		tmp = max(result4, result20);
		result4 = min(result4, result20);
		result20 = tmp;

		tmp = max(result5, result21);
		result5 = min(result5, result21);
		result21 = tmp;

		tmp = max(result6, result22);
		result6 = min(result6, result22);
		result22 = tmp;

		tmp = max(result7, result23);
		result7 = min(result7, result23);
		result23 = tmp;

		tmp = max(result8, result24);
		result8 = min(result8, result24);
		result24 = tmp;

		//Section 12
		tmp = max(result8, result16);
		result8 = min(result8, result16);
		result16 = tmp;

		tmp = max(result9, result17);
		result9 = min(result9, result17);
		result17 = tmp;

		tmp = max(result10, result18);
		result10 = min(result10, result18);
		result18 = tmp;

		tmp = max(result11, result19);
		result11 = min(result11, result19);
		result19 = tmp;

		tmp = max(result12, result20);
		result12 = min(result12, result20);
		result20 = tmp;

		tmp = max(result13, result21);
		result13 = min(result13, result21);
		result21 = tmp;

		//Section 13
		tmp = max(result6, result10);
		result6 = min(result6, result10);
		result10 = tmp;

		tmp = max(result7, result11);
		result7 = min(result7, result11);
		result11 = tmp;

		tmp = max(result12, result16);
		result12 = min(result12, result16);
		result16 = tmp;

		tmp = max(result13, result17);
		result13 = min(result13, result17);
		result17 = tmp;

		//Section 14
		tmp = max(result10, result12);
		result10 = min(result10, result12);
		result12 = tmp;

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		//Last Section
		result12 = max(result11, result12);

		
		gOutput[3 * (PY * imgWidth + PX) + rgb] = (unsigned char)result12;
	};
}; 

//Mivel ebben a kódban egy szál 2 pixelt számol ezért le kellene felezni az indított szálak számát
//conv_filer_ocl-ben át kell írni a kommentelt résznél
//Mivel ebben a kódban egy szál 2 pixelt számol ezért le kellene felezni az indított szálak számát
//conv_filer_ocl-ben át kell írni a kommentelt résznél
__kernel void median_filter_BMS2_global(__global unsigned char* gInput,
	__global unsigned char* gOutput,
	__constant float* filter_coeffs,
	int imgWidth,
	int imgWidthF)
{
	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = 2 * (get_group_id(1) * get_local_size(1) + get_local_id(1));

#pragma unroll
	for (int rgb = 0; rgb < 3; rgb++)
	{
		int loaded_value0 = gInput[3 * ((PY)*imgWidthF + (PX)) + rgb];
		int loaded_value1 = gInput[3 * ((PY)*imgWidthF + (1 + PX)) + rgb];
		int loaded_value2 = gInput[3 * ((PY)*imgWidthF + (2 + PX)) + rgb];
		int loaded_value3 = gInput[3 * ((PY)*imgWidthF + (3 + PX)) + rgb];
		int loaded_value4 = gInput[3 * ((PY)*imgWidthF + (4 + PX)) + rgb];

		int loaded_value5 = gInput[3 * ((PY + 1) * imgWidthF + (PX)) + rgb];
		int loaded_value6 = gInput[3 * ((PY + 1) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value7 = gInput[3 * ((PY + 1) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value8 = gInput[3 * ((PY + 1) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value9 = gInput[3 * ((PY + 1) * imgWidthF + (4 + PX)) + rgb];

		int loaded_value10 = gInput[3 * ((PY + 2) * imgWidthF + (PX)) + rgb];
		int loaded_value11 = gInput[3 * ((PY + 2) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value12 = gInput[3 * ((PY + 2) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value13 = gInput[3 * ((PY + 2) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value14 = gInput[3 * ((PY + 2) * imgWidthF + (4 + PX)) + rgb];

		int loaded_value15 = gInput[3 * ((PY + 3) * imgWidthF + (PX)) + rgb];
		int loaded_value16 = gInput[3 * ((PY + 3) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value17 = gInput[3 * ((PY + 3) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value18 = gInput[3 * ((PY + 3) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value19 = gInput[3 * ((PY + 3) * imgWidthF + (4 + PX)) + rgb];

		int loaded_value20 = gInput[3 * ((PY + 4) * imgWidthF + (PX)) + rgb];
		int loaded_value21 = gInput[3 * ((PY + 4) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value22 = gInput[3 * ((PY + 4) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value23 = gInput[3 * ((PY + 4) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value24 = gInput[3 * ((PY + 4) * imgWidthF + (4 + PX)) + rgb];

		int loaded_value25 = gInput[3 * ((PY + 5) * imgWidthF + (PX)) + rgb];
		int loaded_value26 = gInput[3 * ((PY + 5) * imgWidthF + (1 + PX)) + rgb];
		int loaded_value27 = gInput[3 * ((PY + 5) * imgWidthF + (2 + PX)) + rgb];
		int loaded_value28 = gInput[3 * ((PY + 5) * imgWidthF + (3 + PX)) + rgb];
		int loaded_value29 = gInput[3 * ((PY + 5) * imgWidthF + (4 + PX)) + rgb];


		//Közös halmaz szûrése
		//Section 1
		int result5 = min(loaded_value5, loaded_value6);
		int result6 = max(loaded_value5, loaded_value6);

		int result7 = min(loaded_value7, loaded_value8);
		int result8 = max(loaded_value7, loaded_value8);

		int result9 = min(loaded_value9, loaded_value10);
		int result10 = max(loaded_value9, loaded_value10);

		int result11 = min(loaded_value11, loaded_value12);
		int result12 = max(loaded_value11, loaded_value12);

		int result13 = min(loaded_value13, loaded_value14);
		int result14 = max(loaded_value13, loaded_value14);

		int result15 = min(loaded_value15, loaded_value16);
		int result16 = max(loaded_value15, loaded_value16);

		int result17 = min(loaded_value17, loaded_value18);
		int result18 = max(loaded_value17, loaded_value18);

		int result19 = min(loaded_value19, loaded_value20);
		int result20 = max(loaded_value19, loaded_value20);

		int result21 = min(loaded_value21, loaded_value22);
		int result22 = max(loaded_value21, loaded_value22);

		int result23 = min(loaded_value23, loaded_value24);
		int result24 = max(loaded_value23, loaded_value24);




		//Section 2
		int tmp = max(result5, result7);
		result5 = min(result5, result7);
		result7 = tmp;

		tmp = max(result6, result8);
		result6 = min(result6, result8);
		result8 = tmp;

		tmp = max(result9, result11);
		result9 = min(result9, result11);
		result11 = tmp;

		tmp = max(result10, result12);
		result10 = min(result10, result12);
		result12 = tmp;

		tmp = max(result13, result15);
		result13 = min(result13, result15);
		result15 = tmp;

		tmp = max(result14, result16);
		result14 = min(result14, result16);
		result16 = tmp;

		tmp = max(result17, result19);
		result17 = min(result17, result19);
		result19 = tmp;

		tmp = max(result18, result20);
		result18 = min(result18, result20);
		result20 = tmp;

		tmp = max(result21, result23);
		result21 = min(result21, result23);
		result23 = tmp;

		tmp = max(result22, result24);
		result22 = min(result22, result24);
		result24 = tmp;

		//Section 3

		tmp = max(result6, result7);
		result6 = min(result6, result7);
		result7 = tmp;

		tmp = max(result10, result11);
		result10 = min(result10, result11);
		result11 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result18, result19);
		result18 = min(result18, result19);
		result19 = tmp;

		tmp = max(result22, result23);
		result22 = min(result22, result23);
		result23 = tmp;

		//Section 4

		tmp = max(result5, result9);
		result5 = min(result5, result9);
		result9 = tmp;

		tmp = max(result6, result10);
		result6 = min(result6, result10);
		result10 = tmp;

		tmp = max(result7, result11);
		result7 = min(result7, result11);
		result11 = tmp;

		tmp = max(result8, result12);
		result8 = min(result8, result12);
		result12 = tmp;

		tmp = max(result13, result17);
		result13 = min(result13, result17);
		result17 = tmp;

		tmp = max(result14, result18);
		result14 = min(result14, result18);
		result18 = tmp;

		tmp = max(result15, result19);
		result15 = min(result15, result19);
		result19 = tmp;

		tmp = max(result16, result20);
		result16 = min(result16, result20);
		result20 = tmp;

		//Section 5

		tmp = max(result7, result9);
		result7 = min(result7, result9);
		result9 = tmp;

		tmp = max(result8, result10);
		result8 = min(result8, result10);
		result10 = tmp;

		tmp = max(result15, result17);
		result15 = min(result15, result17);
		result17 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		//Section 6

		tmp = max(result6, result7);
		result6 = min(result6, result7);
		result7 = tmp;

		tmp = max(result8, result9);
		result8 = min(result8, result9);
		result9 = tmp;

		tmp = max(result10, result11);
		result10 = min(result10, result11);
		result11 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result16, result17);
		result16 = min(result16, result17);
		result17 = tmp;

		tmp = max(result18, result19);
		result18 = min(result18, result19);
		result19 = tmp;

		tmp = max(result22, result23);
		result22 = min(result22, result23);
		result23 = tmp;

		//Section 7

		tmp = max(result5, result13);
		result5 = min(result5, result13);
		result13 = tmp;

		tmp = max(result6, result14);
		result6 = min(result6, result14);
		result14 = tmp;

		tmp = max(result7, result15);
		result7 = min(result7, result15);
		result15 = tmp;

		tmp = max(result8, result16);
		result8 = min(result8, result16);
		result16 = tmp;

		tmp = max(result9, result17);
		result9 = min(result9, result17);
		result17 = tmp;

		tmp = max(result10, result18);
		result10 = min(result10, result18);
		result18 = tmp;

		tmp = max(result11, result19);
		result11 = min(result11, result19);
		result19 = tmp;

		tmp = max(result12, result20);
		result12 = min(result12, result20);
		result20 = tmp;

		//Section 8

		tmp = max(result9, result13);
		result9 = min(result9, result13);
		result13 = tmp;

		tmp = max(result10, result14);
		result10 = min(result10, result14);
		result14 = tmp;

		tmp = max(result11, result15);
		result11 = min(result11, result15);
		result15 = tmp;

		tmp = max(result12, result16);
		result12 = min(result12, result16);
		result16 = tmp;

		//Section 9

		tmp = max(result7, result9);
		result7 = min(result7, result9);
		result9 = tmp;

		tmp = max(result8, result10);
		result8 = min(result8, result10);
		result10 = tmp;

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result12, result14);
		result12 = min(result12, result14);
		result14 = tmp;

		tmp = max(result15, result17);
		result15 = min(result15, result17);
		result17 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		//Section 10

		tmp = max(result6, result7);
		result6 = min(result6, result7);
		result7 = tmp;

		tmp = max(result8, result9);
		result8 = min(result8, result9);
		result9 = tmp;

		tmp = max(result10, result11);
		result10 = min(result10, result11);
		result11 = tmp;

		tmp = max(result12, result13);
		result12 = min(result12, result13);
		result13 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result16, result17);
		result16 = min(result16, result17);
		result17 = tmp;

		tmp = max(result18, result19);
		result18 = min(result18, result19);
		result19 = tmp;

		tmp = max(result22, result23);
		result22 = min(result22, result23);
		result23 = tmp;

		//Section 11

		tmp = max(result5, result21);
		result5 = min(result5, result21);
		result21 = tmp;

		tmp = max(result6, result22);
		result6 = min(result6, result22);
		result22 = tmp;

		tmp = max(result7, result23);
		result7 = min(result7, result23);
		result23 = tmp;

		tmp = max(result8, result24);
		result8 = min(result8, result24);
		result24 = tmp;

		//Section 12

		tmp = max(result13, result21);
		result13 = min(result13, result21);
		result21 = tmp;

		tmp = max(result14, result22);
		result14 = min(result14, result22);
		result22 = tmp;

		tmp = max(result15, result23);
		result15 = min(result15, result23);
		result23 = tmp;

		tmp = max(result16, result24);
		result16 = min(result16, result24);
		result24 = tmp;

		//Section 13

		tmp = max(result9, result13);
		result9 = min(result9, result13);
		result13 = tmp;

		tmp = max(result10, result14);
		result10 = min(result10, result14);
		result14 = tmp;

		tmp = max(result11, result15);
		result11 = min(result11, result15);
		result15 = tmp;

		tmp = max(result12, result16);
		result12 = min(result12, result16);
		result16 = tmp;

		tmp = max(result17, result21);
		result17 = min(result17, result21);
		result21 = tmp;

		tmp = max(result18, result22);
		result18 = min(result18, result22);
		result22 = tmp;

		//Section 14

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result12, result14);
		result12 = min(result12, result14);
		result14 = tmp;

		tmp = max(result15, result17);
		result15 = min(result15, result17);
		result17 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		//Section 15
		tmp = max(result12, result13);
		result12 = min(result12, result13);
		result13 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result16, result17);
		result16 = min(result16, result17);
		result17 = tmp;

		//Különbözõ
		// Kernel 1
		//Section 1  
		int kernel1_0 = min(loaded_value0, loaded_value1);
		int kernel1_1 = max(loaded_value0, loaded_value1);

		int kernel1_2 = min(loaded_value2, loaded_value3);
		int kernel1_3 = max(loaded_value2, loaded_value3);

		int kernel1_4 = min(loaded_value4, result12);
		int kernel1_5 = max(loaded_value4, result12);

		int kernel1_6 = result13;
		int kernel1_7 = result14;

		int kernel1_8 = result15;
		int kernel1_9 = result16;

		//Section 2

		tmp = max(kernel1_0, kernel1_2);
		kernel1_0 = min(kernel1_0, kernel1_2);
		kernel1_2 = tmp;

		tmp = max(kernel1_1, kernel1_3);
		kernel1_1 = min(kernel1_1, kernel1_3);
		kernel1_3 = tmp;

		tmp = max(kernel1_4, kernel1_6);
		kernel1_4 = min(kernel1_4, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_5, kernel1_7);
		kernel1_5 = min(kernel1_5, kernel1_7);
		kernel1_7 = tmp;

		tmp = max(kernel1_8, result17);
		kernel1_8 = min(kernel1_8, result17);
		int kernel1_10 = tmp;

		//Section 3

		tmp = max(kernel1_1, kernel1_2);
		kernel1_1 = min(kernel1_1, kernel1_2);
		kernel1_2 = tmp;

		tmp = max(kernel1_5, kernel1_6);
		kernel1_5 = min(kernel1_5, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_9, kernel1_10);
		kernel1_9 = min(kernel1_9, kernel1_10);
		kernel1_10 = tmp;

		//Section 4

		tmp = max(kernel1_0, kernel1_4);
		kernel1_0 = min(kernel1_0, kernel1_4);
		kernel1_4 = tmp;

		tmp = max(kernel1_1, kernel1_5);
		kernel1_1 = min(kernel1_1, kernel1_5);
		kernel1_5 = tmp;

		tmp = max(kernel1_2, kernel1_6);
		kernel1_2 = min(kernel1_2, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_3, kernel1_7);
		kernel1_3 = min(kernel1_3, kernel1_7);
		kernel1_7 = tmp;

		//Section 5

		tmp = max(kernel1_2, kernel1_4);
		kernel1_2 = min(kernel1_2, kernel1_4);
		kernel1_4 = tmp;

		tmp = max(kernel1_3, kernel1_5);
		kernel1_3 = min(kernel1_3, kernel1_5);
		kernel1_5 = tmp;

		//Section 6

		tmp = max(kernel1_1, kernel1_2);
		kernel1_1 = min(kernel1_1, kernel1_2);
		kernel1_2 = tmp;

		tmp = max(kernel1_3, kernel1_4);
		kernel1_3 = min(kernel1_3, kernel1_4);
		kernel1_4 = tmp;

		tmp = max(kernel1_5, kernel1_6);
		kernel1_5 = min(kernel1_5, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_9, kernel1_10);
		kernel1_9 = min(kernel1_9, kernel1_10);
		kernel1_10 = tmp;

		//Section 7

		tmp = max(kernel1_0, kernel1_8);
		kernel1_0 = min(kernel1_0, kernel1_8);
		kernel1_8 = tmp;

		tmp = max(kernel1_1, kernel1_9);
		kernel1_1 = min(kernel1_1, kernel1_9);
		kernel1_9 = tmp;

		tmp = max(kernel1_2, kernel1_10);
		kernel1_2 = min(kernel1_2, kernel1_10);
		kernel1_10 = tmp;

		//Section 8

		tmp = max(kernel1_4, kernel1_8);
		kernel1_4 = min(kernel1_4, kernel1_8);
		kernel1_8 = tmp;

		tmp = max(kernel1_5, kernel1_9);
		kernel1_5 = min(kernel1_5, kernel1_9);
		kernel1_9 = tmp;

		tmp = max(kernel1_6, kernel1_10);
		kernel1_6 = min(kernel1_6, kernel1_10);
		kernel1_10 = tmp;

		//Section 9

		tmp = max(kernel1_3, kernel1_5);
		kernel1_3 = min(kernel1_3, kernel1_5);
		kernel1_5 = tmp;

		tmp = max(kernel1_6, kernel1_8);
		kernel1_6 = min(kernel1_6, kernel1_8);
		kernel1_8 = tmp;

		//Section 10

		tmp = max(kernel1_5, kernel1_6);
		kernel1_5 = min(kernel1_5, kernel1_6);
		kernel1_6 = tmp;

		// Kernel 2
		//Section 1   
		int kernel2_0 = min(loaded_value25, loaded_value26);
		int kernel2_1 = max(loaded_value25, loaded_value26);

		int kernel2_2 = min(loaded_value27, loaded_value28);
		int kernel2_3 = max(loaded_value27, loaded_value28);

		int kernel2_4 = min(loaded_value29, result12);
		int kernel2_5 = max(loaded_value29, result12);

		int kernel2_6 = result13;
		int kernel2_7 = result14;

		int kernel2_8 = result15;
		int kernel2_9 = result16;

		//Section 2

		tmp = max(kernel2_0, kernel2_2);
		kernel2_0 = min(kernel2_0, kernel2_2);
		kernel2_2 = tmp;

		tmp = max(kernel2_1, kernel2_3);
		kernel2_1 = min(kernel2_1, kernel2_3);
		kernel2_3 = tmp;

		tmp = max(kernel2_4, kernel2_6);
		kernel2_4 = min(kernel2_4, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_5, kernel2_7);
		kernel2_5 = min(kernel2_5, kernel2_7);
		kernel2_7 = tmp;

		tmp = max(kernel2_8, result17);
		kernel2_8 = min(kernel2_8, result17);
		int kernel2_10 = tmp;

		//Section 3

		tmp = max(kernel2_1, kernel2_2);
		kernel2_1 = min(kernel2_1, kernel2_2);
		kernel2_2 = tmp;

		tmp = max(kernel2_5, kernel2_6);
		kernel2_5 = min(kernel2_5, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_9, kernel2_10);
		kernel2_9 = min(kernel2_9, kernel2_10);
		kernel2_10 = tmp;

		//Section 4

		tmp = max(kernel2_0, kernel2_4);
		kernel2_0 = min(kernel2_0, kernel2_4);
		kernel2_4 = tmp;

		tmp = max(kernel2_1, kernel2_5);
		kernel2_1 = min(kernel2_1, kernel2_5);
		kernel2_5 = tmp;

		tmp = max(kernel2_2, kernel2_6);
		kernel2_2 = min(kernel2_2, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_3, kernel2_7);
		kernel2_3 = min(kernel2_3, kernel2_7);
		kernel2_7 = tmp;

		//Section 5

		tmp = max(kernel2_2, kernel2_4);
		kernel2_2 = min(kernel2_2, kernel2_4);
		kernel2_4 = tmp;

		tmp = max(kernel2_3, kernel2_5);
		kernel2_3 = min(kernel2_3, kernel2_5);
		kernel2_5 = tmp;

		//Section 6

		tmp = max(kernel2_1, kernel2_2);
		kernel2_1 = min(kernel2_1, kernel2_2);
		kernel2_2 = tmp;

		tmp = max(kernel2_3, kernel2_4);
		kernel2_3 = min(kernel2_3, kernel2_4);
		kernel2_4 = tmp;

		tmp = max(kernel2_5, kernel2_6);
		kernel2_5 = min(kernel2_5, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_9, kernel2_10);
		kernel2_9 = min(kernel2_9, kernel2_10);
		kernel2_10 = tmp;

		//Section 7

		tmp = max(kernel2_0, kernel2_8);
		kernel2_0 = min(kernel2_0, kernel2_8);
		kernel2_8 = tmp;

		tmp = max(kernel2_1, kernel2_9);
		kernel2_1 = min(kernel2_1, kernel2_9);
		kernel2_9 = tmp;

		tmp = max(kernel2_2, kernel2_10);
		kernel2_2 = min(kernel2_2, kernel2_10);
		kernel2_10 = tmp;

		//Section 8

		tmp = max(kernel2_4, kernel2_8);
		kernel2_4 = min(kernel2_4, kernel2_8);
		kernel2_8 = tmp;

		tmp = max(kernel2_5, kernel2_9);
		kernel2_5 = min(kernel2_5, kernel2_9);
		kernel2_9 = tmp;

		tmp = max(kernel2_6, kernel2_10);
		kernel2_6 = min(kernel2_6, kernel2_10);
		kernel2_10 = tmp;

		//Section 9

		tmp = max(kernel2_3, kernel2_5);
		kernel2_3 = min(kernel2_3, kernel2_5);
		kernel2_5 = tmp;

		tmp = max(kernel2_6, kernel2_8);
		kernel2_6 = min(kernel2_6, kernel2_8);
		kernel2_8 = tmp;

		//Section 10

		tmp = max(kernel2_5, kernel2_6);
		kernel2_5 = min(kernel2_5, kernel2_6);
		kernel2_6 = tmp;

		gOutput[3 * (PY * imgWidth + PX) + rgb] = (unsigned char)kernel1_5;
		//
		gOutput[3 * ((PY + 1) * imgWidth + PX) + rgb] = (unsigned char)kernel2_5;

	};
}

//Shared 16*16 kernel méret. 1 kimenet mûködik ez a leggyorsabb jelenleg 2021.nov.8.
__kernel void kernel_median_shared(__global unsigned char* gInput,
	__global unsigned char* gOutput,
	__constant float* filter_coeffs,
	int imgWidth,
	int imgWidthF)
{
	__local int sh_mem[20][20][3];
	int th_load_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int start_id = 3 * (get_group_id(1) * get_local_size(1) * imgWidthF) + 3 * (get_group_id(0) * get_local_size(0));

	for (int loader = 0; loader < 5; loader++)
	{
		if (th_load_id < 240)
		{
			sh_mem[th_load_id / 60 + loader * 4][(th_load_id / 3) % 20][th_load_id % 3] = gInput[start_id + 4 * loader * 3 * imgWidthF + (th_load_id / 60) * 3 * imgWidthF + th_load_id % 60];
		};
	};
	barrier(CLK_LOCAL_MEM_FENCE);


	int X = get_local_id(0);
	int Y = get_local_id(1);

	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = get_group_id(1) * get_local_size(1) + get_local_id(1);


#pragma unroll
	for (int rgb = 0; rgb < 3; rgb++)
	{
		int loaded_value0 = sh_mem[Y][X][rgb];
		int loaded_value1 = sh_mem[Y][X + 1][rgb];
		int loaded_value2 = sh_mem[Y][X + 2][rgb];
		int loaded_value3 = sh_mem[Y][X + 3][rgb];
		int loaded_value4 = sh_mem[Y][X + 4][rgb];

		int loaded_value5 = sh_mem[Y + 1][X][rgb];
		int loaded_value6 = sh_mem[Y + 1][X + 1][rgb];
		int loaded_value7 = sh_mem[Y + 1][X + 2][rgb];
		int loaded_value8 = sh_mem[Y + 1][X + 3][rgb];
		int loaded_value9 = sh_mem[Y + 1][X + 4][rgb];

		int loaded_value10 = sh_mem[Y + 2][X][rgb];
		int loaded_value11 = sh_mem[Y + 2][X + 1][rgb];
		int loaded_value12 = sh_mem[Y + 2][X + 2][rgb];
		int loaded_value13 = sh_mem[Y + 2][X + 3][rgb];
		int loaded_value14 = sh_mem[Y + 2][X + 4][rgb];

		int loaded_value15 = sh_mem[Y + 3][X][rgb];
		int loaded_value16 = sh_mem[Y + 3][X + 1][rgb];
		int loaded_value17 = sh_mem[Y + 3][X + 2][rgb];
		int loaded_value18 = sh_mem[Y + 3][X + 3][rgb];
		int loaded_value19 = sh_mem[Y + 3][X + 4][rgb];

		int loaded_value20 = sh_mem[Y + 4][X][rgb];
		int loaded_value21 = sh_mem[Y + 4][X + 1][rgb];
		int loaded_value22 = sh_mem[Y + 4][X + 2][rgb];
		int loaded_value23 = sh_mem[Y + 4][X + 3][rgb];
		int loaded_value24 = sh_mem[Y + 4][X + 4][rgb];

		//Section 1
		int result0 = min(loaded_value0, loaded_value1);
		int result1 = max(loaded_value0, loaded_value1);

		int result2 = min(loaded_value2, loaded_value3);
		int result3 = max(loaded_value2, loaded_value3);

		int result4 = min(loaded_value4, loaded_value5);
		int result5 = max(loaded_value4, loaded_value5);

		int result6 = min(loaded_value6, loaded_value7);
		int result7 = max(loaded_value6, loaded_value7);

		int result8 = min(loaded_value8, loaded_value9);
		int result9 = max(loaded_value8, loaded_value9);

		int result10 = min(loaded_value10, loaded_value11);
		int result11 = max(loaded_value10, loaded_value11);

		int result12 = min(loaded_value12, loaded_value13);
		int result13 = max(loaded_value12, loaded_value13);

		int result14 = min(loaded_value14, loaded_value15);
		int result15 = max(loaded_value14, loaded_value15);

		int result16 = min(loaded_value16, loaded_value17);
		int result17 = max(loaded_value16, loaded_value17);

		int result18 = min(loaded_value18, loaded_value19);
		int result19 = max(loaded_value18, loaded_value19);

		int result20 = min(loaded_value20, loaded_value21);
		int result21 = max(loaded_value20, loaded_value21);

		int result22 = min(loaded_value22, loaded_value23);
		int result23 = max(loaded_value22, loaded_value23);


		//Section 2
		int tmp = max(result0, result2);
		result0 = min(result0, result2);
		result2 = tmp;

		tmp = max(result1, result3);
		result1 = min(result1, result3);
		result3 = tmp;

		tmp = max(result4, result6);
		result4 = min(result4, result6);
		result6 = tmp;

		tmp = max(result5, result7);
		result5 = min(result5, result7);
		result7 = tmp;

		tmp = max(result8, result10);
		result8 = min(result8, result10);
		result10 = tmp;

		tmp = max(result9, result11);
		result9 = min(result9, result11);
		result11 = tmp;

		tmp = max(result12, result14);
		result12 = min(result12, result14);
		result14 = tmp;

		tmp = max(result13, result15);
		result13 = min(result13, result15);
		result15 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		tmp = max(result17, result19);
		result17 = min(result17, result19);
		result19 = tmp;

		tmp = max(result20, result22);
		result20 = min(result20, result22);
		result22 = tmp;

		tmp = max(result21, result23);
		result21 = min(result21, result23);
		result23 = tmp;



		//Section 3
		tmp = max(result1, result2);
		result1 = min(result1, result2);
		result2 = tmp;

		tmp = max(result5, result6);
		result5 = min(result5, result6);
		result6 = tmp;

		tmp = max(result9, result10);
		result9 = min(result9, result10);
		result10 = tmp;

		tmp = max(result13, result14);
		result13 = min(result13, result14);
		result14 = tmp;

		tmp = max(result17, result18);
		result17 = min(result17, result18);
		result18 = tmp;

		tmp = max(result21, result22);
		result21 = min(result21, result22);
		result22 = tmp;

		//Section 4
		tmp = max(result0, result4);
		result0 = min(result0, result4);
		result4 = tmp;

		tmp = max(result1, result5);
		result1 = min(result1, result5);
		result5 = tmp;

		tmp = max(result2, result6);
		result2 = min(result2, result6);
		result6 = tmp;

		tmp = max(result3, result7);
		result3 = min(result3, result7);
		result7 = tmp;

		tmp = max(result8, result12);
		result8 = min(result8, result12);
		result12 = tmp;

		tmp = max(result9, result13);
		result9 = min(result9, result13);
		result13 = tmp;

		tmp = max(result10, result14);
		result10 = min(result10, result14);
		result14 = tmp;

		tmp = max(result11, result15);
		result11 = min(result11, result15);
		result15 = tmp;

		tmp = max(result16, result20);
		result16 = min(result16, result20);
		result20 = tmp;

		tmp = max(result17, result21);
		result17 = min(result17, result21);
		result21 = tmp;

		tmp = max(result18, result22);
		result18 = min(result18, result22);
		result22 = tmp;

		tmp = max(result19, result23);
		result19 = min(result19, result23);
		result23 = tmp;

		//Section 5
		tmp = max(result2, result4);
		result2 = min(result2, result4);
		result4 = tmp;

		tmp = max(result3, result5);
		result3 = min(result3, result5);
		result5 = tmp;

		tmp = max(result10, result12);
		result10 = min(result10, result12);
		result12 = tmp;

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result18, result20);
		result18 = min(result18, result20);
		result20 = tmp;

		tmp = max(result19, result21);
		result19 = min(result19, result21);
		result21 = tmp;


		//Section 6
		tmp = max(result1, result2);
		result1 = min(result1, result2);
		result2 = tmp;

		tmp = max(result3, result4);
		result3 = min(result3, result4);
		result4 = tmp;

		tmp = max(result5, result6);
		result5 = min(result5, result6);
		result6 = tmp;

		tmp = max(result9, result10);
		result9 = min(result9, result10);
		result10 = tmp;

		tmp = max(result11, result12);
		result11 = min(result11, result12);
		result12 = tmp;

		tmp = max(result13, result14);
		result13 = min(result13, result14);
		result14 = tmp;

		tmp = max(result17, result18);
		result17 = min(result17, result18);
		result18 = tmp;

		tmp = max(result19, result20);
		result19 = min(result19, result20);
		result20 = tmp;

		tmp = max(result21, result22);
		result21 = min(result21, result22);
		result22 = tmp;


		//Section 7
		tmp = max(result0, result8);
		result0 = min(result0, result8);
		result8 = tmp;

		tmp = max(result1, result9);
		result1 = min(result1, result9);
		result9 = tmp;

		tmp = max(result2, result10);
		result2 = min(result2, result10);
		result10 = tmp;

		tmp = max(result3, result11);
		result3 = min(result3, result11);
		result11 = tmp;

		tmp = max(result4, result12);
		result4 = min(result4, result12);
		result12 = tmp;

		tmp = max(result5, result13);
		result5 = min(result5, result13);
		result13 = tmp;

		tmp = max(result6, result14);
		result6 = min(result6, result14);
		result14 = tmp;

		tmp = max(result7, result15);
		result7 = min(result7, result15);
		result15 = tmp;

		tmp = max(result16, loaded_value24);
		result16 = min(result16, loaded_value24);
		int result24 = tmp;

		//Section 8
		tmp = max(result4, result8);
		result4 = min(result4, result8);
		result8 = tmp;

		tmp = max(result5, result9);
		result5 = min(result5, result9);
		result9 = tmp;

		tmp = max(result6, result10);
		result6 = min(result6, result10);
		result10 = tmp;

		tmp = max(result7, result11);
		result7 = min(result7, result11);
		result11 = tmp;

		tmp = max(result20, result24);
		result20 = min(result20, result24);
		result24 = tmp;

		//Section 9
		tmp = max(result2, result4);
		result2 = min(result2, result4);
		result4 = tmp;

		tmp = max(result3, result5);
		result3 = min(result3, result5);
		result5 = tmp;

		tmp = max(result6, result8);
		result6 = min(result6, result8);
		result8 = tmp;

		tmp = max(result7, result9);
		result7 = min(result7, result9);
		result9 = tmp;

		tmp = max(result10, result12);
		result10 = min(result10, result12);
		result12 = tmp;

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result18, result20);
		result18 = min(result18, result20);
		result20 = tmp;

		tmp = max(result19, result21);
		result19 = min(result19, result21);
		result21 = tmp;

		tmp = max(result22, result24);
		result22 = min(result22, result24);
		result24 = tmp;

		//Section 10
		tmp = max(result1, result2);
		result1 = min(result1, result2);
		result2 = tmp;

		tmp = max(result3, result4);
		result3 = min(result3, result4);
		result4 = tmp;

		tmp = max(result5, result6);
		result5 = min(result5, result6);
		result6 = tmp;

		tmp = max(result7, result8);
		result7 = min(result7, result8);
		result8 = tmp;

		tmp = max(result9, result10);
		result9 = min(result9, result10);
		result10 = tmp;

		tmp = max(result11, result12);
		result11 = min(result11, result12);
		result12 = tmp;

		tmp = max(result13, result14);
		result13 = min(result13, result14);
		result14 = tmp;

		tmp = max(result17, result18);
		result17 = min(result17, result18);
		result18 = tmp;

		tmp = max(result19, result20);
		result19 = min(result19, result20);
		result20 = tmp;

		tmp = max(result21, result22);
		result21 = min(result21, result22);
		result22 = tmp;

		tmp = max(result23, result24);
		result23 = min(result23, result24);
		result24 = tmp;

		//Section 11
		tmp = max(result0, result16);
		result0 = min(result0, result16);
		result16 = tmp;

		tmp = max(result1, result17);
		result1 = min(result1, result17);
		result17 = tmp;

		tmp = max(result2, result18);
		result2 = min(result2, result18);
		result18 = tmp;

		tmp = max(result3, result19);
		result3 = min(result3, result19);
		result19 = tmp;

		tmp = max(result4, result20);
		result4 = min(result4, result20);
		result20 = tmp;

		tmp = max(result5, result21);
		result5 = min(result5, result21);
		result21 = tmp;

		tmp = max(result6, result22);
		result6 = min(result6, result22);
		result22 = tmp;

		tmp = max(result7, result23);
		result7 = min(result7, result23);
		result23 = tmp;

		tmp = max(result8, result24);
		result8 = min(result8, result24);
		result24 = tmp;

		//Section 12
		tmp = max(result8, result16);
		result8 = min(result8, result16);
		result16 = tmp;

		tmp = max(result9, result17);
		result9 = min(result9, result17);
		result17 = tmp;

		tmp = max(result10, result18);
		result10 = min(result10, result18);
		result18 = tmp;

		tmp = max(result11, result19);
		result11 = min(result11, result19);
		result19 = tmp;

		tmp = max(result12, result20);
		result12 = min(result12, result20);
		result20 = tmp;

		tmp = max(result13, result21);
		result13 = min(result13, result21);
		result21 = tmp;

		//Section 13
		tmp = max(result6, result10);
		result6 = min(result6, result10);
		result10 = tmp;

		tmp = max(result7, result11);
		result7 = min(result7, result11);
		result11 = tmp;

		tmp = max(result12, result16);
		result12 = min(result12, result16);
		result16 = tmp;

		tmp = max(result13, result17);
		result13 = min(result13, result17);
		result17 = tmp;

		//Section 14
		tmp = max(result10, result12);
		result10 = min(result10, result12);
		result12 = tmp;

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		//Last Section
		result12 = max(result11, result12);


		gOutput[3 * (PY * imgWidth + PX) + rgb] = (unsigned char)result12;
	};
};


//shared 32*16 kernel méret opt. 2 kimenet egyszerre
__kernel void median_filter_BMS2_shared_16x16(__global unsigned char* gInput,
	__global unsigned char* gOutput,
	__constant float* filter_coeffs,
	int imgWidth,
	int imgWidthF)
{
	__local int sh_mem[36][20][3];
	int th_load_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int start_id = 3 * 2 * (get_group_id(1) * get_local_size(1) * imgWidthF) + 3 * (get_group_id(0) * get_local_size(0));

	for (int loader = 0; loader < 9; loader++)
	{
		if (th_load_id < 240)
		{
			sh_mem[th_load_id / 60 + loader * 4][(th_load_id / 3) % 20][th_load_id % 3] = gInput[start_id + 4 * loader * 3 * imgWidthF + (th_load_id / 60) * 3 * imgWidthF + th_load_id % 60];
		};
	};
	barrier(CLK_LOCAL_MEM_FENCE);


	int X = get_local_id(0);
	int Y = 2 * get_local_id(1);

	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = 2 * (get_group_id(1) * get_local_size(1) + get_local_id(1));


#pragma unroll
	for (int rgb = 0; rgb < 3; rgb++)
	{
		int loaded_value0 = sh_mem[Y][X][rgb];
		int loaded_value1 = sh_mem[Y][X + 1][rgb];
		int loaded_value2 = sh_mem[Y][X + 2][rgb];
		int loaded_value3 = sh_mem[Y][X + 3][rgb];
		int loaded_value4 = sh_mem[Y][X + 4][rgb];
		
		int loaded_value5 = sh_mem[Y + 1][X][rgb];
		int loaded_value6 = sh_mem[Y + 1][X + 1][rgb];
		int loaded_value7 = sh_mem[Y + 1][X + 2][rgb];
		int loaded_value8 = sh_mem[Y + 1][X + 3][rgb];
		int loaded_value9 = sh_mem[Y + 1][X + 4][rgb];
		
		int loaded_value10 = sh_mem[Y + 2][X][rgb];
		int loaded_value11 = sh_mem[Y + 2][X + 1][rgb];
		int loaded_value12 = sh_mem[Y + 2][X + 2][rgb];
		int loaded_value13 = sh_mem[Y + 2][X + 3][rgb];
		int loaded_value14 = sh_mem[Y + 2][X + 4][rgb];
		
		int loaded_value15 = sh_mem[Y + 3][X][rgb];
		int loaded_value16 = sh_mem[Y + 3][X + 1][rgb];
		int loaded_value17 = sh_mem[Y + 3][X + 2][rgb];
		int loaded_value18 = sh_mem[Y + 3][X + 3][rgb];
		int loaded_value19 = sh_mem[Y + 3][X + 4][rgb];
		
		int loaded_value20 = sh_mem[Y + 4][X][rgb];
		int loaded_value21 = sh_mem[Y + 4][X + 1][rgb];
		int loaded_value22 = sh_mem[Y + 4][X + 2][rgb];
		int loaded_value23 = sh_mem[Y + 4][X + 3][rgb];
		int loaded_value24 = sh_mem[Y + 4][X + 4][rgb];
		
		int loaded_value25 = sh_mem[Y + 5][X][rgb];
		int loaded_value26 = sh_mem[Y + 5][X + 1][rgb];
		int loaded_value27 = sh_mem[Y + 5][X + 2][rgb];
		int loaded_value28 = sh_mem[Y + 5][X + 3][rgb];
		int loaded_value29 = sh_mem[Y + 5][X + 4][rgb];


		//Közös halmaz szûrése
		//Section 1
		int result5 = min(loaded_value5, loaded_value6);
		int result6 = max(loaded_value5, loaded_value6);

		int result7 = min(loaded_value7, loaded_value8);
		int result8 = max(loaded_value7, loaded_value8);

		int result9 = min(loaded_value9, loaded_value10);
		int result10 = max(loaded_value9, loaded_value10);

		int result11 = min(loaded_value11, loaded_value12);
		int result12 = max(loaded_value11, loaded_value12);

		int result13 = min(loaded_value13, loaded_value14);
		int result14 = max(loaded_value13, loaded_value14);

		int result15 = min(loaded_value15, loaded_value16);
		int result16 = max(loaded_value15, loaded_value16);

		int result17 = min(loaded_value17, loaded_value18);
		int result18 = max(loaded_value17, loaded_value18);

		int result19 = min(loaded_value19, loaded_value20);
		int result20 = max(loaded_value19, loaded_value20);

		int result21 = min(loaded_value21, loaded_value22);
		int result22 = max(loaded_value21, loaded_value22);

		int result23 = min(loaded_value23, loaded_value24);
		int result24 = max(loaded_value23, loaded_value24);




		//Section 2
		int tmp = max(result5, result7);
		result5 = min(result5, result7);
		result7 = tmp;

		tmp = max(result6, result8);
		result6 = min(result6, result8);
		result8 = tmp;

		tmp = max(result9, result11);
		result9 = min(result9, result11);
		result11 = tmp;

		tmp = max(result10, result12);
		result10 = min(result10, result12);
		result12 = tmp;

		tmp = max(result13, result15);
		result13 = min(result13, result15);
		result15 = tmp;

		tmp = max(result14, result16);
		result14 = min(result14, result16);
		result16 = tmp;

		tmp = max(result17, result19);
		result17 = min(result17, result19);
		result19 = tmp;

		tmp = max(result18, result20);
		result18 = min(result18, result20);
		result20 = tmp;

		tmp = max(result21, result23);
		result21 = min(result21, result23);
		result23 = tmp;

		tmp = max(result22, result24);
		result22 = min(result22, result24);
		result24 = tmp;

		//Section 3

		tmp = max(result6, result7);
		result6 = min(result6, result7);
		result7 = tmp;

		tmp = max(result10, result11);
		result10 = min(result10, result11);
		result11 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result18, result19);
		result18 = min(result18, result19);
		result19 = tmp;

		tmp = max(result22, result23);
		result22 = min(result22, result23);
		result23 = tmp;

		//Section 4

		tmp = max(result5, result9);
		result5 = min(result5, result9);
		result9 = tmp;

		tmp = max(result6, result10);
		result6 = min(result6, result10);
		result10 = tmp;

		tmp = max(result7, result11);
		result7 = min(result7, result11);
		result11 = tmp;

		tmp = max(result8, result12);
		result8 = min(result8, result12);
		result12 = tmp;

		tmp = max(result13, result17);
		result13 = min(result13, result17);
		result17 = tmp;

		tmp = max(result14, result18);
		result14 = min(result14, result18);
		result18 = tmp;

		tmp = max(result15, result19);
		result15 = min(result15, result19);
		result19 = tmp;

		tmp = max(result16, result20);
		result16 = min(result16, result20);
		result20 = tmp;

		//Section 5

		tmp = max(result7, result9);
		result7 = min(result7, result9);
		result9 = tmp;

		tmp = max(result8, result10);
		result8 = min(result8, result10);
		result10 = tmp;

		tmp = max(result15, result17);
		result15 = min(result15, result17);
		result17 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		//Section 6

		tmp = max(result6, result7);
		result6 = min(result6, result7);
		result7 = tmp;

		tmp = max(result8, result9);
		result8 = min(result8, result9);
		result9 = tmp;

		tmp = max(result10, result11);
		result10 = min(result10, result11);
		result11 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result16, result17);
		result16 = min(result16, result17);
		result17 = tmp;

		tmp = max(result18, result19);
		result18 = min(result18, result19);
		result19 = tmp;

		tmp = max(result22, result23);
		result22 = min(result22, result23);
		result23 = tmp;

		//Section 7

		tmp = max(result5, result13);
		result5 = min(result5, result13);
		result13 = tmp;

		tmp = max(result6, result14);
		result6 = min(result6, result14);
		result14 = tmp;

		tmp = max(result7, result15);
		result7 = min(result7, result15);
		result15 = tmp;

		tmp = max(result8, result16);
		result8 = min(result8, result16);
		result16 = tmp;

		tmp = max(result9, result17);
		result9 = min(result9, result17);
		result17 = tmp;

		tmp = max(result10, result18);
		result10 = min(result10, result18);
		result18 = tmp;

		tmp = max(result11, result19);
		result11 = min(result11, result19);
		result19 = tmp;

		tmp = max(result12, result20);
		result12 = min(result12, result20);
		result20 = tmp;

		//Section 8

		tmp = max(result9, result13);
		result9 = min(result9, result13);
		result13 = tmp;

		tmp = max(result10, result14);
		result10 = min(result10, result14);
		result14 = tmp;

		tmp = max(result11, result15);
		result11 = min(result11, result15);
		result15 = tmp;

		tmp = max(result12, result16);
		result12 = min(result12, result16);
		result16 = tmp;

		//Section 9

		tmp = max(result7, result9);
		result7 = min(result7, result9);
		result9 = tmp;

		tmp = max(result8, result10);
		result8 = min(result8, result10);
		result10 = tmp;

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result12, result14);
		result12 = min(result12, result14);
		result14 = tmp;

		tmp = max(result15, result17);
		result15 = min(result15, result17);
		result17 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		//Section 10

		tmp = max(result6, result7);
		result6 = min(result6, result7);
		result7 = tmp;

		tmp = max(result8, result9);
		result8 = min(result8, result9);
		result9 = tmp;

		tmp = max(result10, result11);
		result10 = min(result10, result11);
		result11 = tmp;

		tmp = max(result12, result13);
		result12 = min(result12, result13);
		result13 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result16, result17);
		result16 = min(result16, result17);
		result17 = tmp;

		tmp = max(result18, result19);
		result18 = min(result18, result19);
		result19 = tmp;

		tmp = max(result22, result23);
		result22 = min(result22, result23);
		result23 = tmp;

		//Section 11

		tmp = max(result5, result21);
		result5 = min(result5, result21);
		result21 = tmp;

		tmp = max(result6, result22);
		result6 = min(result6, result22);
		result22 = tmp;

		tmp = max(result7, result23);
		result7 = min(result7, result23);
		result23 = tmp;

		tmp = max(result8, result24);
		result8 = min(result8, result24);
		result24 = tmp;

		//Section 12

		tmp = max(result13, result21);
		result13 = min(result13, result21);
		result21 = tmp;

		tmp = max(result14, result22);
		result14 = min(result14, result22);
		result22 = tmp;

		tmp = max(result15, result23);
		result15 = min(result15, result23);
		result23 = tmp;

		tmp = max(result16, result24);
		result16 = min(result16, result24);
		result24 = tmp;

		//Section 13

		tmp = max(result9, result13);
		result9 = min(result9, result13);
		result13 = tmp;

		tmp = max(result10, result14);
		result10 = min(result10, result14);
		result14 = tmp;

		tmp = max(result11, result15);
		result11 = min(result11, result15);
		result15 = tmp;

		tmp = max(result12, result16);
		result12 = min(result12, result16);
		result16 = tmp;

		tmp = max(result17, result21);
		result17 = min(result17, result21);
		result21 = tmp;

		tmp = max(result18, result22);
		result18 = min(result18, result22);
		result22 = tmp;

		//Section 14

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result12, result14);
		result12 = min(result12, result14);
		result14 = tmp;

		tmp = max(result15, result17);
		result15 = min(result15, result17);
		result17 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		//Section 15
		tmp = max(result12, result13);
		result12 = min(result12, result13);
		result13 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result16, result17);
		result16 = min(result16, result17);
		result17 = tmp;

		//Különbözõ
		// Kernel 1
		//Section 1  
		int kernel1_0 = min(loaded_value0, loaded_value1);
		int kernel1_1 = max(loaded_value0, loaded_value1);

		int kernel1_2 = min(loaded_value2, loaded_value3);
		int kernel1_3 = max(loaded_value2, loaded_value3);

		int kernel1_4 = min(loaded_value4, result12);
		int kernel1_5 = max(loaded_value4, result12);

		int kernel1_6 = result13;
		int kernel1_7 = result14;

		int kernel1_8 = result15;
		int kernel1_9 = result16;

		//Section 2

		tmp = max(kernel1_0, kernel1_2);
		kernel1_0 = min(kernel1_0, kernel1_2);
		kernel1_2 = tmp;

		tmp = max(kernel1_1, kernel1_3);
		kernel1_1 = min(kernel1_1, kernel1_3);
		kernel1_3 = tmp;

		tmp = max(kernel1_4, kernel1_6);
		kernel1_4 = min(kernel1_4, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_5, kernel1_7);
		kernel1_5 = min(kernel1_5, kernel1_7);
		kernel1_7 = tmp;

		tmp = max(kernel1_8, result17);
		kernel1_8 = min(kernel1_8, result17);
		int kernel1_10 = tmp;

		//Section 3

		tmp = max(kernel1_1, kernel1_2);
		kernel1_1 = min(kernel1_1, kernel1_2);
		kernel1_2 = tmp;

		tmp = max(kernel1_5, kernel1_6);
		kernel1_5 = min(kernel1_5, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_9, kernel1_10);
		kernel1_9 = min(kernel1_9, kernel1_10);
		kernel1_10 = tmp;

		//Section 4

		tmp = max(kernel1_0, kernel1_4);
		kernel1_0 = min(kernel1_0, kernel1_4);
		kernel1_4 = tmp;

		tmp = max(kernel1_1, kernel1_5);
		kernel1_1 = min(kernel1_1, kernel1_5);
		kernel1_5 = tmp;

		tmp = max(kernel1_2, kernel1_6);
		kernel1_2 = min(kernel1_2, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_3, kernel1_7);
		kernel1_3 = min(kernel1_3, kernel1_7);
		kernel1_7 = tmp;

		//Section 5

		tmp = max(kernel1_2, kernel1_4);
		kernel1_2 = min(kernel1_2, kernel1_4);
		kernel1_4 = tmp;

		tmp = max(kernel1_3, kernel1_5);
		kernel1_3 = min(kernel1_3, kernel1_5);
		kernel1_5 = tmp;

		//Section 6

		tmp = max(kernel1_1, kernel1_2);
		kernel1_1 = min(kernel1_1, kernel1_2);
		kernel1_2 = tmp;

		tmp = max(kernel1_3, kernel1_4);
		kernel1_3 = min(kernel1_3, kernel1_4);
		kernel1_4 = tmp;

		tmp = max(kernel1_5, kernel1_6);
		kernel1_5 = min(kernel1_5, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_9, kernel1_10);
		kernel1_9 = min(kernel1_9, kernel1_10);
		kernel1_10 = tmp;

		//Section 7

		tmp = max(kernel1_0, kernel1_8);
		kernel1_0 = min(kernel1_0, kernel1_8);
		kernel1_8 = tmp;

		tmp = max(kernel1_1, kernel1_9);
		kernel1_1 = min(kernel1_1, kernel1_9);
		kernel1_9 = tmp;

		tmp = max(kernel1_2, kernel1_10);
		kernel1_2 = min(kernel1_2, kernel1_10);
		kernel1_10 = tmp;

		//Section 8

		tmp = max(kernel1_4, kernel1_8);
		kernel1_4 = min(kernel1_4, kernel1_8);
		kernel1_8 = tmp;

		tmp = max(kernel1_5, kernel1_9);
		kernel1_5 = min(kernel1_5, kernel1_9);
		kernel1_9 = tmp;

		tmp = max(kernel1_6, kernel1_10);
		kernel1_6 = min(kernel1_6, kernel1_10);
		kernel1_10 = tmp;

		//Section 9

		tmp = max(kernel1_3, kernel1_5);
		kernel1_3 = min(kernel1_3, kernel1_5);
		kernel1_5 = tmp;

		tmp = max(kernel1_6, kernel1_8);
		kernel1_6 = min(kernel1_6, kernel1_8);
		kernel1_8 = tmp;

		//Section 10

		tmp = max(kernel1_5, kernel1_6);
		kernel1_5 = min(kernel1_5, kernel1_6);
		kernel1_6 = tmp;

		// Kernel 2
		//Section 1   
		int kernel2_0 = min(loaded_value25, loaded_value26);
		int kernel2_1 = max(loaded_value25, loaded_value26);

		int kernel2_2 = min(loaded_value27, loaded_value28);
		int kernel2_3 = max(loaded_value27, loaded_value28);

		int kernel2_4 = min(loaded_value29, result12);
		int kernel2_5 = max(loaded_value29, result12);

		int kernel2_6 = result13;
		int kernel2_7 = result14;

		int kernel2_8 = result15;
		int kernel2_9 = result16;

		//Section 2

		tmp = max(kernel2_0, kernel2_2);
		kernel2_0 = min(kernel2_0, kernel2_2);
		kernel2_2 = tmp;

		tmp = max(kernel2_1, kernel2_3);
		kernel2_1 = min(kernel2_1, kernel2_3);
		kernel2_3 = tmp;

		tmp = max(kernel2_4, kernel2_6);
		kernel2_4 = min(kernel2_4, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_5, kernel2_7);
		kernel2_5 = min(kernel2_5, kernel2_7);
		kernel2_7 = tmp;

		tmp = max(kernel2_8, result17);
		kernel2_8 = min(kernel2_8, result17);
		int kernel2_10 = tmp;

		//Section 3

		tmp = max(kernel2_1, kernel2_2);
		kernel2_1 = min(kernel2_1, kernel2_2);
		kernel2_2 = tmp;

		tmp = max(kernel2_5, kernel2_6);
		kernel2_5 = min(kernel2_5, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_9, kernel2_10);
		kernel2_9 = min(kernel2_9, kernel2_10);
		kernel2_10 = tmp;

		//Section 4

		tmp = max(kernel2_0, kernel2_4);
		kernel2_0 = min(kernel2_0, kernel2_4);
		kernel2_4 = tmp;

		tmp = max(kernel2_1, kernel2_5);
		kernel2_1 = min(kernel2_1, kernel2_5);
		kernel2_5 = tmp;

		tmp = max(kernel2_2, kernel2_6);
		kernel2_2 = min(kernel2_2, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_3, kernel2_7);
		kernel2_3 = min(kernel2_3, kernel2_7);
		kernel2_7 = tmp;

		//Section 5

		tmp = max(kernel2_2, kernel2_4);
		kernel2_2 = min(kernel2_2, kernel2_4);
		kernel2_4 = tmp;

		tmp = max(kernel2_3, kernel2_5);
		kernel2_3 = min(kernel2_3, kernel2_5);
		kernel2_5 = tmp;

		//Section 6

		tmp = max(kernel2_1, kernel2_2);
		kernel2_1 = min(kernel2_1, kernel2_2);
		kernel2_2 = tmp;

		tmp = max(kernel2_3, kernel2_4);
		kernel2_3 = min(kernel2_3, kernel2_4);
		kernel2_4 = tmp;

		tmp = max(kernel2_5, kernel2_6);
		kernel2_5 = min(kernel2_5, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_9, kernel2_10);
		kernel2_9 = min(kernel2_9, kernel2_10);
		kernel2_10 = tmp;

		//Section 7

		tmp = max(kernel2_0, kernel2_8);
		kernel2_0 = min(kernel2_0, kernel2_8);
		kernel2_8 = tmp;

		tmp = max(kernel2_1, kernel2_9);
		kernel2_1 = min(kernel2_1, kernel2_9);
		kernel2_9 = tmp;

		tmp = max(kernel2_2, kernel2_10);
		kernel2_2 = min(kernel2_2, kernel2_10);
		kernel2_10 = tmp;

		//Section 8

		tmp = max(kernel2_4, kernel2_8);
		kernel2_4 = min(kernel2_4, kernel2_8);
		kernel2_8 = tmp;

		tmp = max(kernel2_5, kernel2_9);
		kernel2_5 = min(kernel2_5, kernel2_9);
		kernel2_9 = tmp;

		tmp = max(kernel2_6, kernel2_10);
		kernel2_6 = min(kernel2_6, kernel2_10);
		kernel2_10 = tmp;

		//Section 9

		tmp = max(kernel2_3, kernel2_5);
		kernel2_3 = min(kernel2_3, kernel2_5);
		kernel2_5 = tmp;

		tmp = max(kernel2_6, kernel2_8);
		kernel2_6 = min(kernel2_6, kernel2_8);
		kernel2_8 = tmp;

		//Section 10

		tmp = max(kernel2_5, kernel2_6);
		kernel2_5 = min(kernel2_5, kernel2_6);
		kernel2_6 = tmp;

		gOutput[3 * (PY * imgWidth + PX) + rgb] = kernel1_5;
		//
		gOutput[3 * ((PY + 1) * imgWidth + PX) + rgb] = kernel2_5;

	};
}


__kernel void median_filter_BMS2_shared_32x16(__global unsigned char* gInput,
	__global unsigned char* gOutput,
	__constant float* filter_coeffs,
	int imgWidth,
	int imgWidthF)
{
	__local int sh_mem[36][36][3];
	int th_load_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int start_id = 3 * 2 * (get_group_id(1) * get_local_size(1) * imgWidthF) + 3 * (get_group_id(0) * get_local_size(0));

	for (int loader = 0; loader < 9; loader++)
	{
		if (th_load_id < 432)
		{
			sh_mem[th_load_id / 108 + loader * 4][(th_load_id / 3) % 36][th_load_id % 3] = gInput[start_id + 4 * loader * 3 * imgWidthF + (th_load_id / 108) * 3 * imgWidthF + th_load_id % 108];
		};
	};
	barrier(CLK_LOCAL_MEM_FENCE);


	int X = get_local_id(0);
	int Y = 2 * get_local_id(1);

	int PX = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int PY = 2 * (get_group_id(1) * get_local_size(1) + get_local_id(1));


#pragma unroll
	for (int rgb = 0; rgb < 3; rgb++)
	{
		int loaded_value0 = sh_mem[Y][X][rgb];
		int loaded_value1 = sh_mem[Y][X + 1][rgb];
		int loaded_value2 = sh_mem[Y][X + 2][rgb];
		int loaded_value3 = sh_mem[Y][X + 3][rgb];
		int loaded_value4 = sh_mem[Y][X + 4][rgb];

		int loaded_value5 = sh_mem[Y + 1][X][rgb];
		int loaded_value6 = sh_mem[Y + 1][X + 1][rgb];
		int loaded_value7 = sh_mem[Y + 1][X + 2][rgb];
		int loaded_value8 = sh_mem[Y + 1][X + 3][rgb];
		int loaded_value9 = sh_mem[Y + 1][X + 4][rgb];

		int loaded_value10 = sh_mem[Y + 2][X][rgb];
		int loaded_value11 = sh_mem[Y + 2][X + 1][rgb];
		int loaded_value12 = sh_mem[Y + 2][X + 2][rgb];
		int loaded_value13 = sh_mem[Y + 2][X + 3][rgb];
		int loaded_value14 = sh_mem[Y + 2][X + 4][rgb];

		int loaded_value15 = sh_mem[Y + 3][X][rgb];
		int loaded_value16 = sh_mem[Y + 3][X + 1][rgb];
		int loaded_value17 = sh_mem[Y + 3][X + 2][rgb];
		int loaded_value18 = sh_mem[Y + 3][X + 3][rgb];
		int loaded_value19 = sh_mem[Y + 3][X + 4][rgb];

		int loaded_value20 = sh_mem[Y + 4][X][rgb];
		int loaded_value21 = sh_mem[Y + 4][X + 1][rgb];
		int loaded_value22 = sh_mem[Y + 4][X + 2][rgb];
		int loaded_value23 = sh_mem[Y + 4][X + 3][rgb];
		int loaded_value24 = sh_mem[Y + 4][X + 4][rgb];

		int loaded_value25 = sh_mem[Y + 5][X][rgb];
		int loaded_value26 = sh_mem[Y + 5][X + 1][rgb];
		int loaded_value27 = sh_mem[Y + 5][X + 2][rgb];
		int loaded_value28 = sh_mem[Y + 5][X + 3][rgb];
		int loaded_value29 = sh_mem[Y + 5][X + 4][rgb];


		//Közös halmaz szûrése
		//Section 1
		int result5 = min(loaded_value5, loaded_value6);
		int result6 = max(loaded_value5, loaded_value6);

		int result7 = min(loaded_value7, loaded_value8);
		int result8 = max(loaded_value7, loaded_value8);

		int result9 = min(loaded_value9, loaded_value10);
		int result10 = max(loaded_value9, loaded_value10);

		int result11 = min(loaded_value11, loaded_value12);
		int result12 = max(loaded_value11, loaded_value12);

		int result13 = min(loaded_value13, loaded_value14);
		int result14 = max(loaded_value13, loaded_value14);

		int result15 = min(loaded_value15, loaded_value16);
		int result16 = max(loaded_value15, loaded_value16);

		int result17 = min(loaded_value17, loaded_value18);
		int result18 = max(loaded_value17, loaded_value18);

		int result19 = min(loaded_value19, loaded_value20);
		int result20 = max(loaded_value19, loaded_value20);

		int result21 = min(loaded_value21, loaded_value22);
		int result22 = max(loaded_value21, loaded_value22);

		int result23 = min(loaded_value23, loaded_value24);
		int result24 = max(loaded_value23, loaded_value24);




		//Section 2
		int tmp = max(result5, result7);
		result5 = min(result5, result7);
		result7 = tmp;

		tmp = max(result6, result8);
		result6 = min(result6, result8);
		result8 = tmp;

		tmp = max(result9, result11);
		result9 = min(result9, result11);
		result11 = tmp;

		tmp = max(result10, result12);
		result10 = min(result10, result12);
		result12 = tmp;

		tmp = max(result13, result15);
		result13 = min(result13, result15);
		result15 = tmp;

		tmp = max(result14, result16);
		result14 = min(result14, result16);
		result16 = tmp;

		tmp = max(result17, result19);
		result17 = min(result17, result19);
		result19 = tmp;

		tmp = max(result18, result20);
		result18 = min(result18, result20);
		result20 = tmp;

		tmp = max(result21, result23);
		result21 = min(result21, result23);
		result23 = tmp;

		tmp = max(result22, result24);
		result22 = min(result22, result24);
		result24 = tmp;

		//Section 3

		tmp = max(result6, result7);
		result6 = min(result6, result7);
		result7 = tmp;

		tmp = max(result10, result11);
		result10 = min(result10, result11);
		result11 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result18, result19);
		result18 = min(result18, result19);
		result19 = tmp;

		tmp = max(result22, result23);
		result22 = min(result22, result23);
		result23 = tmp;

		//Section 4

		tmp = max(result5, result9);
		result5 = min(result5, result9);
		result9 = tmp;

		tmp = max(result6, result10);
		result6 = min(result6, result10);
		result10 = tmp;

		tmp = max(result7, result11);
		result7 = min(result7, result11);
		result11 = tmp;

		tmp = max(result8, result12);
		result8 = min(result8, result12);
		result12 = tmp;

		tmp = max(result13, result17);
		result13 = min(result13, result17);
		result17 = tmp;

		tmp = max(result14, result18);
		result14 = min(result14, result18);
		result18 = tmp;

		tmp = max(result15, result19);
		result15 = min(result15, result19);
		result19 = tmp;

		tmp = max(result16, result20);
		result16 = min(result16, result20);
		result20 = tmp;

		//Section 5

		tmp = max(result7, result9);
		result7 = min(result7, result9);
		result9 = tmp;

		tmp = max(result8, result10);
		result8 = min(result8, result10);
		result10 = tmp;

		tmp = max(result15, result17);
		result15 = min(result15, result17);
		result17 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		//Section 6

		tmp = max(result6, result7);
		result6 = min(result6, result7);
		result7 = tmp;

		tmp = max(result8, result9);
		result8 = min(result8, result9);
		result9 = tmp;

		tmp = max(result10, result11);
		result10 = min(result10, result11);
		result11 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result16, result17);
		result16 = min(result16, result17);
		result17 = tmp;

		tmp = max(result18, result19);
		result18 = min(result18, result19);
		result19 = tmp;

		tmp = max(result22, result23);
		result22 = min(result22, result23);
		result23 = tmp;

		//Section 7

		tmp = max(result5, result13);
		result5 = min(result5, result13);
		result13 = tmp;

		tmp = max(result6, result14);
		result6 = min(result6, result14);
		result14 = tmp;

		tmp = max(result7, result15);
		result7 = min(result7, result15);
		result15 = tmp;

		tmp = max(result8, result16);
		result8 = min(result8, result16);
		result16 = tmp;

		tmp = max(result9, result17);
		result9 = min(result9, result17);
		result17 = tmp;

		tmp = max(result10, result18);
		result10 = min(result10, result18);
		result18 = tmp;

		tmp = max(result11, result19);
		result11 = min(result11, result19);
		result19 = tmp;

		tmp = max(result12, result20);
		result12 = min(result12, result20);
		result20 = tmp;

		//Section 8

		tmp = max(result9, result13);
		result9 = min(result9, result13);
		result13 = tmp;

		tmp = max(result10, result14);
		result10 = min(result10, result14);
		result14 = tmp;

		tmp = max(result11, result15);
		result11 = min(result11, result15);
		result15 = tmp;

		tmp = max(result12, result16);
		result12 = min(result12, result16);
		result16 = tmp;

		//Section 9

		tmp = max(result7, result9);
		result7 = min(result7, result9);
		result9 = tmp;

		tmp = max(result8, result10);
		result8 = min(result8, result10);
		result10 = tmp;

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result12, result14);
		result12 = min(result12, result14);
		result14 = tmp;

		tmp = max(result15, result17);
		result15 = min(result15, result17);
		result17 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		//Section 10

		tmp = max(result6, result7);
		result6 = min(result6, result7);
		result7 = tmp;

		tmp = max(result8, result9);
		result8 = min(result8, result9);
		result9 = tmp;

		tmp = max(result10, result11);
		result10 = min(result10, result11);
		result11 = tmp;

		tmp = max(result12, result13);
		result12 = min(result12, result13);
		result13 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result16, result17);
		result16 = min(result16, result17);
		result17 = tmp;

		tmp = max(result18, result19);
		result18 = min(result18, result19);
		result19 = tmp;

		tmp = max(result22, result23);
		result22 = min(result22, result23);
		result23 = tmp;

		//Section 11

		tmp = max(result5, result21);
		result5 = min(result5, result21);
		result21 = tmp;

		tmp = max(result6, result22);
		result6 = min(result6, result22);
		result22 = tmp;

		tmp = max(result7, result23);
		result7 = min(result7, result23);
		result23 = tmp;

		tmp = max(result8, result24);
		result8 = min(result8, result24);
		result24 = tmp;

		//Section 12

		tmp = max(result13, result21);
		result13 = min(result13, result21);
		result21 = tmp;

		tmp = max(result14, result22);
		result14 = min(result14, result22);
		result22 = tmp;

		tmp = max(result15, result23);
		result15 = min(result15, result23);
		result23 = tmp;

		tmp = max(result16, result24);
		result16 = min(result16, result24);
		result24 = tmp;

		//Section 13

		tmp = max(result9, result13);
		result9 = min(result9, result13);
		result13 = tmp;

		tmp = max(result10, result14);
		result10 = min(result10, result14);
		result14 = tmp;

		tmp = max(result11, result15);
		result11 = min(result11, result15);
		result15 = tmp;

		tmp = max(result12, result16);
		result12 = min(result12, result16);
		result16 = tmp;

		tmp = max(result17, result21);
		result17 = min(result17, result21);
		result21 = tmp;

		tmp = max(result18, result22);
		result18 = min(result18, result22);
		result22 = tmp;

		//Section 14

		tmp = max(result11, result13);
		result11 = min(result11, result13);
		result13 = tmp;

		tmp = max(result12, result14);
		result12 = min(result12, result14);
		result14 = tmp;

		tmp = max(result15, result17);
		result15 = min(result15, result17);
		result17 = tmp;

		tmp = max(result16, result18);
		result16 = min(result16, result18);
		result18 = tmp;

		//Section 15
		tmp = max(result12, result13);
		result12 = min(result12, result13);
		result13 = tmp;

		tmp = max(result14, result15);
		result14 = min(result14, result15);
		result15 = tmp;

		tmp = max(result16, result17);
		result16 = min(result16, result17);
		result17 = tmp;

		//Különbözõ
		// Kernel 1
		//Section 1  
		int kernel1_0 = min(loaded_value0, loaded_value1);
		int kernel1_1 = max(loaded_value0, loaded_value1);

		int kernel1_2 = min(loaded_value2, loaded_value3);
		int kernel1_3 = max(loaded_value2, loaded_value3);

		int kernel1_4 = min(loaded_value4, result12);
		int kernel1_5 = max(loaded_value4, result12);

		int kernel1_6 = result13;
		int kernel1_7 = result14;

		int kernel1_8 = result15;
		int kernel1_9 = result16;

		//Section 2

		tmp = max(kernel1_0, kernel1_2);
		kernel1_0 = min(kernel1_0, kernel1_2);
		kernel1_2 = tmp;

		tmp = max(kernel1_1, kernel1_3);
		kernel1_1 = min(kernel1_1, kernel1_3);
		kernel1_3 = tmp;

		tmp = max(kernel1_4, kernel1_6);
		kernel1_4 = min(kernel1_4, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_5, kernel1_7);
		kernel1_5 = min(kernel1_5, kernel1_7);
		kernel1_7 = tmp;

		tmp = max(kernel1_8, result17);
		kernel1_8 = min(kernel1_8, result17);
		int kernel1_10 = tmp;

		//Section 3

		tmp = max(kernel1_1, kernel1_2);
		kernel1_1 = min(kernel1_1, kernel1_2);
		kernel1_2 = tmp;

		tmp = max(kernel1_5, kernel1_6);
		kernel1_5 = min(kernel1_5, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_9, kernel1_10);
		kernel1_9 = min(kernel1_9, kernel1_10);
		kernel1_10 = tmp;

		//Section 4

		tmp = max(kernel1_0, kernel1_4);
		kernel1_0 = min(kernel1_0, kernel1_4);
		kernel1_4 = tmp;

		tmp = max(kernel1_1, kernel1_5);
		kernel1_1 = min(kernel1_1, kernel1_5);
		kernel1_5 = tmp;

		tmp = max(kernel1_2, kernel1_6);
		kernel1_2 = min(kernel1_2, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_3, kernel1_7);
		kernel1_3 = min(kernel1_3, kernel1_7);
		kernel1_7 = tmp;

		//Section 5

		tmp = max(kernel1_2, kernel1_4);
		kernel1_2 = min(kernel1_2, kernel1_4);
		kernel1_4 = tmp;

		tmp = max(kernel1_3, kernel1_5);
		kernel1_3 = min(kernel1_3, kernel1_5);
		kernel1_5 = tmp;

		//Section 6

		tmp = max(kernel1_1, kernel1_2);
		kernel1_1 = min(kernel1_1, kernel1_2);
		kernel1_2 = tmp;

		tmp = max(kernel1_3, kernel1_4);
		kernel1_3 = min(kernel1_3, kernel1_4);
		kernel1_4 = tmp;

		tmp = max(kernel1_5, kernel1_6);
		kernel1_5 = min(kernel1_5, kernel1_6);
		kernel1_6 = tmp;

		tmp = max(kernel1_9, kernel1_10);
		kernel1_9 = min(kernel1_9, kernel1_10);
		kernel1_10 = tmp;

		//Section 7

		tmp = max(kernel1_0, kernel1_8);
		kernel1_0 = min(kernel1_0, kernel1_8);
		kernel1_8 = tmp;

		tmp = max(kernel1_1, kernel1_9);
		kernel1_1 = min(kernel1_1, kernel1_9);
		kernel1_9 = tmp;

		tmp = max(kernel1_2, kernel1_10);
		kernel1_2 = min(kernel1_2, kernel1_10);
		kernel1_10 = tmp;

		//Section 8

		tmp = max(kernel1_4, kernel1_8);
		kernel1_4 = min(kernel1_4, kernel1_8);
		kernel1_8 = tmp;

		tmp = max(kernel1_5, kernel1_9);
		kernel1_5 = min(kernel1_5, kernel1_9);
		kernel1_9 = tmp;

		tmp = max(kernel1_6, kernel1_10);
		kernel1_6 = min(kernel1_6, kernel1_10);
		kernel1_10 = tmp;

		//Section 9

		tmp = max(kernel1_3, kernel1_5);
		kernel1_3 = min(kernel1_3, kernel1_5);
		kernel1_5 = tmp;

		tmp = max(kernel1_6, kernel1_8);
		kernel1_6 = min(kernel1_6, kernel1_8);
		kernel1_8 = tmp;

		//Section 10

		tmp = max(kernel1_5, kernel1_6);
		kernel1_5 = min(kernel1_5, kernel1_6);
		kernel1_6 = tmp;

		// Kernel 2
		//Section 1   
		int kernel2_0 = min(loaded_value25, loaded_value26);
		int kernel2_1 = max(loaded_value25, loaded_value26);

		int kernel2_2 = min(loaded_value27, loaded_value28);
		int kernel2_3 = max(loaded_value27, loaded_value28);

		int kernel2_4 = min(loaded_value29, result12);
		int kernel2_5 = max(loaded_value29, result12);

		int kernel2_6 = result13;
		int kernel2_7 = result14;

		int kernel2_8 = result15;
		int kernel2_9 = result16;

		//Section 2

		tmp = max(kernel2_0, kernel2_2);
		kernel2_0 = min(kernel2_0, kernel2_2);
		kernel2_2 = tmp;

		tmp = max(kernel2_1, kernel2_3);
		kernel2_1 = min(kernel2_1, kernel2_3);
		kernel2_3 = tmp;

		tmp = max(kernel2_4, kernel2_6);
		kernel2_4 = min(kernel2_4, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_5, kernel2_7);
		kernel2_5 = min(kernel2_5, kernel2_7);
		kernel2_7 = tmp;

		tmp = max(kernel2_8, result17);
		kernel2_8 = min(kernel2_8, result17);
		int kernel2_10 = tmp;

		//Section 3

		tmp = max(kernel2_1, kernel2_2);
		kernel2_1 = min(kernel2_1, kernel2_2);
		kernel2_2 = tmp;

		tmp = max(kernel2_5, kernel2_6);
		kernel2_5 = min(kernel2_5, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_9, kernel2_10);
		kernel2_9 = min(kernel2_9, kernel2_10);
		kernel2_10 = tmp;

		//Section 4

		tmp = max(kernel2_0, kernel2_4);
		kernel2_0 = min(kernel2_0, kernel2_4);
		kernel2_4 = tmp;

		tmp = max(kernel2_1, kernel2_5);
		kernel2_1 = min(kernel2_1, kernel2_5);
		kernel2_5 = tmp;

		tmp = max(kernel2_2, kernel2_6);
		kernel2_2 = min(kernel2_2, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_3, kernel2_7);
		kernel2_3 = min(kernel2_3, kernel2_7);
		kernel2_7 = tmp;

		//Section 5

		tmp = max(kernel2_2, kernel2_4);
		kernel2_2 = min(kernel2_2, kernel2_4);
		kernel2_4 = tmp;

		tmp = max(kernel2_3, kernel2_5);
		kernel2_3 = min(kernel2_3, kernel2_5);
		kernel2_5 = tmp;

		//Section 6

		tmp = max(kernel2_1, kernel2_2);
		kernel2_1 = min(kernel2_1, kernel2_2);
		kernel2_2 = tmp;

		tmp = max(kernel2_3, kernel2_4);
		kernel2_3 = min(kernel2_3, kernel2_4);
		kernel2_4 = tmp;

		tmp = max(kernel2_5, kernel2_6);
		kernel2_5 = min(kernel2_5, kernel2_6);
		kernel2_6 = tmp;

		tmp = max(kernel2_9, kernel2_10);
		kernel2_9 = min(kernel2_9, kernel2_10);
		kernel2_10 = tmp;

		//Section 7

		tmp = max(kernel2_0, kernel2_8);
		kernel2_0 = min(kernel2_0, kernel2_8);
		kernel2_8 = tmp;

		tmp = max(kernel2_1, kernel2_9);
		kernel2_1 = min(kernel2_1, kernel2_9);
		kernel2_9 = tmp;

		tmp = max(kernel2_2, kernel2_10);
		kernel2_2 = min(kernel2_2, kernel2_10);
		kernel2_10 = tmp;

		//Section 8

		tmp = max(kernel2_4, kernel2_8);
		kernel2_4 = min(kernel2_4, kernel2_8);
		kernel2_8 = tmp;

		tmp = max(kernel2_5, kernel2_9);
		kernel2_5 = min(kernel2_5, kernel2_9);
		kernel2_9 = tmp;

		tmp = max(kernel2_6, kernel2_10);
		kernel2_6 = min(kernel2_6, kernel2_10);
		kernel2_10 = tmp;

		//Section 9

		tmp = max(kernel2_3, kernel2_5);
		kernel2_3 = min(kernel2_3, kernel2_5);
		kernel2_5 = tmp;

		tmp = max(kernel2_6, kernel2_8);
		kernel2_6 = min(kernel2_6, kernel2_8);
		kernel2_8 = tmp;

		//Section 10

		tmp = max(kernel2_5, kernel2_6);
		kernel2_5 = min(kernel2_5, kernel2_6);
		kernel2_6 = tmp;

		gOutput[3 * (PY * imgWidth + PX) + rgb] = kernel1_5;
		//
		gOutput[3 * ((PY + 1) * imgWidth + PX) + rgb] = kernel2_5;

	};
}