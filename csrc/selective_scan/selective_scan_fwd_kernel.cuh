/******************************************************************************
 * Copyright (c) 2023, Tri Dao.+
 ******************************************************************************/

#pragma once

/* Define the BrainFloat format + some conversions.
    see details here https://github.com/pytorch/pytorch/blob/main/c10/util/BFloat16.h
    Note: has some ROCm-specific code.
*/
#include <c10/util/BFloat16.h> 

/*
Define the half-precision format. Notably better documented than BF16. Interestingly, computations are
performed by converting to 32 bits. For faster computations, CUDA intrinsics need to be invoked directly by the user.                             
*/
#include <c10/util/Half.h> 

#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

/*
CUB library imports. In general, CUB comes with a number of pre-made algorithms (including scan),
intended to make CUDA code easier to write and more modular/composable. See https://nvidia.github.io/cccl/cub/  
*/
                                    // From CUB docs:
#include <cub/block/block_load.cuh> // ... collective data movement methods for loading a linear segment of items from memory into a blocked arrangement across a CUDA thread block.
#include <cub/block/block_store.cuh> // ... collective data movement methods for writing a blocked arrangement of items partitioned across a CUDA thread block to a linear segment of memory.
#include <cub/block/block_scan.cuh> // ... collective methods for computing a parallel prefix sum/scan of items partitioned across a CUDA thread block.


#include "selective_scan.h" // importantly, defines SSM parameter structures both for fwd and bwd.
#include "selective_scan_common.h" // redefines a + operator for 2,3, & 4-element float vectors (float2, float3, ...) + a BytesToType helper struct
                                   // Also defines some type converters.
                                   // Perhaps most importantly, defines a SSMScanOp template which gives the key scan 
                                   // element (a pair for real-valued, a quadruple for complex ones) needed for the 
                                   // parallel scan algorithm to work. # TODO: link to the paper with theory on this.

#include "static_switch.h" // conditional code execution lambda macro.

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
         bool kIsVariableB_, bool kIsVariableC_,
         bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_fwd_kernel_traits { 
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy. // original comment
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3; // TODO: Potentially needs to be adjusted for AMD
    static constexpr int kNItems = kNItems_; // TODO
    static constexpr int kNRows = kNRows_; // How many rows to process per block. grid size is (batch_size, dim/kNRows). Currently set to 1 and only tested with 1, so not really important. Potential for optimization.
    static constexpr int kNBytes = sizeof(input_t); // Selecting the precision regime
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts; // TODO
    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_; // Whether we have selection applied to B. Only tested with True
    static constexpr bool kIsVariableC = kIsVariableC_; // whether we have selection applied to C. Only tested with True
    static constexpr bool kHasZ = kHasZ_; // TODO. Interestingly, only tested with True in test_selective_scan.py

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;


    // Below-instantiating cub templates for different ways of loading/storing data.

    // cub::BLOCK_LOAD_WARP_TRANSPOSE defines how the data is loaded.
    // From docs: A striped arrangement of data is read efficiently from memory and then locally transposed into a blocked arrangement.
    // More on memory arrangements: https://nvidia.github.io/cccl/cub/index.html#flexible-data-arrangement
    // In general, the docs for these constants can be found in Namespaces->cub->Enums
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    
    // From docs: A blocked arrangement of data is read directly from memory. The utilization of memory transactions (coalescing) decreases as the access stride between threads increases (i.e., the number items per thread).
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;

    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, !kIsComplex ? kNItems : kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, !kIsComplex ? kNLoads : kNLoads * 2,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>; // Commented out in the original code.
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>; // Commented out in the original code.

    // Specifying block scan parameters.
    // See https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockScan.html
    // cub::BLOCK_SCAN_WARP_SCANS indicates a specific version of cub's implementation of the prefix scan/sum.
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;

    // Below - determining the buffer size needed to transfer/store relevant data. TODO: double-check the usage
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};


// Regarding __launch_bounds__: the signature is __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
// Used to more finely tune/optimize GPU resource allocation
// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds for more.
template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks) 
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ; // TODO
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems; // How many elements of the input sequence to process per thread. Notice that CUB aggregates across threads in a block, so we only need to process a part of the sequence per thread. See 'A Simple Example' here: https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockScan.html
    constexpr int kNRows = Ktraits::kNRows; // Is actualy set to 1 in the kernel launch and is not tested with other values.
    constexpr bool kDirectIO = Ktraits::kDirectIO; // TODO
    using input_t = typename Ktraits::input_t; // input type
    using weight_t = typename Ktraits::weight_t; // weight type
    using scan_t = typename Ktraits::scan_t; // The type of the scan variable, will be float2 or float4 vector depending on 
                                             // whether we're working with a complex case or not. Needs 2 elements even for the real case to 
                                             // parameterize the process to fit the scan algorithm.
=
    // Shared memory. // original comment
    // See more here https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
    // "Dynamic Shared Memory" section in particular.
    // Note that when the memory size is not specified during shared memory array creation,
    // It must be specified during the kernel launch.
    // kSmemSize parameter holds the size sufficient to guarantee that we can store all we need.

    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type // original comment
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t); // original comment
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t)); // original comment
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan); // original comment


    
    // What we are doing below is creating pointers of different types to the same memory that we plan to reuse #TODO: double check

    // Also notice that Ktraits will be an instance of a structure defined above (Selective_Scan_fwd_kernel_traits)
    // So internally it will have a BlockLoadT member, defined so as to fit the specific setup in that particular traits object.

    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    // weight_t *smem_a = reinterpret_cast<weight_t *>(smem_ + smem_loadstorescan_size); // original comment
    // weight_t *smem_bc = reinterpret_cast<weight_t *>(smem_a + MAX_DSTATE); // original comment
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);

    // Below: get S6 value pointers relevant for this particular kernel/thread. 

    // note that u (input) tensor dimension is B D L.
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    // Note that since we are using cub's block scan, we will be running the scan & aggregating across multiple threads in a block.

    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            delta_bias[r] = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }

    
    // for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {                 // original comment
    //     smem_a[state_idx] = A[state_idx * params.A_dstate_stride];                                          // original comment
    //     smem_bc[state_idx] = B[state_idx * params.B_dstate_stride] * C[state_idx * params.C_dstate_stride]; // original comment
    // }                                                                                                       // original comment

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); } // TODO: generally. conditional synchthreads is considered really dangerous, I'd need to check how this is supposed to work. NRows is usually 1, so this is untested and irrelevant.
            }

            // see selective_scan_common.h for load_input definition. Uses cub's data managing primitives.
            // written strangely substantially different depending on whether the length is even
            // potential oversight as the code is mostly tested with even lengths (see tests/ops/test_selective_scan.py line 20)
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); } // this conditional syncing is okay, since it will always be executed for all threads or for none.
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u += kChunkSize; // advance pointers
        delta += kChunkSize; // advance pointers 

        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = float(u_vals[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                if (params.delta_softplus) {
                            // above 20.0 (arbitrary) softplus is close enough to linear, so the authors bypass it. 
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        
        
        __syncthreads();
        // The loop below does a lot of work, going over the hidden state dimensions and handling them one by one.
        // The cub scan is applied to one hidden dimension at at a time.
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {

            // Below - load values of A looping over hidden state dimensions.
            weight_t A_val[kNRows]; // kNRows is tested with a value of 1.
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf. // original comment
                constexpr float kLog2e = M_LOG2E;
                if constexpr (!kIsComplex) {
                    A_val[r] *= kLog2e;
                } else {
                    A_val[r].real_ *= kLog2e;
                }
            }
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies // original comment
            // across seqlen, this holds C. If only C varies across seqlen, this holds B. // original comment
            // If both B and C vary, this is unused. // original comment

            //Next few code blocks - load B and C values
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB) {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }
            if constexpr (kIsVariableC) {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC) {
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }
            // Done loading B and C 

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem // original comment
                scan_t thread_data[kNItems]; // note that thread data will hold the key pairs for scanning over.
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    if constexpr (!kIsComplex) {
                        
                        // note that we are discretizing A when constructing the first element of the pair.
                        // TODO: where/when does B discretization happen? Could be implicit in some other part of the code.
                        // Conceptually, the code follows equation 33 in "SIMPLIFIED STATE SPACE LAYERS FOR SEQUENCE MODELING"
                        thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]),
                                                     !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]);
                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct // original comment
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                                thread_data[i] = make_float2(1.f, 0.f);
                            }
                        }
                    } else {
                        // Pytorch's implementation of complex exp (which calls thrust) is very slow // original comment
                        complex_t delta_a_exp = cexp2f(delta_vals[r][i] * A_val[r]);
                        weight_t B_delta_u_val = !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i];
                        thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_delta_u_val.real_, B_delta_u_val.imag_);
                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct // original comment
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                                thread_data[i] = make_float4(1.f, 0.f, 0.f, 0.f);
                            }
                        }
                    }
                }
                // Initialize running total // original comment
                scan_t running_prefix;
                if constexpr (!kIsComplex) {
                    // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read // original comment
                    running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float2(1.f, 0.f);
                    // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f); // original comment
                } else {
                    running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float4(1.f, 0.f, 0.f, 0.f);
                    // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float4(1.f, 0.f, 0.f, 0.f); // original comment
                }
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                // There's a syncthreads in the scan op, so we don't need to sync here. // original comment 
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.// original comment

                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = prefix_op.running_prefix;
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const weight_t C_val = !kIsVariableC
                        ? BC_val[r]
                        : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
                    if constexpr (!kIsComplex) {
                        out_vals[r][i] += thread_data[i].y * C_val;
                    } else {
                        out_vals[r][i] += (complex_t(thread_data[i].z, thread_data[i].w) * C_val).real_ * 2;
                    }
                }
            }
        }

        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize * (!kIsComplex ? 1 : 2);
        Cvar += kChunkSize * (!kIsComplex ? 1 : 2);
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block // original comment
    // processing 1 row. // original comment

    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] { // we'd like to define kIsEvenLen as constexpr, hence the BOOL_SWITCH macro wrapper.
        BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] { // same for kIsVariableB and the next two rows
            BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] { // 
                BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] { //
                    using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ, input_t, weight_t>;
                    // constexpr int kSmemSize = Ktraits::kSmemSize; // original comment
                    constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                    // printf("smem_size = %d\n", kSmemSize); // original comment
                    dim3 grid(params.batch, params.dim / kNRows);

                    auto kernel = &selective_scan_fwd_kernel<Ktraits>; // Taking a function pointer here in order to 
                                                                       // set/tweak the shared memory size before the actual kernel launch.
                    if (kSmemSize >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                    }
                    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}


// Below making a template over input types, so that later we can instantiate them in separate files.
// This helps with compilation speed (we can compile for different datatypes in parallel).
template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 256) {
        selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 512) {
        selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    } else {
        selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
}
