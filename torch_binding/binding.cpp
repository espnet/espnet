#include <iostream>
#include <vector>

#include "utils.h"

#include <lua.h>
#include <luaT.h>

#include <numeric>

#include "ctc.h"

#ifdef TORCH_NOGPU
    #include "TH.h"
#else
    #include "THC.h"
    #include "THCTensor.h"
    #include "detail/reduce.h"
#endif

int processTargets(lua_State* L, int** sizes, int** labels, int** label_sizes) {
    // sizes table is 4 item on stack
    // labels is 3 item on stack

    if (!lua_istable(L, 4)) {
        lua_pushfstring(L, "invalid argument 4 for sequence lengths (expected table, got %s)",
                        luaL_typename(L, -1));
        lua_error(L);
    }

    int minibatch_size = lua_objlen(L, 4);

    *sizes = new int[minibatch_size];

    for(int i = 0; i < minibatch_size; i++) {
        lua_pushinteger(L, i+1);
        lua_gettable(L, 4);
        if(lua_isnumber(L, -1)) {
            (*sizes)[i] = (int) lua_tonumber(L, -1);
        } else {
            lua_pushfstring(L, "invalid entry #%d in array sizes (expected number, got %s)",
                            i, luaL_typename(L, -1));
            lua_error(L);
        }
        lua_pop(L, 1);
    }

    if (!lua_istable(L, 3)) {
        lua_pushfstring(L, "invalid argument 3 for sequence labels (expected table, got %s)",
                        luaL_typename(L, -1));
        lua_error(L);
    }

    int number_of_target_seq = lua_objlen(L, 3);

    if (number_of_target_seq != minibatch_size) {
        lua_pushfstring(L, "The minibatch size %d and the number of target sequences %d must be the same",
                        minibatch_size, number_of_target_seq);
        lua_error(L);
    }

    std::vector<int> labels_vec;
    *label_sizes = new int[minibatch_size];

    for(int i = 0; i < minibatch_size; i++) {
        lua_pushinteger(L, i+1);
        lua_gettable(L, 3);

        if(lua_istable(L, -1)) {

            int current_label_length = (int) lua_objlen(L, -1);
            (*label_sizes)[i] = current_label_length;

            for (int ix = 0; ix < current_label_length; ix++) {
                lua_pushinteger(L, ix + 1);
                lua_gettable(L, -2);
                if(lua_isnumber(L, -1)) {
                    labels_vec.push_back((int) lua_tonumber(L, -1));
                } else {
                    lua_pushfstring(L, "invalid entry #%d in array labels (expected number, got %s)",
                                    ix + 1, luaL_typename(L, -1));
                    lua_error(L);
                }

                lua_pop(L, 1);
            }

        } else {
            lua_pushfstring(L, "invalid entry #%d in table labels (expected table, got %s)",
                            i + 1, luaL_typename(L, -1));
            lua_error(L);
        }

        lua_pop(L, 1);

    }

    *labels = new int[labels_vec.size()];

    std::copy(labels_vec.begin(), labels_vec.end(), *labels);

    return minibatch_size;
}

extern "C" int gpu_ctc(lua_State* L) {
#ifdef TORCH_NOGPU
    std::cout << "Compiled without CUDA support." << std::endl;
    lua_newtable(L);


    lua_pushnumber(L, -999999.0);
    lua_rawseti(L, -2, 1);

#else
    THCudaTensor *probs =
        static_cast<THCudaTensor *>(luaT_checkudata(L, 1, "torch.CudaTensor"));
    THCudaTensor *grads =
        static_cast<THCudaTensor *>(luaT_checkudata(L, 2, "torch.CudaTensor"));

    int* sizes;
    int* labels_ptr;
    int* label_sizes_ptr;

    int minibatch_size = processTargets(L, &sizes, &labels_ptr, &label_sizes_ptr);

    float *probs_ptr;

    if (probs->storage) {
        probs_ptr = probs->storage->data + probs->storageOffset;
    } else {
        lua_pushfstring(L, "probs cannot be an empty tensor");
        lua_error(L);
    }

    float *grads_ptr;

    if (grads->storage) {
        grads_ptr = grads->storage->data + grads->storageOffset;;
    } else {
        grads_ptr = NULL; // this will trigger the score forward code path
    }

    float* costs = new float[minibatch_size];

    ctcComputeInfo info;
    info.loc = CTC_GPU;
    info.stream = THCState_getCurrentStream(cutorch_getstate(L));

    size_t gpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes,
                       (int) probs->size[1], minibatch_size,
                       info, &gpu_size_bytes);

    float* gpu_workspace;
    THCudaMalloc(cutorch_getstate(L), (void **) &gpu_workspace, gpu_size_bytes);

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes, (int) probs->size[1],
                     minibatch_size, costs,
                     gpu_workspace, info);

    lua_newtable(L);

    for (int ix = 0; ix < minibatch_size; ix++) {
        lua_pushnumber(L, costs[ix]);
        lua_rawseti(L, -2, ix+1);
    }

    THCudaFree(cutorch_getstate(L), (void *) gpu_workspace);

    delete sizes;
    delete labels_ptr;
    delete label_sizes_ptr;
    delete costs;
#endif
    return 1;
}

extern "C" int cpu_ctc(lua_State* L) {

    THFloatTensor *probs =
        static_cast<THFloatTensor *>(luaT_checkudata(L, 1, "torch.FloatTensor"));
    THFloatTensor *grads =
        static_cast<THFloatTensor *>(luaT_checkudata(L, 2, "torch.FloatTensor"));

    int* sizes;
    int* labels_ptr;
    int* label_sizes_ptr;

    int minibatch_size = processTargets(L, &sizes, &labels_ptr, &label_sizes_ptr);
    float *probs_ptr;

    if (probs->storage) {
        probs_ptr = probs->storage->data + probs->storageOffset;
    } else {
        lua_pushfstring(L, "probs cannot be an empty tensor");
        lua_error(L);
    }

    float *grads_ptr;

    if (grads->storage) {
        grads_ptr = grads->storage->data + grads->storageOffset;;
    } else {
        grads_ptr = NULL; // this will trigger the score forward code path
    }

    float* costs = new float[minibatch_size];

    ctcComputeInfo info;
    info.loc = CTC_CPU;
    info.num_threads = 0; // will use default number of threads

#if defined(CTC_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    info.num_threads = std::max(info.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes,
                       (int) probs->size[1], minibatch_size,
                       info, &cpu_size_bytes);

    float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes, probs->size[1],
                     minibatch_size, costs,
                     cpu_workspace, info);

    lua_newtable(L);

    for (int ix = 0; ix < minibatch_size; ix++) {
        lua_pushnumber(L, costs[ix]);
        lua_rawseti(L, -2, ix+1);
    }

    delete cpu_workspace;
    delete sizes;
    delete labels_ptr;
    delete label_sizes_ptr;
    delete costs;

    return 1;
}

extern "C" int luaopen_libwarp_ctc(lua_State *L) {
    lua_register(L, "gpu_ctc", gpu_ctc);
    lua_register(L, "cpu_ctc", cpu_ctc);

    return 0;
}
