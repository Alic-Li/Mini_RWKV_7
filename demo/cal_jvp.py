import os

import torch
from torch.utils.cpp_extension import load

os.environ["RWKV_HEAD_SIZE"] = "64"
os.environ["RWKV_MY_TESTING"] = "x070"
os.environ["RWKV_COMPILE_ON"] = "0"
os.environ["RWKV_FLOAT_MODE"] = "bf16"

ROCm_flag = torch.version.hip is not None

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

if "x070" in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16

    if ROCm_flag:
        flags = [
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "-xhip",
            "-fopenmp",
            "-ffast-math",
            "-O3",
            "-munsafe-fp-atomics",
        ]
        try:
            load(
                name="wind_backstepping_hip",
                sources=["cuda/wkv7_hip.hip", "cuda/wkv7_op.hip"],
                is_python_module=False,
                verbose=True,
                extra_cuda_cflags=flags,
            )
            print("ROCm kernel 'wind_backstepping_hip' loaded successfully.")
        except Exception as e:
            print(f"Failed to load ROCm kernel: {e}")
            print(
                "Please ensure 'cuda/wkv7_hip.hip' and 'cuda/wkv7_op.hip' exist and ROCm is set up.")
            exit()
    else:
        flags = [
            "-res-usage",
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ]
        try:
            load(
                name="wind_backstepping",
                sources=["cuda/wkv7_cuda.cu", "cuda/wkv7_op.cpp"],
                is_python_module=False,
                verbose=True,
                extra_cuda_cflags=flags,
            )
            print("CUDA kernel 'wind_backstepping' loaded successfully.")
        except Exception as e:
            print(f"Failed to load CUDA kernel: {e}")
            print(
                "Please ensure 'cuda/wkv7_cuda.cu' and 'cuda/wkv7_op.cpp' exist and CUDA is set up.")
            exit()


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])

        y = torch.empty_like(v)
        s = torch.empty(
            B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device
        )
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)

        if ROCm_flag:
            torch.ops.wind_backstepping_hip.forward(w, q, k, v, z, b, y, s, sa)
        else:
            torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)

        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype == torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])

        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [
            w, q, k, v, z, b]]

        if ROCm_flag:
            torch.ops.wind_backstepping_hip.backward(
                w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db
            )
        else:
            torch.ops.wind_backstepping.backward(
                w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db
            )
        return dw, dq, dk, dv, dz, db


def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    B, T, HC = q.shape
    q_reshaped, w_reshaped, k_reshaped, v_reshaped, a_reshaped, b_reshaped = [
        i.view(B, T, HC // HEAD_SIZE, HEAD_SIZE) for i in [q, w, k, v, a, b]
    ]
    return WindBackstepping.apply(w_reshaped, q_reshaped, k_reshaped, v_reshaped, a_reshaped, b_reshaped).view(B, T, HC)


def compute_jvp_functional(func, inputs, tangents):
    inputs_on_device = tuple(i.to("cuda").requires_grad_(True) for i in inputs)
    tangents_on_device = tuple(t.to("cuda") for t in tangents)

    output, jvp_result = torch.autograd.functional.jvp(
        func, inputs_on_device, tangents_on_device, create_graph=True
    )
    return output, jvp_result


if __name__ == "__main__":
    B = 2
    T = 32
    H = 8
    C = HEAD_SIZE
    HC = H * C

    q_input = torch.rand(B, T, HC, dtype=torch.bfloat16, device="cuda")
    w_input = torch.rand(B, T, HC, dtype=torch.bfloat16, device="cuda")
    k_input = torch.rand(B, T, HC, dtype=torch.bfloat16, device="cuda")
    v_input = torch.rand(B, T, HC, dtype=torch.bfloat16, device="cuda")
    a_input = torch.rand(B, T, HC, dtype=torch.bfloat16, device="cuda")
    b_input = torch.rand(B, T, HC, dtype=torch.bfloat16, device="cuda")

    vq_input = torch.rand_like(q_input)
    vw_input = torch.rand_like(w_input)
    vk_input = torch.rand_like(k_input)
    vv_input = torch.rand_like(v_input)
    va_input = torch.rand_like(a_input)
    vb_input = torch.rand_like(b_input)

    inputs_for_jvp = (q_input, w_input, k_input, v_input, a_input, b_input)
    tangents_for_jvp = (vq_input, vw_input, vk_input,
                        vv_input, va_input, vb_input)

    print(f"Running JVP for RUN_CUDA_RWKV7g with B={B}, T={T}, HC={HC}")

    try:
        output_y, jvp_result = compute_jvp_functional(
            RUN_CUDA_RWKV7g, inputs_for_jvp, tangents_for_jvp
        )

        print("\nJVP Output (value of the function at the inputs):")
        print(
            f"Shape: {output_y.shape}, Dtype: {output_y.dtype}, Device: {output_y.device}")

        print("\nJVP Result (Jacobian-vector product):")
        print(
            f"Shape: {jvp_result.shape}, Dtype: {jvp_result.dtype}, Device: {jvp_result.device}")

        print("\nJVP calculation successful!")

    except Exception as e:
        print(f"\nAn error occurred during JVP computation: {e}")
        print("Please ensure the CUDA/HIP kernel files are in the 'cuda/' directory relative to this script,")
        print("and your PyTorch installation supports your GPU.")
