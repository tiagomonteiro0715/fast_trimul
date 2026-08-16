"""Fused Triangle Multiplicative Update with CUDA-graph replay.

Builds on the fused kernels (prebound weights + fused gate / gated-residual) and
adds CUDA-graph capture: the whole ~12-launch forward is recorded once and then
replayed as a single op, removing the per-launch Python/driver overhead that
starves the GPU at small sizes.

The __main__ sweeps several problem sizes (growing N) so you can see the kernel
path go from launch-bound (loses at small N) to compute-bound (wins at large N,
where its fp16 tensor-core GEMMs matter and the fixed launch cost is amortized).

Norms use a fused LayerNorm kernel that matches nn.LayerNorm (one block per row,
all SMs, D threads per row, warp-shuffle reduction), so the kernel output matches
the torch reference to fp16 tolerance. fp16 kernel path.

Run:  python fused_trimul_update.py
"""

import copy
import os

import torch
import torch.nn as nn

import cuda.bindings.driver as cuda

from ._bootstrap import ensure_cutlass_on_path
ensure_cutlass_on_path()

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.tensor_helpers import create_cute_tensor_for_fp8

from ._tensorop_gemm import TensorOpGemm, bmm


# ===========================================================================
# GEMM infrastructure on the imported TensorOpGemm
# ===========================================================================
_TORCH_TO_CUTLASS = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}
_GEMM_CACHE = {}


def _to_cute(t, dtype, leading_dim):
    x = create_cute_tensor_for_fp8(t, dtype, leading_dim)
    x.mark_compact_shape_dynamic(
        mode=leading_dim,
        stride_order=(0, 1, 2) if leading_dim == 2 else (0, 2, 1),
        divisibility=128 // dtype.width,
    )
    return x


def _compiled(a_, b_, c_, ab_dtype, c_dtype, acc_dtype, atom, epilogue_op, key):
    if key not in _GEMM_CACHE:
        gemm = TensorOpGemm(ab_dtype, c_dtype, acc_dtype, atom, c_.leading_dim == 0)
        _GEMM_CACHE[key] = cute.compile(
            bmm, gemm, a_, b_, c_, make_fake_stream(), epilogue_op
        )
    return _GEMM_CACHE[key]


# --------------------------------------------------------------------------- #
# GEMM autotuner: try several warp (atom_layout) configs per shape, keep the
# fastest. cuBLAS picks per-shape; one fixed tile does not. Set the env var
# FAST_TRIMUL_AUTOTUNE=0 to disable (uses the (2,2,1) default, no tuning cost).
# The tuning cost is one-time per unique GEMM shape, paid on the first call.
# --------------------------------------------------------------------------- #
_AUTOTUNE = os.environ.get("FAST_TRIMUL_AUTOTUNE", "1") != "0"
_ATOM_CANDIDATES = [(2, 2, 1), (4, 1, 1), (1, 4, 1), (4, 2, 1), (2, 4, 1)]
_TUNE_CACHE = {}


def _compile_atom(a_, b_, c_, ab, cd, acc, atom, epi):
    gemm = TensorOpGemm(ab, cd, acc, atom, c_.leading_dim == 0)
    return cute.compile(bmm, gemm, a_, b_, c_, make_fake_stream(), epi)


def _time_gemm(fn, a_, b_, c_, iters=10):
    st = _stream()
    for _ in range(3):
        fn(a_, b_, c_, st)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(a_, b_, c_, st)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _autotuned(a_, b_, c_, ab, cd, acc, epi, key):
    fn = _TUNE_CACHE.get(key)
    if fn is not None:
        return fn
    # never time (which syncs) inside a CUDA-graph capture; fall back to default
    if not _AUTOTUNE or torch.cuda.is_current_stream_capturing():
        _TUNE_CACHE[key] = fn = _compile_atom(a_, b_, c_, ab, cd, acc, (2, 2, 1), epi)
        return fn
    best_fn, best_t = None, float("inf")
    for atom in _ATOM_CANDIDATES:
        try:
            cand = _compile_atom(a_, b_, c_, ab, cd, acc, atom, epi)
            t = _time_gemm(cand, a_, b_, c_)          # overwrites c_ (scratch); recomputed after
        except Exception:
            continue                                  # invalid config for this shape -> skip
        if t < best_t:
            best_fn, best_t = cand, t
    if best_fn is None:                               # nothing valid -> default (2,2,1)
        best_fn = _compile_atom(a_, b_, c_, ab, cd, acc, (2, 2, 1), epi)
    _TUNE_CACHE[key] = best_fn
    return best_fn


def _stream():
    # current torch stream, fetched every call so it follows CUDA-graph capture
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _check_dtype(t):
    if t.dtype not in _TORCH_TO_CUTLASS:
        raise TypeError(f"dtype {t.dtype} not supported; use float16 or bfloat16")
    return _TORCH_TO_CUTLASS[t.dtype]


def outgoing(a, b, acc_dtype=cutlass.Float32, atom_layout_mnk=(2, 2, 1)):
    """a, b: (B,N,N,C). z[b,i,j,c] = sum_k a[b,i,k,c]*b[b,j,k,c]. Returns (B,N,N,C)."""
    assert a.shape == b.shape and a.dtype == b.dtype
    B, N, _, C = a.shape
    assert N % 8 == 0, f"N={N} must be a multiple of 8; pad it"
    L = B * C
    ab = _check_dtype(a)
    A = a.permute(0, 3, 1, 2).reshape(L, N, N).contiguous()      # (l,m,k) k-major
    Bt = b.permute(0, 3, 1, 2).reshape(L, N, N).contiguous().transpose(-1, -2)
    Z = torch.empty(L, N, N, dtype=a.dtype, device=a.device)
    a_, b_, c_ = _to_cute(A, ab, 2), _to_cute(Bt, ab, 1), _to_cute(Z, ab, 2)
    key = ("out", L, N, a.dtype, atom_layout_mnk)
    fn = _autotuned(a_, b_, c_, ab, ab, acc_dtype, lambda x: x, key)
    fn(a_, b_, c_, _stream())
    return Z.view(B, C, N, N).permute(0, 2, 3, 1)


def incoming(a, b, acc_dtype=cutlass.Float32, atom_layout_mnk=(2, 2, 1)):
    """a, b: (B,N,N,C). z[b,i,j,c] = sum_k a[b,k,i,c]*b[b,k,j,c]. Returns (B,N,N,C)."""
    assert a.shape == b.shape and a.dtype == b.dtype
    B, N, _, C = a.shape
    assert N % 8 == 0, f"N={N} must be a multiple of 8; pad it"
    L = B * C
    ab = _check_dtype(a)
    At = a.permute(0, 3, 1, 2).reshape(L, N, N).contiguous().transpose(-1, -2)
    Bn = b.permute(0, 3, 1, 2).reshape(L, N, N).contiguous()
    Z = torch.empty(L, N, N, dtype=a.dtype, device=a.device)
    a_, b_, c_ = _to_cute(At, ab, 1), _to_cute(Bn, ab, 2), _to_cute(Z, ab, 2)
    key = ("in", L, N, a.dtype, atom_layout_mnk)
    fn = _autotuned(a_, b_, c_, ab, ab, acc_dtype, lambda x: x, key)
    fn(a_, b_, c_, _stream())
    return Z.view(B, C, N, N).permute(0, 2, 3, 1)


class KernelLinear(nn.Module):
    """nn.Linear on the `project` GEMM with the weight cute-tensor prebound.
    `raw(x)` returns the bias-free GEMM x @ weight^T (bias folded into the fused
    gate / residual kernels)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.empty(d_out))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)         # match nn.Linear default init so a
        nn.init.uniform_(self.bias, -d_in ** -0.5, d_in ** -0.5)  # fresh (unloaded) module isn't NaN
        self._key = None
        self._wcute = None
        self._fn = None

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        lead, K = tuple(x.shape[:-1]), x.shape[-1]
        N = self.weight.shape[0]
        M = 1
        for d in lead:
            M *= d
        ab = _check_dtype(x)
        X = x.detach().reshape(1, M, K).contiguous()
        Y = torch.empty(1, M, N, dtype=x.dtype, device=x.device)
        x_ = _to_cute(X, ab, 2)
        y_ = _to_cute(Y, ab, 2)
        key = (M, N, K, x.dtype)
        if self._key != key:                       # prebind weight + compile once
            W = self.weight.detach().unsqueeze(0).contiguous().transpose(-1, -2)   # (1,K,N)
            self._wcute = _to_cute(W, ab, 1)
            self._fn = _autotuned(x_, self._wcute, y_, ab, ab, cutlass.Float32,
                                  lambda t: t, ("proj", M, N, K, x.dtype))
            self._key = key
        self._fn(x_, self._wcute, y_, _stream())
        return Y.view(*lead, N)


def _proj_gate(x, lin_p, lin_g):
    """One wider GEMM for [proj; gate] (QKV-style fusion); returns the two
    bias-free halves. Turns 2 branch GEMMs into 1, and a bigger GEMM runs more
    efficiently. Weights concatenated each call (tiny); activations are the big
    traffic, so this stays correct under weight updates and cuda-graph replay."""
    K, dout = lin_p.weight.shape[1], lin_p.weight.shape[0]
    M = 1
    for d in x.shape[:-1]:
        M *= d
    ab = _check_dtype(x)
    X = x.detach().reshape(1, M, K).contiguous()
    W = torch.cat([lin_p.weight.detach(), lin_g.weight.detach()], 0)   # (2*dout, K)
    Wc = W.unsqueeze(0).contiguous().transpose(-1, -2)                 # (1, K, 2*dout)
    Y = torch.empty(1, M, 2 * dout, dtype=x.dtype, device=x.device)
    x_ = _to_cute(X, ab, 2)
    w_ = _to_cute(Wc, ab, 1)
    y_ = _to_cute(Y, ab, 2)
    key = ("pg", M, 2 * dout, K, x.dtype)
    fn = _autotuned(x_, w_, y_, ab, ab, cutlass.Float32, lambda t: t, key)
    fn(x_, w_, y_, _stream())
    Y = Y.view(*x.shape[:-1], 2 * dout)
    return Y[..., :dout], Y[..., dout:]


# ===========================================================================
# LayerNorm  (over the last/feature dim; matches nn.LayerNorm)
#   one block per row, all SMs, D threads/row, warp-shuffle reduction
# ===========================================================================
_RB = 256
_RW = _RB // 32
_Z = cutlass.Float32(0.0)
_ONE = cutlass.Float32(1.0)


@dsl_user_op
def _sigmoid_f32(x: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    return 1 / (cutlass.Float32(1.0) + cute.exp(-x))


@dsl_user_op
def _warp_sum(v: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    for off in range(5):
        v = v + cute.arch.shuffle_sync_down(v, 1 << off)
    return v


@cute.kernel
def _layernorm_kernel(inp: cute.Tensor, weight: cute.Tensor, bias: cute.Tensor,
                      out: cute.Tensor, rows: cute.Int32, D: cute.Int32, eps: cute.Float32,
                      num_blocks: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    smem = cutlass.utils.SmemAllocator()
    ssum = smem.allocate_tensor(cutlass.Float32, cute.make_layout(_RW), byte_alignment=4)
    ssq = smem.allocate_tensor(cutlass.Float32, cute.make_layout(_RW), byte_alignment=4)
    row = bx
    while row < rows:                                    # one block per row, all SMs
        s = _Z
        sq = _Z
        col = tx
        while col < D:                                   # D threads/row hide memory latency
            v = cutlass.Float32(inp[row, col])
            s = s + v
            sq = sq + v * v
            col += _RB
        s = _warp_sum(s)                                 # warp-shuffle, no smem, no barrier
        sq = _warp_sum(sq)
        if tx % 32 == 0:
            ssum[tx // 32] = s
            ssq[tx // 32] = sq
        cute.arch.barrier()
        ts = _Z
        tq = _Z
        for w in cutlass.range_constexpr(_RW):
            ts = ts + ssum[w]
            tq = tq + ssq[w]
        dn = cutlass.Float32(D)
        mean = ts / dn
        rstd = _ONE / cute.sqrt(tq / dn - mean * mean + eps)   # var = E[x^2] - E[x]^2
        col = tx
        while col < D:
            v = cutlass.Float32(inp[row, col])
            out[row, col] = (cutlass.Float32(weight[col]) * (v - mean) * rstd
                             + cutlass.Float32(bias[col])).to(out.element_type)
            col += _RB
        cute.arch.barrier()
        row += num_blocks


@cute.jit
def _layernorm_solve(inp: cute.Tensor, weight: cute.Tensor, bias: cute.Tensor,
                     out: cute.Tensor, rows: cute.Int32, D: cute.Int32, eps: cute.Float32,
                     stream: cuda.CUstream, num_blocks: cutlass.Constexpr):
    _layernorm_kernel(inp, weight, bias, out, rows, D, eps, num_blocks).launch(
        grid=(num_blocks, 1, 1), block=(_RB, 1, 1), stream=stream)


_ln_cache = {}
_P = torch.cuda.get_device_properties(0)
_RESIDENT = _P.multi_processor_count * (_P.max_threads_per_multi_processor // _RB)


def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dim, matching nn.LayerNorm:
        y = weight * (x - mean) / sqrt(var + eps) + bias
    mean/var reduced per row over the last dim (biased var, eps inside sqrt)."""
    x = x.detach().contiguous()                          # detach: dlpack can't export grad tensors
    out = torch.empty_like(x)
    D = x.shape[-1]
    rows = x.numel() // D
    x2, o2 = x.reshape(rows, D), out.reshape(rows, D)
    blocks = max(1, min(_RESIDENT, rows))
    ins = (from_dlpack(x2), from_dlpack(weight.detach().contiguous()),
           from_dlpack(bias.detach().contiguous()),
           from_dlpack(o2), cutlass.Int32(rows), cutlass.Int32(D), cutlass.Float32(eps))
    key = (rows, D, x.dtype)
    if key not in _ln_cache:
        _ln_cache[key] = cute.compile(_layernorm_solve, *ins, make_fake_stream(), blocks)
    _ln_cache[key](*ins, _stream())
    return out


# ===========================================================================
# Fused elementwise kernels (bias folded in via col = idx % D)
# ===========================================================================
@cute.kernel
def _gate_kernel(p: cute.Tensor, pb: cute.Tensor, g: cute.Tensor, gb: cute.Tensor,
                 out: cute.Tensor, NV: cute.Int32, D: cute.Int32, vec: cutlass.Constexpr):
    bx, _, _ = cute.arch.block_idx()
    bd, _, _ = cute.arch.block_dim()
    tx, _, _ = cute.arch.thread_idx()
    c = cutlass.Int32(bx * bd + tx)
    if c < NV:
        gp, gg, go = p[c, None], g[c, None], out[c, None]      # vec-wide slices
        fp = cute.make_rmem_tensor_like(gp)
        fg = cute.make_rmem_tensor_like(gg)
        fo = cute.make_rmem_tensor_like(go)
        cute.autovec_copy(gp, fp)                              # -> 128-bit load
        cute.autovec_copy(gg, fg)
        col0 = (c * vec) % D
        for k in cutlass.range_constexpr(vec):
            pv = cutlass.Float32(fp[k]) + cutlass.Float32(pb[col0 + k])
            gv = cutlass.Float32(fg[k]) + cutlass.Float32(gb[col0 + k])
            fo[k] = (pv * _sigmoid_f32(gv)).to(fo.element_type)
        cute.autovec_copy(fo, go)                              # -> 128-bit store


@cute.jit
def _gate_solve(p: cute.Tensor, pb: cute.Tensor, g: cute.Tensor, gb: cute.Tensor,
                out: cute.Tensor, NV: cute.Int32, D: cute.Int32, stream: cuda.CUstream,
                vec: cutlass.Constexpr):
    bs = 256
    _gate_kernel(p, pb, g, gb, out, NV, D, vec).launch(
        grid=((NV + bs - 1) // bs, 1, 1), block=(bs, 1, 1), stream=stream)


@cute.kernel
def _res_kernel(z: cute.Tensor, gp: cute.Tensor, gpb: cute.Tensor, y: cute.Tensor,
                yb: cute.Tensor, out: cute.Tensor, NV: cute.Int32, D: cute.Int32,
                vec: cutlass.Constexpr):
    bx, _, _ = cute.arch.block_idx()
    bd, _, _ = cute.arch.block_dim()
    tx, _, _ = cute.arch.thread_idx()
    c = cutlass.Int32(bx * bd + tx)
    if c < NV:
        gz, ggp, gy, go = z[c, None], gp[c, None], y[c, None], out[c, None]
        fz = cute.make_rmem_tensor_like(gz)
        fgp = cute.make_rmem_tensor_like(ggp)
        fy = cute.make_rmem_tensor_like(gy)
        fo = cute.make_rmem_tensor_like(go)
        cute.autovec_copy(gz, fz)                              # -> 128-bit loads
        cute.autovec_copy(ggp, fgp)
        cute.autovec_copy(gy, fy)
        col0 = (c * vec) % D
        for k in cutlass.range_constexpr(vec):
            gv = cutlass.Float32(fgp[k]) + cutlass.Float32(gpb[col0 + k])
            yv = cutlass.Float32(fy[k]) + cutlass.Float32(yb[col0 + k])
            fo[k] = (cutlass.Float32(fz[k]) + _sigmoid_f32(gv) * yv).to(fo.element_type)
        cute.autovec_copy(fo, go)                              # -> 128-bit store


@cute.jit
def _res_solve(z: cute.Tensor, gp: cute.Tensor, gpb: cute.Tensor, y: cute.Tensor,
               yb: cute.Tensor, out: cute.Tensor, NV: cute.Int32, D: cute.Int32,
               stream: cuda.CUstream, vec: cutlass.Constexpr):
    bs = 256
    _res_kernel(z, gp, gpb, y, yb, out, NV, D, vec).launch(
        grid=((NV + bs - 1) // bs, 1, 1), block=(bs, 1, 1), stream=stream)


_gate_cache, _res_cache = {}, {}


def _vec_for(t, D):
    return 8 if (t.dtype in (torch.float16, torch.bfloat16) and D % 8 == 0) else 1


def gate(p, pb, g, gb):
    """(p + pb) * sigmoid(g + gb), bias broadcast over the last dim. One kernel."""
    p, g = p.detach().contiguous(), g.detach().contiguous()
    out = torch.empty_like(p)
    D, n = p.shape[-1], p.numel()
    vec = _vec_for(p, D)
    nv, al = n // vec, (16 if vec > 1 else p.element_size())
    p2, g2, o2 = (t.reshape(nv, vec) for t in (p, g, out))
    ins = (from_dlpack(p2, assumed_align=al), from_dlpack(pb.detach()),
           from_dlpack(g2, assumed_align=al), from_dlpack(gb.detach()),
           from_dlpack(o2, assumed_align=al), cutlass.Int32(nv), cutlass.Int32(D))
    key = (n, D, p.dtype, vec)
    if key not in _gate_cache:
        _gate_cache[key] = cute.compile(_gate_solve, *ins, make_fake_stream(), vec)
    _gate_cache[key](*ins, _stream())
    return out


def gated_residual(z, gp, gpb, y, yb):
    """z + sigmoid(gp + gpb) * (y + yb), bias broadcast over the last dim. One kernel."""
    z, gp, y = z.detach().contiguous(), gp.detach().contiguous(), y.detach().contiguous()
    out = torch.empty_like(z)
    D, n = z.shape[-1], z.numel()
    vec = _vec_for(z, D)
    nv, al = n // vec, (16 if vec > 1 else z.element_size())
    z2, gp2, y2, o2 = (t.reshape(nv, vec) for t in (z, gp, y, out))
    ins = (from_dlpack(z2, assumed_align=al), from_dlpack(gp2, assumed_align=al),
           from_dlpack(gpb.detach()), from_dlpack(y2, assumed_align=al),
           from_dlpack(yb.detach()), from_dlpack(o2, assumed_align=al),
           cutlass.Int32(nv), cutlass.Int32(D))
    key = (n, D, z.dtype, vec)
    if key not in _res_cache:
        _res_cache[key] = cute.compile(_res_solve, *ins, make_fake_stream(), vec)
    _res_cache[key](*ins, _stream())
    return out


# ===========================================================================
# The module
# ===========================================================================
class TriangleMultiplicativeUpdate(nn.Module):
    """Reference (AlphaFold2 / OpenFold) implementation, native PyTorch."""

    def __init__(self, d_z: int = 128, d_c: int = 128, mode: str = 'outgoing'):
        super().__init__()
        assert mode in ['outgoing', 'incoming'], "Mode must be 'outgoing' or 'incoming'"
        self.mode = mode
        self.norm_in = nn.LayerNorm(d_z)
        self.proj_a = nn.Linear(d_z, d_c)
        self.gate_a = nn.Linear(d_z, d_c)
        self.proj_b = nn.Linear(d_z, d_c)
        self.gate_b = nn.Linear(d_z, d_c)
        self.norm_out = nn.LayerNorm(d_c)
        self.proj_g = nn.Linear(d_z, d_z)
        self.proj_out = nn.Linear(d_c, d_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = self.norm_in(z)
        a = self.proj_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))
        b = self.proj_b(z_norm) * torch.sigmoid(self.gate_b(z_norm))
        if self.mode == 'outgoing':
            m = torch.einsum('bikc,bjkc->bijc', a, b)
        else:
            m = torch.einsum('bkic,bkjc->bijc', a, b)
        m_norm = self.norm_out(m)
        g = torch.sigmoid(self.proj_g(z_norm))
        return z + g * self.proj_out(m_norm)


class _TrainableTriMul(torch.autograd.Function):
    """Makes the fused module trainable: fast fp16 kernel forward, and a CORRECT
    (but torch-recomputed, i.e. NOT yet fast) backward. Enough to train and to
    validate gradients; a fused fast backward is future work."""

    @staticmethod
    def forward(ctx, mod, z, *params):
        ctx.mod = mod
        ctx.save_for_backward(z, *params)
        with torch.no_grad():
            return mod._forward_kernels(z)

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        z = saved[0].detach().requires_grad_(True)
        params = [t.detach().requires_grad_(True) for t in saved[1:]]
        with torch.enable_grad():
            out = ctx.mod._torch_forward(z, params)
        grads = torch.autograd.grad(out, [z] + list(params), grad_out)
        return (None, *grads)


class TriangleMultiplicativeUpdateKernelFused(nn.Module):
    """Fused kernel path: prebound KernelLinear projections, fused LayerNorm
    (matches nn.LayerNorm, so norm weights load 1:1 from the reference), and
    fused gate / gated-residual kernels. Trainable via `_TrainableTriMul` when
    grad is needed (correct backward, not yet fast)."""

    def __init__(self, d_z: int = 128, d_c: int = 128, mode: str = 'outgoing',
                 residual: bool = True):
        super().__init__()
        assert mode in ['outgoing', 'incoming'], "Mode must be 'outgoing' or 'incoming'"
        self.mode = mode
        self.residual = residual              # True: return z + delta; False: return delta only
        self.norm_in = nn.LayerNorm(d_z)
        self.proj_a = KernelLinear(d_z, d_c)
        self.gate_a = KernelLinear(d_z, d_c)
        self.proj_b = KernelLinear(d_z, d_c)
        self.gate_b = KernelLinear(d_z, d_c)
        self.norm_out = nn.LayerNorm(d_c)
        self.proj_g = KernelLinear(d_z, d_z)
        self.proj_out = KernelLinear(d_c, d_z)

    def _params(self):
        return [self.norm_in.weight, self.norm_in.bias,
                self.norm_out.weight, self.norm_out.bias,
                self.proj_a.weight, self.proj_a.bias, self.gate_a.weight, self.gate_a.bias,
                self.proj_b.weight, self.proj_b.bias, self.gate_b.weight, self.gate_b.bias,
                self.proj_g.weight, self.proj_g.bias, self.proj_out.weight, self.proj_out.bias]

    def _forward_kernels(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = layer_norm(z, self.norm_in.weight, self.norm_in.bias, self.norm_in.eps)
        pa, ga = _proj_gate(z_norm, self.proj_a, self.gate_a)      # one wider GEMM
        a = gate(pa, self.proj_a.bias, ga, self.gate_a.bias)
        pb, gb = _proj_gate(z_norm, self.proj_b, self.gate_b)      # one wider GEMM
        b = gate(pb, self.proj_b.bias, gb, self.gate_b.bias)
        m = outgoing(a, b) if self.mode == 'outgoing' else incoming(a, b)
        m_norm = layer_norm(m, self.norm_out.weight, self.norm_out.bias, self.norm_out.eps)
        gp, y = self.proj_g.raw(z_norm), self.proj_out.raw(m_norm)
        if self.residual:                                          # z + sigmoid(gp)*(y+bias)
            return gated_residual(z, gp, self.proj_g.bias, y, self.proj_out.bias)
        return gate(y, self.proj_out.bias, gp, self.proj_g.bias)   # delta only (no residual)

    def _torch_forward(self, z, p):
        """Pure-torch equivalent of the kernel forward (used for the backward
        recompute); p is the parameter list from _params(), in that order."""
        import torch.nn.functional as F
        dz, dc = z.shape[-1], p[4].shape[0]
        zc = F.layer_norm(z, (dz,), p[0], p[1], self.norm_in.eps)
        a = F.linear(zc, p[4], p[5]) * torch.sigmoid(F.linear(zc, p[6], p[7]))
        b = F.linear(zc, p[8], p[9]) * torch.sigmoid(F.linear(zc, p[10], p[11]))
        if self.mode == 'outgoing':
            m = torch.einsum('bikc,bjkc->bijc', a, b)
        else:
            m = torch.einsum('bkic,bkjc->bijc', a, b)
        mc = F.layer_norm(m, (dc,), p[2], p[3], self.norm_out.eps)
        g = torch.sigmoid(F.linear(zc, p[12], p[13]))
        delta = g * F.linear(mc, p[14], p[15])
        return z + delta if self.residual else delta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and (z.requires_grad or any(t.requires_grad for t in self._params())):
            return _TrainableTriMul.apply(self, z, *self._params())   # trainable path
        return self._forward_kernels(z)                               # fast inference path


# ===========================================================================
# CUDA-graph capture + benchmark
# ===========================================================================
def make_graphed(model, z_h):
    """Capture model(z_h) into a CUDA graph; return a zero-launch-overhead replay
    closure. Falls back to eager calls if capture is rejected."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):                       # warm up: compile every kernel
                model(z_h)
        torch.cuda.current_stream().wait_stream(s)

        static_in = z_h.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = model(static_in)

        def run():
            g.replay()
            return static_out
        return run, "graph"
    except Exception as e:
        print(f"    [cuda-graph capture failed: {type(e).__name__}: {e}; eager fallback]")
        return (lambda: model(z_h)), "eager"


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


