import torch as tr
import torch.nn as nn
from torch import autograd
import numpy as np
import jax
import jax.numpy as jnp


class mmd2_noise_injection(autograd.Function):

    @staticmethod
    def forward(ctx,true_feature,fake_feature,noisy_feature):
        b_size,d, n_particles = noisy_feature.shape
        with  tr.enable_grad():

            mmd2 = tr.mean((true_feature-fake_feature)**2)
            mean_noisy_feature = tr.mean(noisy_feature,dim = -1 )

            mmd2_for_grad = (n_particles/b_size)*(tr.einsum('nd,nd->',fake_feature,mean_noisy_feature) - tr.einsum('nd,nd->',true_feature,mean_noisy_feature))

        ctx.save_for_backward(mmd2_for_grad,noisy_feature)

        return mmd2

    @staticmethod
    def backward(ctx, grad_output):
        mmd2_for_grad, noisy_feature = ctx.saved_tensors
        with  tr.enable_grad():
            gradients = autograd.grad(outputs=mmd2_for_grad, inputs=noisy_feature,
                        grad_outputs=grad_output,
                        create_graph=True, only_inputs=True)[0] 
                
        return None, None, gradients


class mmd2_func(autograd.Function):

    @staticmethod
    def forward(ctx,true_feature,fake_feature):

        b_size,d, n_particles = fake_feature.shape

        with  tr.enable_grad():

            mmd2 = (n_particles/b_size)*tr.sum((true_feature-tr.mean(fake_feature,dim=-1))**2)

        ctx.save_for_backward(mmd2,fake_feature)

        return (1./n_particles)*mmd2

    @staticmethod
    def backward(ctx, grad_output):

        mmd2, fake_feature = ctx.saved_tensors
        with  tr.enable_grad():
            gradients = autograd.grad(outputs=mmd2, inputs=fake_feature,
                        grad_outputs=grad_output,
                        create_graph=True, only_inputs=True)[0] 
                
        return None, gradients


class drmmd_func(autograd.Function):

    @staticmethod
    def forward(ctx,true_feature,fake_feature, noisy_fake_feature, lmbda):
        b_size, d, n_particles = fake_feature.shape
        m_particles = 1
        true_feature = true_feature[:, :, None]
        fake_feature = fake_feature.clone().detach()
        with  tr.enable_grad():
            K_XX = tr.einsum('ijk,ijl->jkl', true_feature, true_feature).sum(0) / b_size
            K_YX = tr.einsum('ijk,ijl->jkl', fake_feature, true_feature).sum(0) / b_size
            K_YY = tr.einsum('ijk,ijl->jkl', fake_feature, fake_feature).sum(0) / b_size
            inv_K_XX = tr.linalg.inv(K_XX + m_particles * lmbda * tr.eye(K_XX.shape[0]).to(fake_feature.device))
            part1 = K_YY.mean() + K_XX.mean() - 2 * K_YX.mean()
            part2 = -(K_YX @ inv_K_XX @ K_YX.T).mean()
            part3 = (K_XX.T @ inv_K_XX @ K_YX.T).mean() * 2
            part4 = -(K_XX.T @ inv_K_XX @ K_XX).mean()
            drmmd = 0.5 * (part1 + part2 + part3 + part4) * (1 + lmbda) / lmbda

            K_noisyY_X = tr.einsum('ijk,ijl->jkl', noisy_fake_feature, true_feature).sum(0) / b_size
            K_noisyY_Y = tr.einsum('ijk,ijl->jkl', noisy_fake_feature, fake_feature).sum(0) / b_size

            part1 = K_noisyY_Y.mean() - K_noisyY_X.mean()
            part2 = - (K_noisyY_X @ inv_K_XX @ K_YX.T).mean()
            part3 = (K_noisyY_X @ inv_K_XX @ K_XX).mean()
            drmmd_first_variation = (part1 + part2 + part3) / lmbda * (1 + lmbda)
            drmmd_first_variation = drmmd_first_variation * n_particles

        ctx.save_for_backward(drmmd_first_variation, noisy_fake_feature)
        return drmmd

    @staticmethod
    def backward(ctx, grad_output):

        drmmd_first_variation, noisy_fake_feature = ctx.saved_tensors
        with  tr.enable_grad():
            gradients = autograd.grad(outputs=drmmd_first_variation, inputs=noisy_fake_feature,
                        grad_outputs=grad_output,
                        create_graph=True, only_inputs=True)[0] 
        return None, None, gradients, None
# class srmmd_func(autograd.Function):
#     """使用JAX计算梯度，PyTorch进行反向传播的混合SRMMD实现"""

#     @staticmethod
#     def _to_jax(tensor):
#         """PyTorch -> JAX"""
#         return jnp.array(tensor.detach().cpu().numpy())
    
#     @staticmethod
#     def _to_torch(array, device, dtype):
#         """JAX -> PyTorch"""
#         return tr.from_numpy(np.array(array)).to(device=device, dtype=dtype)

#     @staticmethod
#     @jax.jit
#     def _compute_srmmd_jax(fake_features_jax, true_features_jax, lmbda):
#         """
#         使用JAX计算SRMMD的所有矩阵（JIT编译加速）
        
#         Args:
#             fake_features_jax: [b_size, d, N] - JAX array
#             true_features_jax: [b_size, d, M] - JAX array (M=1 for single teacher)
#             lmbda: float
            
#         Returns:
#             srmmd_val, v, D_YY, D_YX (all JAX arrays)
#         """
#         b_size, d, N = fake_features_jax.shape
#         M = true_features_jax.shape[2]
        
#         # ============================================
#         # 1. Gram matrices
#         # ============================================
#         K_YY = jnp.einsum('ijk,ijl->kl', fake_features_jax, fake_features_jax) / b_size
#         K_XX = jnp.einsum('ijk,ijl->kl', true_features_jax, true_features_jax) / b_size
#         K_YX = jnp.einsum('ijk,ijl->kl', fake_features_jax, true_features_jax) / b_size
        
#         base_mmd = K_YY.mean() + K_XX.mean() - 2.0 * K_YX.mean()
        
#         # ============================================
#         # 2. 定义kernel和导数函数
#         # ============================================
#         def k_empirical(psi_i, psi_j):
#             """k(y_i, y_j) = (1/B) Σ_z ψ(z,y_i)^T ψ(z,y_j)"""
#             return jnp.sum(psi_i * psi_j) / b_size
        
#         grad_fn = jax.grad(k_empirical, argnums=0)
#         hess_fn = jax.jacfwd(jax.jacrev(k_empirical, argnums=1), argnums=0)
        
#         # ============================================
#         # 3. 计算D矩阵: [N*d, M]
#         # ============================================
#         def compute_D_matrix(features_A, features_B):
#             """D[i*d:(i+1)*d, j] = ∇_{ψ_i} k(ψ_i, ψ_j) averaged over batch"""
#             def grad_ij(i, j):
#                 psi_i = features_A[:, :, i]  # [b_size, d]
#                 psi_j = features_B[:, :, j]  # [b_size, d]
#                 grad = grad_fn(psi_i, psi_j)  # [b_size, d]
#                 return grad.mean(axis=0)      # [d]
            
#             N_A = features_A.shape[2]
#             N_B = features_B.shape[2]
            
#             # 使用vmap的正确方式：将i作为参数传入
#             def compute_row_i(i):
#                 return jax.vmap(lambda j: grad_ij(i, j))(jnp.arange(N_B))
            
#             D = jax.vmap(compute_row_i)(jnp.arange(N_A))  # [N_A, N_B, d]
#             D = jnp.transpose(D, (0, 2, 1))                # [N_A, d, N_B]
#             return D.reshape(N_A * d, N_B)
        
#         D_YY = compute_D_matrix(fake_features_jax, fake_features_jax)  # [N*d, N]
#         D_YX = compute_D_matrix(fake_features_jax, true_features_jax)  # [N*d, M]
        
#         # ============================================
#         # 4. 计算H_YY: [N*d, N*d]
#         # ============================================
#         def compute_H_matrix(features):
#             """H[i*d:(i+1)*d, j*d:(j+1)*d] = ∇²_{ψ_i,ψ_j} k(ψ_i, ψ_j)"""
#             def hess_ij(i, j):
#                 psi_i = features[:, :, i]
#                 psi_j = features[:, :, j]
#                 H = hess_fn(psi_i, psi_j)     # [b_size, d, b_size, d]
#                 return H.mean(axis=(0, 2))    # [d, d]
            
#             N_f = features.shape[2]
            
#             def compute_row_i(i):
#                 return jax.vmap(lambda j: hess_ij(i, j))(jnp.arange(N_f))
            
#             H = jax.vmap(compute_row_i)(jnp.arange(N_f))  # [N, N, d, d]
#             H = jnp.transpose(H, (0, 2, 1, 3))             # [N, d, N, d]
#             return H.reshape(N_f * d, N_f * d)
        
#         H_YY = compute_H_matrix(fake_features_jax)  # [N*d, N*d]
        
#         # ============================================
#         # 5. 求解witness函数
#         # ============================================
#         ones_N = jnp.ones(N)
#         ones_M = jnp.ones(M)
        
#         r = (D_YY @ ones_N) / N - (D_YX @ ones_M) / M
        
#         reg = lmbda * N
#         reg_matrix = H_YY + reg * jnp.eye(H_YY.shape[0])
#         v = jax.scipy.linalg.solve(reg_matrix, r, assume_a='pos')
        
#         srmmd_val = (base_mmd - (r @ v)) / lmbda
        
#         return srmmd_val, v, D_YY, D_YX

#     @staticmethod
#     def forward(ctx, true_feature, fake_feature, noisy_fake_feature, lmbda):
#         """
#         Forward pass using JAX.
        
#         Args:
#             true_feature: [b_size, d] - teacher features (PyTorch)
#             fake_feature: [b_size, d, n_particles] - student features, detached (PyTorch)
#             noisy_fake_feature: [b_size, d, n_particles] - for backward (PyTorch)
#             lmbda: float
#         """
#         device = fake_feature.device
#         dtype = fake_feature.dtype
#         b_size, d = true_feature.shape
#         _, _, n_particles = fake_feature.shape
        
#         # ============================================
#         # 转换到JAX并计算（快速！）
#         # ============================================
#         true_jax = srmmd_func._to_jax(true_feature).reshape(b_size, d, 1)
#         fake_jax = srmmd_func._to_jax(fake_feature)
#         noisy_jax = srmmd_func._to_jax(noisy_fake_feature)
        
#         # 使用JIT编译的JAX函数计算主要矩阵
#         srmmd_val_jax, v_jax, D_YY_jax, D_YX_jax = srmmd_func._compute_srmmd_jax(
#             fake_jax, true_jax, float(lmbda)
#         )
        
#         # 转换回PyTorch
#         srmmd_val = srmmd_func._to_torch(srmmd_val_jax, device, dtype)
#         v = srmmd_func._to_torch(v_jax, device, dtype)
        
#         # ============================================
#         # 计算first variation (PyTorch, 用于backward)
#         # ============================================
#         with tr.enable_grad():
#             # 确保noisy_fake_feature有梯度
#             noisy_fake_feature = noisy_fake_feature.requires_grad_(True)
            
#             # Kernel均值项
#             K_noisyY_Y = tr.einsum('ijk,ijl->kl', 
#                 noisy_fake_feature, fake_feature.detach()) / b_size
#             K_noisyY_X = tr.einsum('ijk,ijl->kl', 
#                 noisy_fake_feature, true_feature.unsqueeze(2)) / b_size
            
#             mean_diff = K_noisyY_Y.mean() - K_noisyY_X.mean()
            
#             # 为noisy features计算d_noisy
#             # 关键修复：分别计算两个D矩阵，而不是调用完整的srmmd计算
            
#             # 定义计算单个D矩阵的函数
#             @jax.jit
#             def compute_D_only(features_A, features_B):
#                 """只计算D矩阵，不计算完整SRMMD"""
#                 b_size_local = features_A.shape[0]
#                 d_local = features_A.shape[1]
                
#                 def k_emp(psi_i, psi_j):
#                     return jnp.sum(psi_i * psi_j) / b_size_local
                
#                 grad_fn = jax.grad(k_emp, argnums=0)
                
#                 def grad_ij(i, j):
#                     psi_i = features_A[:, :, i]
#                     psi_j = features_B[:, :, j]
#                     grad = grad_fn(psi_i, psi_j)
#                     return grad.mean(axis=0)
                
#                 N_A = features_A.shape[2]
#                 N_B = features_B.shape[2]
                
#                 def compute_row_i(i):
#                     return jax.vmap(lambda j: grad_ij(i, j))(jnp.arange(N_B))
                
#                 D = jax.vmap(compute_row_i)(jnp.arange(N_A))
#                 D = jnp.transpose(D, (0, 2, 1))
#                 return D.reshape(N_A * d_local, N_B)
            
#             # 计算d_noisy的两个部分
#             d_noisy_Y_jax = compute_D_only(noisy_jax, fake_jax)    # [N*d, N]
#             d_noisy_X_jax = compute_D_only(noisy_jax, true_jax)    # [N*d, 1]
            
#             ones_N_jax = jnp.ones(n_particles)
#             ones_M_jax = jnp.ones(1)
            
#             d_noisy_jax = (d_noisy_Y_jax @ ones_N_jax) / n_particles - (d_noisy_X_jax @ ones_M_jax)
#             d_noisy = srmmd_func._to_torch(d_noisy_jax, device, dtype)
            
#             correction = d_noisy @ v
#             srmmd_first_variation = (mean_diff - correction) / lmbda * n_particles
        
#         ctx.save_for_backward(srmmd_first_variation, noisy_fake_feature)
#         return srmmd_val

#     @staticmethod
#     def backward(ctx, grad_output):
#         srmmd_first_variation, noisy_fake_feature = ctx.saved_tensors
        
#         with tr.enable_grad():
#             gradients = autograd.grad(
#                 outputs=srmmd_first_variation,
#                 inputs=noisy_fake_feature,
#                 grad_outputs=grad_output,
#                 create_graph=True,
#                 only_inputs=True
#             )[0]
        
#         return None, None, gradients, None
class srmmd_func(autograd.Function):
    """使用JAX计算梯度，PyTorch进行反向传播的混合SRMMD实现"""

    @staticmethod
    def _to_jax(tensor):
        """PyTorch -> JAX"""
        return jnp.array(tensor.detach().cpu().numpy())
    
    @staticmethod
    def _to_torch(array, device, dtype):
        """JAX -> PyTorch"""
        return tr.from_numpy(np.array(array)).to(device=device, dtype=dtype)

    @staticmethod
    @jax.jit
    def _compute_D_matrix_jax(features_A, features_B, b_size):
        """
        JIT编译的D矩阵计算函数
        
        Args:
            features_A: [b_size, d, N_A]
            features_B: [b_size, d, N_B]
            b_size: batch size (作为参数传入以支持JIT)
            
        Returns:
            D: [N_A*d, N_B]
        """
        d = features_A.shape[1]
        
        def k_empirical(psi_i, psi_j):
            """k(y_i, y_j) = (1/B) Σ_z ψ(z,y_i)^T ψ(z,y_j)"""
            return jnp.sum(psi_i * psi_j) / b_size
        
        grad_fn = jax.grad(k_empirical, argnums=0)
        
        def grad_ij(i, j):
            psi_i = features_A[:, :, i]  # [b_size, d]
            psi_j = features_B[:, :, j]  # [b_size, d]
            grad = grad_fn(psi_i, psi_j)  # [b_size, d]
            return grad.mean(axis=0)      # [d]
        
        N_A = features_A.shape[2]
        N_B = features_B.shape[2]
        
        # 使用vmap的正确方式：将i作为参数传入
        def compute_row_i(i):
            return jax.vmap(lambda j: grad_ij(i, j))(jnp.arange(N_B))
        
        D = jax.vmap(compute_row_i)(jnp.arange(N_A))  # [N_A, N_B, d]
        D = jnp.transpose(D, (0, 2, 1))                # [N_A, d, N_B]
        return D.reshape(N_A * d, N_B)

    @staticmethod
    @jax.jit
    def _compute_H_matrix_jax(features, b_size):
        """
        JIT编译的Hessian矩阵计算函数
        
        Args:
            features: [b_size, d, N]
            b_size: batch size
            
        Returns:
            H: [N*d, N*d]
        """
        d = features.shape[1]
        
        def k_empirical(psi_i, psi_j):
            return jnp.sum(psi_i * psi_j) / b_size
        
        hess_fn = jax.jacfwd(jax.jacrev(k_empirical, argnums=1), argnums=0)
        
        def hess_ij(i, j):
            psi_i = features[:, :, i]
            psi_j = features[:, :, j]
            H = hess_fn(psi_i, psi_j)     # [b_size, d, b_size, d]
            return H.mean(axis=(0, 2))    # [d, d]
        
        N_f = features.shape[2]
        
        def compute_row_i(i):
            return jax.vmap(lambda j: hess_ij(i, j))(jnp.arange(N_f))
        
        H = jax.vmap(compute_row_i)(jnp.arange(N_f))  # [N, N, d, d]
        H = jnp.transpose(H, (0, 2, 1, 3))             # [N, d, N, d]
        return H.reshape(N_f * d, N_f * d)

    @staticmethod
    @jax.jit
    def _compute_srmmd_jax(fake_features_jax, true_features_jax, lmbda, b_size):
        """
        使用JAX计算SRMMD的所有矩阵（JIT编译加速）
        
        Args:
            fake_features_jax: [b_size, d, N] - JAX array
            true_features_jax: [b_size, d, M] - JAX array (M=1 for single teacher)
            lmbda: float
            b_size: int (作为参数传入以支持JIT)
            
        Returns:
            srmmd_val, v (all JAX arrays)
        """
        d, N = fake_features_jax.shape[1], fake_features_jax.shape[2]
        M = true_features_jax.shape[2]
        
        # ============================================
        # 1. Gram matrices
        # ============================================
        K_YY = jnp.einsum('ijk,ijl->kl', fake_features_jax, fake_features_jax) / b_size
        K_XX = jnp.einsum('ijk,ijl->kl', true_features_jax, true_features_jax) / b_size
        K_YX = jnp.einsum('ijk,ijl->kl', fake_features_jax, true_features_jax) / b_size
        
        base_mmd = K_YY.mean() + K_XX.mean() - 2.0 * K_YX.mean()
        
        # ============================================
        # 2. 计算D矩阵使用专门的JIT函数
        # ============================================
        D_YY = srmmd_func._compute_D_matrix_jax(fake_features_jax, fake_features_jax, b_size)
        D_YX = srmmd_func._compute_D_matrix_jax(fake_features_jax, true_features_jax, b_size)
        
        # ============================================
        # 3. 计算H矩阵
        # ============================================
        H_YY = srmmd_func._compute_H_matrix_jax(fake_features_jax, b_size)
        
        # ============================================
        # 4. 求解witness函数
        # ============================================
        ones_N = jnp.ones(N)
        ones_M = jnp.ones(M)
        
        r = (D_YY @ ones_N) / N - (D_YX @ ones_M) / M
        
        reg = lmbda * N
        reg_matrix = H_YY + reg * jnp.eye(H_YY.shape[0])
        # v = jax.scipy.linalg.solve(reg_matrix, r, assume_a='pos')
        # v = jax.scipy.linalg.solve(reg_matrix, r)
        v, info = jax.scipy.sparse.linalg.cg(reg_matrix, r, tol=1e-6, maxiter=100)
        
        srmmd_val = (base_mmd - (r @ v)) / lmbda
        
        return srmmd_val, v

    @staticmethod
    def forward(ctx, true_feature, fake_feature, noisy_fake_feature, lmbda):
        """
        Forward pass using JAX.
        
        Args:
            true_feature: [b_size, d] - teacher features (PyTorch)
            fake_feature: [b_size, d, n_particles] - student features, detached (PyTorch)
            noisy_fake_feature: [b_size, d, n_particles] - for backward (PyTorch)
            lmbda: float
        """
        device = fake_feature.device
        dtype = fake_feature.dtype
        b_size, d = true_feature.shape
        _, _, n_particles = fake_feature.shape
        
        # ============================================
        # 转换到JAX并计算（快速！）
        # ============================================
        true_jax = srmmd_func._to_jax(true_feature).reshape(b_size, d, 1)
        fake_jax = srmmd_func._to_jax(fake_feature)
        noisy_jax = srmmd_func._to_jax(noisy_fake_feature)
        
        # 使用JIT编译的JAX函数计算主要矩阵
        srmmd_val_jax, v_jax = srmmd_func._compute_srmmd_jax(
            fake_jax, true_jax, float(lmbda), b_size
        )
        
        # 转换回PyTorch
        srmmd_val = srmmd_func._to_torch(srmmd_val_jax, device, dtype)
        v = srmmd_func._to_torch(v_jax, device, dtype)
        
        # ============================================
        # 计算first variation (PyTorch, 用于backward)
        # ============================================
        with tr.enable_grad():
            # 确保noisy_fake_feature有梯度
            noisy_fake_feature = noisy_fake_feature.requires_grad_(True)
            
            # Kernel均值项
            K_noisyY_Y = tr.einsum('ijk,ijl->kl', 
                noisy_fake_feature, fake_feature.detach()) / b_size
            K_noisyY_X = tr.einsum('ijk,ijl->kl', 
                noisy_fake_feature, true_feature.unsqueeze(2)) / b_size
            
            mean_diff = K_noisyY_Y.mean() - K_noisyY_X.mean()
            
            # 使用预编译的JIT函数计算d_noisy（现在很快！）
            d_noisy_Y_jax = srmmd_func._compute_D_matrix_jax(noisy_jax, fake_jax, b_size)
            d_noisy_X_jax = srmmd_func._compute_D_matrix_jax(noisy_jax, true_jax, b_size)
            
            ones_N_jax = jnp.ones(n_particles)
            ones_M_jax = jnp.ones(1)
            
            d_noisy_jax = (d_noisy_Y_jax @ ones_N_jax) / n_particles - (d_noisy_X_jax @ ones_M_jax)
            d_noisy = srmmd_func._to_torch(d_noisy_jax, device, dtype)
            
            correction = d_noisy @ v
            srmmd_first_variation = (mean_diff - correction) / lmbda * n_particles
        
        ctx.save_for_backward(srmmd_first_variation, noisy_fake_feature)
        return srmmd_val

    @staticmethod
    def backward(ctx, grad_output):
        srmmd_first_variation, noisy_fake_feature = ctx.saved_tensors
        
        with tr.enable_grad():
            gradients = autograd.grad(
                outputs=srmmd_first_variation,
                inputs=noisy_fake_feature,
                grad_outputs=grad_output,
                create_graph=True,
                only_inputs=True
            )[0]
        
        return None, None, gradients, None

class hrmmd_func(autograd.Function):
    """同时包含函数值 L2(mu) 惩罚 + 梯度惩罚 的 block-SrMMD 实现"""

    @staticmethod
    def _to_jax(tensor):
        return jnp.array(tensor.detach().cpu().numpy())

    @staticmethod
    def _to_torch(array, device, dtype):
        return tr.from_numpy(np.array(array)).to(device=device, dtype=dtype)

    @staticmethod
    @jax.jit
    def _compute_D_matrix_jax(features_A, features_B, b_size):
        """
        D[(i,ell), j] = d/d first-arg_ell k(x_i, y_j)
        返回 shape: [N_A * d, N_B]
        """
        d = features_A.shape[1]

        def k_empirical(psi_i, psi_j):
            return jnp.sum(psi_i * psi_j) / b_size

        grad_fn = jax.grad(k_empirical, argnums=0)

        def grad_ij(i, j):
            psi_i = features_A[:, :, i]   # [b_size, d]
            psi_j = features_B[:, :, j]   # [b_size, d]
            grad = grad_fn(psi_i, psi_j)  # [b_size, d]
            return grad.mean(axis=0)      # [d]

        N_A = features_A.shape[2]
        N_B = features_B.shape[2]

        def compute_row_i(i):
            return jax.vmap(lambda j: grad_ij(i, j))(jnp.arange(N_B))

        D = jax.vmap(compute_row_i)(jnp.arange(N_A))  # [N_A, N_B, d]
        D = jnp.transpose(D, (0, 2, 1))               # [N_A, d, N_B]
        return D.reshape(N_A * d, N_B)

    @staticmethod
    @jax.jit
    def _compute_H_matrix_jax(features, b_size):
        """
        H[(i,ell),(j,m)] = d^2 / d first-arg_ell d second-arg_m k(x_i, x_j)
        返回 shape: [N*d, N*d]
        """
        d = features.shape[1]

        def k_empirical(psi_i, psi_j):
            return jnp.sum(psi_i * psi_j) / b_size

        hess_fn = jax.jacfwd(jax.jacrev(k_empirical, argnums=1), argnums=0)

        def hess_ij(i, j):
            psi_i = features[:, :, i]
            psi_j = features[:, :, j]
            H = hess_fn(psi_i, psi_j)   # [b_size, d, b_size, d]
            return H.mean(axis=(0, 2))  # [d, d]

        N_f = features.shape[2]

        def compute_row_i(i):
            return jax.vmap(lambda j: hess_ij(i, j))(jnp.arange(N_f))

        H = jax.vmap(compute_row_i)(jnp.arange(N_f))  # [N, N, d, d]
        H = jnp.transpose(H, (0, 2, 1, 3))            # [N, d, N, d]
        return H.reshape(N_f * d, N_f * d)

    @staticmethod
    @jax.jit
    def _compute_hrmmd_jax(fake_features_jax, true_features_jax, lmbda, b_size):
        """
        按 block system 计算:
            A = [[K_YY, D_YY^T],
                 [D_YY, H_YY]] + N lambda I

        rhs = [g; r]
        g = K_YY 1_N / N - K_YX 1_M / M
        r = D_YY 1_N / N - D_YX 1_M / M

        hrmmd = (base_mmd - rhs^T A^{-1} rhs) / lambda
        """
        N = fake_features_jax.shape[2]
        M = true_features_jax.shape[2]
        d = fake_features_jax.shape[1]

        # =========================
        # 1. Gram matrices
        # =========================
        K_YY = jnp.einsum('ijk,ijl->kl', fake_features_jax, fake_features_jax) / b_size  # [N, N]
        K_XX = jnp.einsum('ijk,ijl->kl', true_features_jax, true_features_jax) / b_size  # [M, M]
        K_YX = jnp.einsum('ijk,ijl->kl', fake_features_jax, true_features_jax) / b_size  # [N, M]

        base_mmd = K_YY.mean() + K_XX.mean() - 2.0 * K_YX.mean()

        # =========================
        # 2. D / H matrices
        # =========================
        D_YY = hrmmd_func._compute_D_matrix_jax(fake_features_jax, fake_features_jax, b_size)  # [N*d, N]
        D_YX = hrmmd_func._compute_D_matrix_jax(fake_features_jax, true_features_jax, b_size)  # [N*d, M]
        H_YY = hrmmd_func._compute_H_matrix_jax(fake_features_jax, b_size)                      # [N*d, N*d]

        # =========================
        # 3. block rhs
        # =========================
        ones_N = jnp.ones((N,), dtype=fake_features_jax.dtype)
        ones_M = jnp.ones((M,), dtype=fake_features_jax.dtype)

        g = (K_YY @ ones_N) / N - (K_YX @ ones_M) / M          # [N]
        r = (D_YY @ ones_N) / N - (D_YX @ ones_M) / M          # [N*d]
        rhs = jnp.concatenate([g, r], axis=0)                  # [N + N*d]

        # =========================
        # 4. block system
        # =========================
        top = jnp.concatenate([K_YY, D_YY.T], axis=1)          # [N, N + N*d]
        bot = jnp.concatenate([D_YY, H_YY], axis=1)            # [N*d, N + N*d]
        A = jnp.concatenate([top, bot], axis=0)                # [N + N*d, N + N*d]

        reg = (N * lmbda) * jnp.eye(N + N * d, dtype=fake_features_jax.dtype)
        A = A + reg

        # sol = jnp.linalg.solve(A, rhs)
        sol, info = jax.scipy.sparse.linalg.cg(A, rhs, tol=1e-6, maxiter=100)
        alpha_k = sol[:N]       # 对应函数值特征 K_X
        alpha_d = sol[N:]       # 对应导数特征 D_X

        hrmmd_val = (base_mmd - (rhs @ sol)) / lmbda
        return hrmmd_val, alpha_k, alpha_d

    @staticmethod
    def forward(ctx, true_feature, fake_feature, noisy_fake_feature, lmbda):
        """
        Args:
            true_feature: [b_size, d]
            fake_feature: [b_size, d, n_particles]
            noisy_fake_feature: [b_size, d, n_particles]
            lmbda: float
        """
        device = fake_feature.device
        dtype = fake_feature.dtype
        b_size, d = true_feature.shape
        _, _, n_particles = fake_feature.shape
        m_particles = 1

        true_jax = hrmmd_func._to_jax(true_feature).reshape(b_size, d, 1)
        fake_jax = hrmmd_func._to_jax(fake_feature)
        noisy_jax = hrmmd_func._to_jax(noisy_fake_feature)

        hrmmd_val_jax, alpha_k_jax, alpha_d_jax = hrmmd_func._compute_hrmmd_jax(
            fake_jax, true_jax, float(lmbda), b_size
        )

        hrmmd_val = hrmmd_func._to_torch(hrmmd_val_jax, device, dtype)
        alpha_k = hrmmd_func._to_torch(alpha_k_jax, device, dtype)
        alpha_d = hrmmd_func._to_torch(alpha_d_jax, device, dtype)

        # =========================
        # first variation for backward
        # =========================
        with tr.enable_grad():
            noisy_fake_feature = noisy_fake_feature.requires_grad_(True)

            # k(noisy, fake) and k(noisy, true)
            K_noisyY_Y = tr.einsum(
                'ijk,ijl->kl',
                noisy_fake_feature,
                fake_feature.detach()
            ) / b_size                                                  # [N, N]

            K_noisyY_X = tr.einsum(
                'ijk,ijl->kl',
                noisy_fake_feature,
                true_feature.unsqueeze(2)
            ) / b_size                                                  # [N, 1]

            mean_diff = K_noisyY_Y.mean() - K_noisyY_X.mean()

            ones_N_t = tr.ones(n_particles, device=device, dtype=dtype)
            ones_M_t = tr.ones(m_particles, device=device, dtype=dtype)

            # 新增的函数值特征校正项
            k_noisy = (K_noisyY_Y @ ones_N_t) / n_particles - (K_noisyY_X @ ones_M_t) / m_particles  # [N]

            # 原来的导数特征校正项
            D_noisyY_Y_jax = hrmmd_func._compute_D_matrix_jax(noisy_jax, fake_jax, b_size)   # [N*d, N]
            D_noisyY_X_jax = hrmmd_func._compute_D_matrix_jax(noisy_jax, true_jax, b_size)   # [N*d, 1]

            D_noisyY_Y = hrmmd_func._to_torch(D_noisyY_Y_jax, device, dtype)
            D_noisyY_X = hrmmd_func._to_torch(D_noisyY_X_jax, device, dtype)

            d_noisy = (D_noisyY_Y @ ones_N_t) / n_particles - (D_noisyY_X @ ones_M_t) / m_particles  # [N*d]

            correction = k_noisy @ alpha_k + d_noisy @ alpha_d
            hrmmd_first_variation = (mean_diff - correction) / lmbda * n_particles

        ctx.save_for_backward(hrmmd_first_variation, noisy_fake_feature)
        return hrmmd_val

    @staticmethod
    def backward(ctx, grad_output):
        hrmmd_first_variation, noisy_fake_feature = ctx.saved_tensors

        with tr.enable_grad():
            gradients = autograd.grad(
                outputs=hrmmd_first_variation,
                inputs=noisy_fake_feature,
                grad_outputs=grad_output,
                create_graph=True,
                only_inputs=True
            )[0]

        return None, None, gradients, None
    

class sobolev(autograd.Function):
    @staticmethod
    def forward(ctx,true_feature,fake_feature,matrix):

        b_size,_, n_particles = fake_feature.shape

        m = tr.mean(fake_feature,dim=-1) -  true_feature

        alpha = tr.solve(m,matrix)[0].clone().detach()

        with  tr.enable_grad():

            mmd2 = (0.5*n_particles/b_size)*tr.sum((true_feature-tr.mean(fake_feature,dim=-1))**2)
            mmd2_for_grad = (1./b_size)*tr.einsum('id,idm->',alpha,fake_feature)
        
        ctx.save_for_backward(mmd2_for_grad,fake_feature)

        return (1./n_particles)*mmd2

    @staticmethod
    def backward(ctx, grad_output):
        mmd2, fake_feature = ctx.saved_tensors
        with  tr.enable_grad():
            gradients = autograd.grad(outputs=mmd2, inputs=fake_feature,
                        grad_outputs=grad_output,
                        create_graph=True, only_inputs=True)[0] 
                
        return None, gradients,None


class drmmd(nn.Module):
    def __init__(self,student,with_noise,lmbda):
        super(drmmd, self).__init__()
        self.student = student
        self.drmmd = drmmd_func.apply
        self.with_noise=with_noise
        self.lmbda = lmbda
    def forward(self,x,y):
        out = self.student(x)
        self.student.set_noisy_mode(self.with_noise)
        noisy_out = self.student(x)
        loss = self.drmmd(y, out, noisy_out, self.lmbda)
        return loss


class srmmd(nn.Module):
    def __init__(self,student,with_noise,lmbda):
        super(srmmd, self).__init__()
        self.student = student
        self.srmmd = srmmd_func.apply
        self.with_noise=with_noise
        self.lmbda = lmbda
        
    def forward(self,x,y):
        out = self.student(x)
        self.student.set_noisy_mode(self.with_noise)
        noisy_out = self.student(x)
        loss = self.srmmd(y, out, noisy_out, self.lmbda)
        return loss

class hrmmd(nn.Module):
    def __init__(self, student, with_noise, lmbda):
        super(hrmmd, self).__init__()
        self.student = student
        self.hrmmd = hrmmd_func.apply
        self.with_noise = with_noise
        self.lmbda = lmbda

    def forward(self, x, y):
        out = self.student(x)
        self.student.set_noisy_mode(self.with_noise)
        noisy_out = self.student(x)
        loss = self.hrmmd(y, out, noisy_out, self.lmbda)
        return loss
    
class MMD(nn.Module):
    def __init__(self,student,with_noise):
        super(MMD, self).__init__()
        self.student = student
        self.mmd2 = mmd2_noise_injection.apply
        self.with_noise=with_noise
    def forward(self,x,y):
        if self.with_noise:
            out = tr.mean(self.student(x),dim = -1).clone().detach()
            self.student.set_noisy_mode(True)
            noisy_out = self.student(x)
            loss = 0.5*self.mmd2(y,out,noisy_out)
        else:
            out = tr.mean(self.student(x),dim = -1).clone().detach()
            self.student.set_noisy_mode(False)
            noisy_out = self.student(x)
            loss = 0.5*self.mmd2(y,out,noisy_out)
        return loss


class MMD_Diffusion(nn.Module):
    def __init__(self,student):
        super(MMD_Diffusion, self).__init__()
        self.student = student
        self.mmd2 = mmd2_func.apply
    def forward(self,x,y):
        self.student.add_noise()
        noisy_out = self.student(x)
        
        loss = 0.5*self.mmd2(y,noisy_out)
        return loss


class Sobolev(nn.Module):
    def __init__(self,student):
        super(Sobolev, self).__init__()
        self.student = student
        self.sobolev = sobolev.apply
        self.lmbda = 1e-6
    def forward(self,x,y):
        self.student.zero_grad()
        out = self.student(x)
        b_size,_,num_particles = out.shape
        grad_out = compute_grad(self.student,x)
        matrix = (1./(num_particles*b_size))*tr.einsum('im,jm->ij',grad_out,grad_out)+self.lmbda*tr.eye(b_size, dtype= x.dtype, device=x.device)
        matrix = matrix.clone().detach()
        loss = self.sobolev(y,out,matrix)
        return loss


def compute_grad(net,x):
    J = []
    F = net(x)
    F = tr.einsum('idm->i',F)
    b_size = F.shape[0]
    for i in range(b_size):
        if i==b_size-1:
            grads =  autograd.grad(F[i], net.parameters(),retain_graph=False)
        else:
            grads =  autograd.grad(F[i], net.parameters(),retain_graph=True)
        grads = [x.view(-1) for x in grads]
        grads = tr.cat(grads)
        J.append(grads)

    return tr.stack(J,dim=0)
