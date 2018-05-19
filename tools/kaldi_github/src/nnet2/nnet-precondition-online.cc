// nnet2/nnet-precondition-online.cc

// Copyright 2013-2015   Johns Hopkins University (author: Daniel Povey)
//                2015   Xiaohui Zhang

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet2/nnet-precondition-online.h"

namespace kaldi {
namespace nnet2 {


OnlinePreconditioner::OnlinePreconditioner():
    rank_(40), update_period_(1), num_samples_history_(2000.0), alpha_(4.0),
    epsilon_(1.0e-10), delta_(5.0e-04), t_(-1),
    num_updates_skipped_(0), self_debug_(false) { }


/**
  This function creates a matrix with orthonormal rows that is like the
  following matrix, except with each row normalized to have unit 2-norm:
  [  1.1 0   1   0   1   0
     0   1.1 0   1   0   1  ]
  The reason why the first element in each row is 1.1 and not 1, is for
  symmetry-breaking... we don't want any weighted sum of all these rows to be
  all ones, because the derivative in that direction can be zero in some
  architectures and it causes us to have to do an inefficient CPU-based
  renormalization.
 */
// static
void OnlinePreconditioner::InitOrthonormalSpecial(CuMatrixBase<BaseFloat> *R) {
  int32 num_rows = R->NumRows(), num_cols = R->NumCols();
  KALDI_ASSERT(num_cols >= num_rows);
  R->SetZero();
  std::vector<MatrixElement<BaseFloat> > elems;
  elems.reserve(num_cols);
  BaseFloat first_elem = 1.1;
  for (int32 r = 0; r < num_rows; r++) {
    std::vector<int32> cols;  // columns that have an entry for this row
    for (int32 c = r; c < num_cols; c += num_rows)
      cols.push_back(c);
    BaseFloat normalizer = 1.0 / sqrt(first_elem * first_elem +
                                      cols.size() - 1);
    for (size_t i = 0; i < cols.size(); i++) {
      int32 c = cols[i];
      MatrixElement<BaseFloat> e = { r, c,
                                     normalizer * (i == 0 ? first_elem :
                                                   BaseFloat(1.0)) };
      elems.push_back(e);
    }
  }
  R->AddElements(1.0, elems);
  { // TODO: remove this testing code.
    CuMatrix<BaseFloat> prod(num_rows, num_rows);
    prod.AddMatMat(1.0, *R, kNoTrans, *R, kTrans, 0.0);
    KALDI_ASSERT(prod.IsUnit());
  }
}


void OnlinePreconditioner::InitDefault(int32 D) {
  if (rank_ >= D) {
    KALDI_WARN << "Rank " << rank_ << " of online preconditioner is >= dim " << D
               << ", setting it to "
               << (D - 1) << " (but this is probably still too high)";
    rank_ = D - 1;
  }
  if (rank_ == 0) {
    // Dimension of input data was 1, so the natural gradient preconditioner
    // would always be the unit matrix.
    // We'll handle this as a special case, for generality.
    return;
  }
  KALDI_ASSERT(num_samples_history_ > 0.0 && num_samples_history_ <= 1.0e+6);
  KALDI_ASSERT(alpha_ >= 0.0);
  KALDI_ASSERT(rank_ > 0);
  KALDI_ASSERT(epsilon_ > 0.0 && epsilon_ <= 1.0e-05);  // plausible values.
  KALDI_ASSERT(delta_ > 0.0 && delta_ <= 1.0e-02);  // plausible values.

  // to initialize, in the equation
  //   F_t =(def) R_t^T D_t R_t + \rho_t I
  // we will set the orthogonal R_t to a special orthogonal matrix with no zero
  // rows or columns (see the function), rho_t to epsilon,
  // and D_t to epsilon.  But we don't store R_t directly.  Instead, we store
  //   W_t =(def)  E_t^{0.5} R_t,
  // where E_t =(def)  1/\beta_t (D_t^{-1} + 1/\beta_t I)^{-1}
  // from (eqn:tii),
  //  e_{tii} =   1/(\beta_t/d_{tii} + 1),
  // where
  // \beta_t =(def) \rho_t + \alpha/D tr(F_t)
  //         =      epsilon + alpha/D * (epsilon * D + epsilon * rank)
  //         =     epsilon * (1 + alpha * (D + rank) / D)
  // And  d_{tii} is epsilon, so
  //  e_{tii} =   1/((1 + alpha * (D + rank) / D) + 1)  [for each i.]
  //          =   1/(2 + alpha * (D + rank) / D)).
  BaseFloat epsilon = epsilon_;  // we could make this a bit more.
  rho_t_ = epsilon;
  d_t_.Resize(rank_, kUndefined);
  d_t_.Set(epsilon);
  W_t_.Resize(rank_, D, kUndefined);
  // after the next line, W_ will store the orthogonal matrix R_t.
  InitOrthonormalSpecial(&W_t_);
  BaseFloat E_tii = 1.0 / ( 2.0 + (D + rank_) * alpha_ / D );
  // W_t =(def) E_t^{0.5} R_t.
  W_t_.Scale(sqrt(E_tii));
  t_ = 0;
}

void OnlinePreconditioner::Init(const CuMatrixBase<BaseFloat> &R0) {
  int32 D = R0.NumCols();
  // for locking reasons it's better to use a different object.
  OnlinePreconditioner this_copy(*this);
  this_copy.InitDefault(D);

  CuMatrix<BaseFloat> R0_copy(R0.NumRows(), R0.NumCols(), kUndefined);
  // number of iterations with the same data from a pseudorandom start.
  // this is a faster way of starting than doing eigenvalue decomposition.
  int32 num_init_iters = 3;
  for (int32 i = 0; i < num_init_iters; i++) {
    BaseFloat scale;
    R0_copy.CopyFromMat(R0);
    this_copy.PreconditionDirections(&R0_copy, NULL, &scale);
  }
  rank_ = this_copy.rank_;
  W_t_.Swap(&this_copy.W_t_);
  d_t_.Swap(&this_copy.d_t_);
  rho_t_ = this_copy.rho_t_;
  t_ = 0;
}

void OnlinePreconditioner::PreconditionDirections(
    CuMatrixBase<BaseFloat> *X_t,
    CuVectorBase<BaseFloat> *row_prod,
    BaseFloat *scale) {
  if (X_t->NumCols() == 1) {
    // If the dimension of the space equals one then our natural gradient update
    // with rescaling becomes a no-op, but the code wouldn't naturally handle it
    // because rank would be zero.  Support this as a special case.
    if (row_prod)
      row_prod->AddDiagMat2(1.0, *X_t, kNoTrans, 0.0);
    *scale = 1.0;
    return;
  }

  if (row_prod == NULL) {
    CuVector<BaseFloat> row_prod_tmp(X_t->NumRows());
    PreconditionDirections(X_t, &row_prod_tmp, scale);
    return;
  }

  read_write_mutex_.lock();
  if (t_ == -1) // not initialized
    Init(*X_t);

  // Now t_ >= 0.
  // We create local copies  of the class variables... this is intended for
  // multi-threaded safety so we can't read them in an inconsistent state,
  // but we don't really waste anything here (a copy of W_t is needed anyway,
  // if we're to update it).
  int32 t = t_, R = W_t_.NumRows(), D = W_t_.NumCols();
  // space for W_t, J_t, K_t, L_t.
  CuMatrix<BaseFloat> WJKL_t(2 * R, D + R);
  WJKL_t.Range(0, R, 0, D).CopyFromMat(W_t_);
  BaseFloat rho_t(rho_t_);
  Vector<BaseFloat> d_t(d_t_);
  read_write_mutex_.unlock();
  PreconditionDirectionsInternal(t, rho_t, d_t, &WJKL_t, X_t, row_prod, scale);
}

void OnlinePreconditioner::ReorthogonalizeXt1(
    const VectorBase<BaseFloat> &d_t1,
    BaseFloat rho_t1,
    CuMatrixBase<BaseFloat> *W_t1,
    CuMatrixBase<BaseFloat> *temp_W,
    CuMatrixBase<BaseFloat> *temp_O) {
  // threshold is a configuration value: a desired threshold on orthogonality,
  // below which we won't reorthogonalize.
  const BaseFloat threshold = 1.0e-03;

  int32 R = W_t1->NumRows(), D = W_t1->NumCols();
  BaseFloat beta_t1 = rho_t1 * (1.0 + alpha_) + alpha_ * d_t1.Sum() / D;
  Vector<BaseFloat> e_t1(R, kUndefined), sqrt_e_t1(R, kUndefined),
      inv_sqrt_e_t1(R, kUndefined);
  ComputeEt(d_t1, beta_t1, &e_t1, &sqrt_e_t1, &inv_sqrt_e_t1);

  temp_O->SymAddMat2(1.0, *W_t1, kNoTrans, 0.0);
  // O_t =  E_t^{-0.5} W_t W_t^T E_t^{-0.5}
  Matrix<BaseFloat> O_mat(*temp_O);
  SpMatrix<BaseFloat> O(O_mat, kTakeLower);
  for (int32 i = 0; i < R; i++) {
    BaseFloat i_factor = inv_sqrt_e_t1(i);
    for (int32 j = 0; j <= i; j++) {
      BaseFloat j_factor = inv_sqrt_e_t1(j);
      O(i, j) *= i_factor * j_factor;
    }
  }
  if (O.IsUnit(threshold)) {
    if (self_debug_) {
      KALDI_WARN << "Not reorthogonalizing since already orthognoal: " << O;
    }
    return;
  }
  TpMatrix<BaseFloat> C(R);
  try {
    C.Cholesky(O);
    C.Invert();  // Now it's C^{-1}.
    if (!(C.Max() < 100.0))
      KALDI_ERR << "Cholesky out of expected range, "
                << "reorthogonalizing with Gram-Schmidt";
  } catch (...) {
    // We do a Gram-Schmidt orthogonalization, which is a bit less efficient but
    // more robust than the method using Cholesky.
    KALDI_WARN << "Cholesky or Invert() failed while re-orthogonalizing R_t. "
               << "Re-orthogonalizing on CPU.";
    Matrix<BaseFloat> cpu_W_t1(*W_t1);
    cpu_W_t1.OrthogonalizeRows();
    W_t1->CopyFromMat(cpu_W_t1);
    // at this point cpu_W_t1 represents R_{t+1}- it has orthonormal
    // rows.  Do: W_{t+1} = E_{t+1}^{0.5} R_{t+1}
    CuVector<BaseFloat> sqrt_e_t1_gpu(sqrt_e_t1);
    W_t1->MulRowsVec(sqrt_e_t1_gpu);
    return;
  }
  // Next, compute (E_t^{0.5} C^{-1} E_t^{-0.5})
  // but it's really t+1, not t.
  for (int32 i = 0; i < R; i++) {
    BaseFloat i_factor = sqrt_e_t1(i);
    for (int32 j = 0; j < i; j++) {
      // skip j == i because i_factor * j_factor == 1 for j == i.
      BaseFloat j_factor = inv_sqrt_e_t1(j);
      C(i, j) *= i_factor * j_factor;
    }
  }
  O_mat.CopyFromTp(C);
  temp_O->CopyFromMat(O_mat);
  temp_W->CopyFromMat(*W_t1);
  W_t1->AddMatMat(1.0, *temp_O, kNoTrans, *temp_W, kNoTrans, 0.0);
}

// makes sure certain invariants are being preserved
void OnlinePreconditioner::SelfTest() const {
  KALDI_ASSERT(rho_t_ >= epsilon_);
  BaseFloat d_t_max = d_t_.Max(), d_t_min = d_t_.Min();
  KALDI_ASSERT(d_t_min >= epsilon_);
  KALDI_ASSERT(d_t_min > 0.9 * delta_ * d_t_max);
  KALDI_ASSERT(rho_t_ > 0.9 * delta_ * d_t_max);

  int32 D = W_t_.NumCols(), R = W_t_.NumRows();
  BaseFloat beta_t = rho_t_ * (1.0 + alpha_) + alpha_ * d_t_.Sum() / D;
  Vector<BaseFloat> e_t(R, kUndefined), sqrt_e_t(R, kUndefined),
      inv_sqrt_e_t(R, kUndefined);
  ComputeEt(d_t_, beta_t, &e_t, &sqrt_e_t, &inv_sqrt_e_t);

  CuSpMatrix<BaseFloat> S(R);
  S.AddMat2(1.0, W_t_, kNoTrans, 0.0);
  SpMatrix<BaseFloat> O(S);
  for (int32 i = 0; i < R; i++) {
    BaseFloat i_factor = inv_sqrt_e_t(i);
    for (int32 j = 0; j <= i; j++) {
      BaseFloat j_factor = inv_sqrt_e_t(j);
      O(i, j) *= i_factor * j_factor;
    }
  }
  if (!O.IsUnit(1.0e-04) || O(0, 0) != O(0, 0)) {
    BaseFloat worst_error = 0.0;
    int32 worst_i = 0, worst_j = 0;
    for (int32 i = 0; i < R; i++) {
      for (int32 j = 0; j < R; j++) {
        BaseFloat elem = O(i, j);
        BaseFloat error = fabs(elem - (i == j ? 1.0 : 0.0));
        if (error > worst_error || error != error) {
          worst_error = error;
          worst_i = i;
          worst_j = j;
        }
      }
    }
    if (worst_error > 1.0e-02 || worst_error != worst_error) {
      KALDI_WARN << "Failed to verify W_t (worst error: O[" << worst_i << ','
                 << worst_j << "] = " << O(worst_i, worst_j)
                 << ", d_t = " << d_t_;
    }
  }
}

void OnlinePreconditioner::PreconditionDirectionsInternal(
    const int32 t,
    const BaseFloat rho_t,
    const Vector<BaseFloat> &d_t,
    CuMatrixBase<BaseFloat> *WJKL_t,
    CuMatrixBase<BaseFloat> *X_t,
    CuVectorBase<BaseFloat> *row_prod,
    BaseFloat *scale) {
  int32 N = X_t->NumRows(),  // Minibatch size.
      D = X_t->NumCols(),  // Dimensions of vectors we're preconditioning
      R = rank_;  // Rank of correction to unit matrix.
  KALDI_ASSERT(R > 0 && R < D);
  BaseFloat eta = Eta(N);

  CuMatrix<BaseFloat> H_t(N, R);
  const CuSubMatrix<BaseFloat> W_t(*WJKL_t, 0, R, 0, D);
  // Below, WJ_t and LK_t are combinations of two matrices,
  // which we define in order to combine two separate multiplications into one.
  CuSubMatrix<BaseFloat> J_t(*WJKL_t, R, R, 0, D),
      L_t(*WJKL_t, 0, R, D, R),
      K_t(*WJKL_t, R, R, D, R),
      WJ_t(*WJKL_t, 0, 2 * R, 0, D),
      LK_t(*WJKL_t, 0, 2 * R, D, R);

  H_t.AddMatMat(1.0, *X_t, kNoTrans, W_t, kTrans, 0.0);  // H_t = X_t W_t^T

  bool locked = update_mutex_.try_lock();
  if (locked) {
    // Just hard-code it here that we do 10 updates before skipping any.
    const int num_initial_updates = 10;
    if (t_ > t || (num_updates_skipped_ < update_period_ - 1 &&
                   t_ >= num_initial_updates)) {
      update_mutex_.unlock();
      // We got the lock but we were already beaten to it by another thread, or
      // we don't want to update yet due to update_period_ > 1 (this saves
      // compute), so release the lock.
      locked = false;
    }
  }

  if (!locked) {
    // We're not updating the parameters, either because another thread is
    // working on updating them, or because another thread already did so from
    // the same or later starting point (making our update stale), or because
    // update_period_ > 1.  We just apply the preconditioning and return.

    // note: we don't bother with any locks before incrementing
    // num_updates_skipped_ below, because the worst that could happen is that,
    // on very rare occasions, we could skip one or two more updates than we
    // intended.
    num_updates_skipped_++;

    BaseFloat tr_Xt_XtT = TraceMatMat(*X_t, *X_t, kTrans);
    // X_hat_t = X_t - H_t W_t
    X_t->AddMatMat(-1.0, H_t, kNoTrans, W_t, kNoTrans, 1.0);
    // each element i of row_prod will be inner product of row i of X_hat_t with
    // itself.
    row_prod->AddDiagMat2(1.0, *X_t, kNoTrans, 0.0);
    BaseFloat tr_Xhat_XhatT = row_prod->Sum();
    KALDI_ASSERT(tr_Xhat_XhatT == tr_Xhat_XhatT);  // Check for NaN.
    BaseFloat gamma_t = (tr_Xhat_XhatT == 0.0 ? 1.0 :
                         sqrt(tr_Xt_XtT / tr_Xhat_XhatT));
    *scale = gamma_t;
    return;
  }
  J_t.AddMatMat(1.0, H_t, kTrans, *X_t, kNoTrans, 0.0);  // J_t = H_t^T X_t

  bool compute_lk_together = (N > D);

  if (compute_lk_together) {
    // do the following two multiplies in one operation...
    // note
    // L_t = W_t J_t^T
    // K_t = J_t J_t^T
    // Note: L_t was defined as L_t = J_t W_t^T, but it's actually symmetric,
    // so we can compute it as L_t = W_t J_t^T.
    LK_t.AddMatMat(1.0, WJ_t, kNoTrans, J_t, kTrans, 0.0);
  } else {
    K_t.SymAddMat2(1.0, J_t, kNoTrans, 0.0);
    L_t.SymAddMat2(1.0, H_t, kTrans, 0.0);
  }

  Matrix<BaseFloat> LK_cpu(LK_t);  // contains L and K on the CPU.
  SubMatrix<BaseFloat> L_t_cpu(LK_cpu, 0, R, 0, R),
      K_t_cpu(LK_cpu, R, R, 0, R);
  if (!compute_lk_together) {
    // the SymAddMat2 operations only set the lower triangle and diagonal.
    L_t_cpu.CopyLowerToUpper();
    K_t_cpu.CopyLowerToUpper();
  }

  // beta_t = \rho_t(1+\alpha) + \alpha/D tr(D_t)
  BaseFloat beta_t = rho_t * (1.0 + alpha_) + alpha_ * d_t.Sum() / D;
  Vector<BaseFloat> e_t(R), sqrt_e_t(R), inv_sqrt_e_t(R);
  ComputeEt(d_t, beta_t, &e_t, &sqrt_e_t, &inv_sqrt_e_t);
  KALDI_VLOG(5) << "e_t = " << e_t;

  // The double-precision Z_t here, and the scaling, is to avoid potential
  // overflow, because Z_t is proportional to the fourth power of data.
  SpMatrix<double> Z_t_double(R);
  ComputeZt(N, rho_t, d_t, inv_sqrt_e_t, K_t_cpu, L_t_cpu, &Z_t_double);
  BaseFloat z_t_scale = std::max<double>(1.0, Z_t_double.Trace());
  Z_t_double.Scale(1.0 / z_t_scale);
  SpMatrix<BaseFloat> Z_t_scaled(Z_t_double);

  Matrix<BaseFloat> U_t(R, R);
  Vector<BaseFloat> c_t(R);
  // do the symmetric eigenvalue decomposition Z_t = U_t C_t U_t^T.
  Z_t_scaled.Eig(&c_t, &U_t);
  SortSvd(&c_t, &U_t);
  c_t.Scale(z_t_scale);

  const BaseFloat condition_threshold = 1.0e+06;
  // must_reorthogonalize will be true if the last diagonal element of c_t is
  // negative, since we don't take the absolute value, but this is the right
  // thing anyway.
  bool must_reorthogonalize = (c_t(0) > condition_threshold * c_t(R - 1));

  BaseFloat c_t_floor = pow(rho_t * (1 - eta), 2);
  int32 nf;
  c_t.ApplyFloor(c_t_floor, &nf);
  if (nf > 0)
    must_reorthogonalize = true;
  if (nf > 0 && self_debug_) {
    KALDI_WARN << "Floored " << nf << " elements of C_t.";
  }
  BaseFloat tr_Xt_XtT_check;
  if (self_debug_)
    tr_Xt_XtT_check = TraceMatMat(*X_t, *X_t, kTrans);

  X_t->AddMatMat(-1.0, H_t, kNoTrans, W_t, kNoTrans, 1.0);  // X_hat_t = X_t - H_t W_t
  // set *row_prod to inner products of each row of X_hat_t with itself.
  row_prod->AddDiagMat2(1.0, *X_t, kNoTrans, 0.0);

  BaseFloat tr_Xhat_XhatT = row_prod->Sum();
  //  tr(X_t X_t^T) = tr(X_hat_t X_hat_t^T) - tr(L_t E_t) + 2 tr(L_t)
  double tr_Xt_XtT = tr_Xhat_XhatT;
  for (int32 i = 0; i < R; i++)
    tr_Xt_XtT += L_t_cpu(i, i) * (2.0 - e_t(i));
  if (self_debug_) {
    KALDI_ASSERT(ApproxEqual(tr_Xt_XtT, tr_Xt_XtT_check));
  }
  BaseFloat gamma_t = (tr_Xhat_XhatT == 0.0 ? 1.0 :
                       sqrt(tr_Xt_XtT / tr_Xhat_XhatT));
  *scale = gamma_t;

  Vector<BaseFloat> sqrt_c_t(c_t);
  sqrt_c_t.ApplyPow(0.5);

  // \rho_{t+1} = 1/(D - R) (\eta/N tr(X_t X_t^T) + (1-\eta)(D \rho_t + tr(D_t)) - tr(C_t^{0.5})).
  BaseFloat rho_t1 = 1.0 / (D - R) * (eta / N * tr_Xt_XtT
                                      + (1-eta)*(D * rho_t + d_t.Sum())
                                      - sqrt_c_t.Sum());
  // D_{t+1} = C_t^{0.5} - \rho_{t+1} I
  Vector<BaseFloat> d_t1(sqrt_c_t);
  d_t1.Add(-rho_t1);
  BaseFloat floor_val = std::max(epsilon_, delta_ * sqrt_c_t.Max());
  if (rho_t1 < floor_val)
    rho_t1 = floor_val;
  d_t1.ApplyFloor(floor_val);

  CuMatrix<BaseFloat> W_t1(R, D);  // W_{t+1}
  ComputeWt1(N, d_t, d_t1, rho_t, rho_t1, U_t, sqrt_c_t, inv_sqrt_e_t,
             W_t, &J_t, &W_t1);

  if (must_reorthogonalize) {
    if (self_debug_) {
      KALDI_WARN << "Reorthogonalizing.";
    }
    ReorthogonalizeXt1(d_t1,
                       rho_t1,
                       &W_t1,
                       &J_t,
                       &L_t);
  }

  // Commit the new parameters.
  read_write_mutex_.lock();
  KALDI_ASSERT(t_ == t);  // we already ensured this.
  t_ = t + 1;
  num_updates_skipped_ = 0;
  W_t_.Swap(&W_t1);
  d_t_.CopyFromVec(d_t1);
  rho_t_ = rho_t1;

  if (self_debug_)
    SelfTest();

  read_write_mutex_.unlock();
  update_mutex_.unlock();
}

BaseFloat OnlinePreconditioner::Eta(int32 N) const {
  KALDI_ASSERT(num_samples_history_ > 0.0);
  BaseFloat ans = 1.0 - exp(-N / num_samples_history_);
  // Don't let eta approach 1 too closely, as it can lead to NaN's appearing if
  // the input is all zero.
  if (ans > 0.9) ans = 0.9;
  return ans;
}

void OnlinePreconditioner::ComputeWt1(int32 N,
                                       const VectorBase<BaseFloat> &d_t,
                                       const VectorBase<BaseFloat> &d_t1,
                                       BaseFloat rho_t,
                                       BaseFloat rho_t1,
                                       const MatrixBase<BaseFloat> &U_t,
                                       const VectorBase<BaseFloat> &sqrt_c_t,
                                       const VectorBase<BaseFloat> &inv_sqrt_e_t,
                                       const CuMatrixBase<BaseFloat> &W_t,
                                       CuMatrixBase<BaseFloat> *J_t,
                                       CuMatrixBase<BaseFloat> *W_t1) const {

  int32 R = d_t.Dim(), D = W_t.NumCols();
  BaseFloat eta = Eta(N);

  // \beta_{t+1} = \rho_{t+1} (1+\alpha) + \alpha/D tr(D_{t+1})
  BaseFloat beta_t1 = rho_t1 * (1.0 + alpha_) + alpha_ * d_t1.Sum() / D;
  KALDI_ASSERT(beta_t1 > 0.0);
  Vector<BaseFloat> e_t1(R, kUndefined), sqrt_e_t1(R, kUndefined),
      inv_sqrt_e_t1(R, kUndefined);
  ComputeEt(d_t1, beta_t1, &e_t1, &sqrt_e_t1, &inv_sqrt_e_t1);
  Vector<BaseFloat> inv_sqrt_c_t(sqrt_c_t);
  inv_sqrt_c_t.InvertElements();

  Vector<BaseFloat> w_t_coeff(R);
  for (int32 i = 0; i < R; i++)
    w_t_coeff(i) = (1.0 - eta) / (eta/N) * (d_t(i) + rho_t);
  CuVector<BaseFloat> w_t_coeff_gpu(w_t_coeff);
  // B_t = J_t + (1-\eta)/(\eta/N) (D_t + \rho_t I) W_t
  J_t->AddDiagVecMat(1.0, w_t_coeff_gpu, W_t, kNoTrans, 1.0);

  // A_t = (\eta/N) E_{t+1}^{0.5} C_t^{-0.5} U_t^T E_t^{-0.5} B_t
  Matrix<BaseFloat> A_t(U_t, kTrans);
  for (int32 i = 0; i < R; i++) {
    BaseFloat i_factor = (eta / N) * sqrt_e_t1(i) * inv_sqrt_c_t(i);
    for (int32 j = 0; j < R; j++) {
      BaseFloat j_factor = inv_sqrt_e_t(j);
      A_t(i, j) *= i_factor * j_factor;
    }
  }
  // W_{t+1} = A_t B_t
  CuMatrix<BaseFloat> A_t_gpu(A_t);
  W_t1->AddMatMat(1.0, A_t_gpu, kNoTrans, *J_t, kNoTrans, 0.0);
}

void OnlinePreconditioner::ComputeZt(int32 N,
                                     BaseFloat rho_t,
                                     const VectorBase<BaseFloat> &d_t,
                                     const VectorBase<BaseFloat> &inv_sqrt_e_t,
                                     const MatrixBase<BaseFloat> &K_t,
                                     const MatrixBase<BaseFloat> &L_t,
                                     SpMatrix<double> *Z_t) const {
  // Use doubles because the range of quantities in Z_t can get large (fourth
  // power of data), and we want to avoid overflow.  This routine is fast.
  BaseFloat eta = Eta(N);
  Vector<BaseFloat> d_t_rho_t(d_t);
  d_t_rho_t.Add(rho_t);  // now d_t_rho_t is diag(D_t + \rho_t I).
  double etaN = eta / N, eta1 = 1.0 - eta,
      etaN_sq = etaN * etaN, eta1_sq = eta1 * eta1,
      etaN_eta1 = etaN * eta1;
  int32 R = d_t.Dim();
  for (int32 i = 0; i < R; i++) {
    double inv_sqrt_e_t_i = inv_sqrt_e_t(i), d_t_rho_t_i = d_t_rho_t(i);
    for (int32 j = 0; j <= i; j++) {
      double inv_sqrt_e_t_j = inv_sqrt_e_t(j), d_t_rho_t_j = d_t_rho_t(j),
          L_t_i_j = 0.5 * (L_t(i, j) + L_t(j, i)),
          K_t_i_j = 0.5 * (K_t(i, j) + K_t(j, i));
      // See (eqn:Zt) in header.
      (*Z_t)(i, j) = etaN_sq * inv_sqrt_e_t_i * K_t_i_j * inv_sqrt_e_t_j
          + etaN_eta1 * inv_sqrt_e_t_i * L_t_i_j * inv_sqrt_e_t_j * d_t_rho_t_j
          + etaN_eta1 * d_t_rho_t_i * inv_sqrt_e_t_i * L_t_i_j * inv_sqrt_e_t_j
          + (i == j ? eta1_sq * d_t_rho_t_i * d_t_rho_t_i : 0.0);
    }
  }
}

void OnlinePreconditioner::ComputeEt(const VectorBase<BaseFloat> &d_t,
                                     BaseFloat beta_t,
                                     VectorBase<BaseFloat> *e_t,
                                     VectorBase<BaseFloat> *sqrt_e_t,
                                     VectorBase<BaseFloat> *inv_sqrt_e_t) const {
  // e_{tii} = 1/(\beta_t/d_{tii} + 1)
  int32 D = d_t.Dim();
  const BaseFloat *d = d_t.Data();
  BaseFloat *e = e_t->Data();
  for (int32 i = 0; i < D; i++)
    e[i] = 1.0 / (beta_t / d[i]  +  1);
  sqrt_e_t->CopyFromVec(*e_t);
  sqrt_e_t->ApplyPow(0.5);
  inv_sqrt_e_t->CopyFromVec(*sqrt_e_t);
  inv_sqrt_e_t->InvertElements();
}


OnlinePreconditioner::OnlinePreconditioner(const OnlinePreconditioner &other):
    rank_(other.rank_), update_period_(other.update_period_),
    num_samples_history_(other.num_samples_history_),
    alpha_(other.alpha_), epsilon_(other.epsilon_), delta_(other.delta_),
    t_(other.t_), num_updates_skipped_(other.num_updates_skipped_),
    self_debug_(other.self_debug_), W_t_(other.W_t_),
    rho_t_(other.rho_t_), d_t_(other.d_t_) {
  // use default constructor for the mutexes.
}

OnlinePreconditioner& OnlinePreconditioner::operator = (
    const OnlinePreconditioner &other) {
  rank_ = other.rank_;
  update_period_ = other.update_period_;
  num_samples_history_ = other.num_samples_history_;
  alpha_ = other.alpha_;
  epsilon_ = other.epsilon_;
  delta_ = other.delta_;
  t_ = other.t_;
  self_debug_ = other.self_debug_;
  W_t_ = other.W_t_;
  rho_t_ = other.rho_t_;
  d_t_ = other.d_t_;
  return *this;
}

void OnlinePreconditioner::SetRank(int32 rank) {
  KALDI_ASSERT(rank > 0);
  rank_ = rank;
}
void OnlinePreconditioner::SetUpdatePeriod(int32 update_period) {
  KALDI_ASSERT(update_period > 0);
  update_period_ = update_period;
}
void OnlinePreconditioner::SetNumSamplesHistory(BaseFloat num_samples_history) {
  KALDI_ASSERT(num_samples_history > 0.0 &&
               num_samples_history < 1.0e+6);
  num_samples_history_ = num_samples_history;
}
void OnlinePreconditioner::SetAlpha(BaseFloat alpha) {
  KALDI_ASSERT(alpha >= 0.0);
  alpha_ = alpha;
}


}
}
