// nnet/nnet-affine-transform.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_AFFINE_TRANSFORM_H_
#define KALDI_NNET_NNET_AFFINE_TRANSFORM_H_
/*
#include <sstream>
#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"
*/
#include "net/layer.h"
#include "net/trainable-layer.h"
#include "net/utils-functions.h"
#include "gpucompute/cuda-math.h"
#include "net/nnet-precondition.h"
#include "net/nnet-precondition-online.h"

namespace eesen {

class AffineTransform : public TrainableLayer {
 public:
  AffineTransform(int32 dim_in, int32 dim_out) 
    : TrainableLayer(dim_in, dim_out), 
      linearity_(dim_out, dim_in), bias_(dim_out),
      linearity_corr_(dim_out, dim_in), bias_corr_(dim_out),
      linearity_grad_(dim_out, dim_in), bias_grad_(dim_out),
      linearity_grad_square_(dim_out, dim_in), bias_grad_square_(dim_out),
      linearity_ada_(dim_out, dim_in), bias_ada_(dim_out),
      learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0), max_norm_(0.0) 
  { }
  ~AffineTransform()
  { }

  Layer* Copy() const { return new AffineTransform(*this); }
  LayerType GetType() const { return l_Affine_Transform; }
	LayerType GetTypeNonParal() const { return l_Affine_Transform; } 

	void Scale(BaseFloat scale) {
    linearity_.Scale(scale);
    bias_.Scale(scale);
  }

  void Add(BaseFloat scale, const TrainableLayer & layer_other) {
    const AffineTransform *other = dynamic_cast<const AffineTransform*>(&layer_other);
    linearity_.AddMat(scale, other->linearity_);
    bias_.AddVec(scale, other->bias_);
  }

 
  void InitData(std::istream &is) {
    // define options
    float param_range = 0.02;
    float bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1;
    float learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
    float max_norm = 0.0;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
			/**/ if (token == "<ParamRange>") ReadBasicType(is, false, &param_range);
      else if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
      else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef);
      else if (token == "<MaxNorm>") ReadBasicType(is, false, &max_norm);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws; // eat-up whitespace
    }

    //
    // initialize
    //
    Matrix<BaseFloat> mat(output_dim_, input_dim_);
    for (int32 r=0; r<output_dim_; r++) {
      for (int32 c=0; c<input_dim_; c++) {
        mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
      }
    }
    linearity_ = mat;
    //
    Vector<BaseFloat> vec(output_dim_);
    for (int32 i=0; i<output_dim_; i++) {
      // +/- 1/2*bias_range from bias_mean:
      vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range; 
    }
    bias_ = vec;
    //
    learn_rate_coef_ = learn_rate_coef;
    bias_learn_rate_coef_ = bias_learn_rate_coef;
    max_norm_ = max_norm;
    //
  }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coefs
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
			// TODO clean this
      ExpectToken(is, binary, "<BiasLearnRateCoef>");
      ReadBasicType(is, binary, &bias_learn_rate_coef_);
    }
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<MaxNorm>");
      ReadBasicType(is, binary, &max_norm_);
    }
    // weights
    linearity_.Read(is, binary);
    bias_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<MaxNorm>");
    WriteBasicType(os, binary, max_norm_);
    // weights
    linearity_.Write(os, binary);
    bias_.Write(os, binary);
  }

  void SetMaxNorm(BaseFloat max_norm) { max_norm_ = max_norm; }

  int32 NumParams() const { return linearity_.NumRows()*linearity_.NumCols() + bias_.Dim(); }
  int32 NumElements(std::string content) const {
    KALDI_ASSERT(content == "model" || content == "momentum" || content == "all" || content == "gradient");
    if (content == "gradient")
      return linearity_grad_.NumRows()*linearity_grad_.Stride() + bias_grad_.Dim();
    if (content == "model")
      return linearity_.NumRows()*linearity_.Stride() + bias_.Dim(); 
    if (content == "momentum")
      return linearity_corr_.NumRows()*linearity_corr_.Stride() + bias_corr_.Dim();
    if (content == "all")
      return linearity_.NumRows()*linearity_.Stride() + bias_.Dim() + linearity_corr_.NumRows()*linearity_corr_.Stride() + bias_corr_.Dim();
    return 0;
  }
  
  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 linearity_num_elem = linearity_.NumRows() * linearity_.NumCols(); 
    wei_copy->Range(0,linearity_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(linearity_));
    wei_copy->Range(linearity_num_elem, bias_.Dim()).CopyFromVec(Vector<BaseFloat>(bias_));
  }
  
  void GetElements(BaseFloat* wei_copy, const std::string content) {
    KALDI_ASSERT(content == "model" || content == "momentum" || content == "all" || content == "gradient");
    int32 offset = 0;
    if (content == "gradient") {
      linearity_ada_.CopyToArray(&wei_copy[offset]);    // accumulated gradient, we are using ada buffer temporarily
      offset += linearity_ada_.NumRows() * linearity_ada_.Stride();
      bias_ada_.CopyToArray(&wei_copy[offset]);
      offset += bias_ada_.Dim();
      linearity_ada_.SetZero();
      bias_ada_.SetZero();
    }
    if (content == "model" || content == "all") {
      linearity_.CopyToArray(&wei_copy[offset]);
      offset += linearity_.NumRows() * linearity_.Stride();
      bias_.CopyToArray(&wei_copy[offset]);
      offset += bias_.Dim();
    }
    if (content == "momentum" || content == "all") {
      linearity_corr_.CopyToArray(&wei_copy[offset]);
      offset += linearity_corr_.NumRows() * linearity_corr_.Stride();
      bias_corr_.CopyToArray(&wei_copy[offset]);
      offset += bias_corr_.Dim();
    }
  }

  void AverageElements(const BaseFloat alpha, const BaseFloat* v, const BaseFloat beta, const std::string content) {
    KALDI_ASSERT(content == "model" || content == "momentum" || content == "all" || content == "gradient");
    int32 offset = 0;
    if (content == "gradient") {
      linearity_grad_.AverageArray(alpha, &v[offset], beta);
      offset += linearity_grad_.NumRows() * linearity_grad_.Stride();
      bias_grad_.AverageArray(alpha, &v[offset], beta);
      offset += bias_grad_.Dim();
    }
    if (content == "model" || content == "all") {
      linearity_.AverageArray(alpha, &v[offset], beta);
      offset += linearity_.NumRows() * linearity_.Stride();
      bias_.AverageArray(alpha, &v[offset], beta);
      offset += bias_.Dim();
    }
    if (content == "momentum" || content == "all") {
      linearity_corr_.AverageArray(alpha, &v[offset], beta);
      offset += linearity_corr_.NumRows() * linearity_corr_.Stride();
      bias_corr_.AverageArray(alpha, &v[offset], beta);
      offset += bias_corr_.Dim();
    }
  }

  void BufferUpdate(const BaseFloat* v, const std::string content) {
    KALDI_ASSERT(content == "gradient");
    int32 offset = 0;
    linearity_grad_.CopyFromArray(v);
    offset += linearity_grad_.NumRows() * linearity_grad_.Stride();
    bias_grad_.CopyFromArray(&v[offset]);
    offset += bias_grad_.Dim();

    Update();
  }
  
  std::string Info() const {
    std::string ref_str = "";
    return std::string("\n  linearity") + MomentStatistics(linearity_) +
           "\n  bias" + MomentStatistics(bias_) + ref_str;
  }
  std::string InfoGradient() const {
    return std::string("\n  linearity_grad") + MomentStatistics(linearity_corr_) + 
           ", lr-coef " + ToString(learn_rate_coef_) +
           ", max-norm " + ToString(max_norm_) +
           "\n  bias_grad" + MomentStatistics(bias_corr_) + 
           ", lr-coef " + ToString(bias_learn_rate_coef_);
           
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
  }
  
  void PropagateFnc(const std::vector<std::vector<CuMatrix<BaseFloat> > > &in,CuMatrixBase<BaseFloat> *out) {
    KALDI_ERR << __func__ << "Not implemented!";
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by weights
    in_diff->AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);
  }
  
  void BackpropagateFnc(const std::vector<std::vector<CuMatrix<BaseFloat> > > &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, std::vector<std::vector<CuMatrix<BaseFloat> > > &in_diff) {
    KALDI_ERR << __func__ << "Not implemented!";
  }

  // Sometimes we don't want to compute grad because this might be copy from another nnet, in this scenario,
  // we will call UpdateComponent directly.
  // Update will compute the gradient and call UpdateComponent()
  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // compute gradient
    linearity_grad_.AddMatMat(1.0, diff, kTrans, input, kNoTrans, 0.0);
    bias_grad_.AddRowSumMat(1.0, diff, 0.0);

    linearity_ada_.AddMat(1.0, linearity_grad_, kNoTrans);
    bias_ada_.AddVec(1.0, bias_grad_);

    num_frames_ = input.NumRows();
    Update();
  }

  void Update()  {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum < 0 ? 0 - opts_.momentum : opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch

    if (opts_.momentum < 0) {
      // use this for adagrad temporarily
      linearity_grad_square_.CopyFromMat(linearity_grad_);
      bias_grad_square_.CopyFromVec(bias_grad_);
      linearity_grad_square_.MulElements(linearity_grad_square_);
      bias_grad_square_.MulElements(bias_grad_square_);
      
      linearity_ada_.AddMat(1.0, linearity_grad_square_, kNoTrans);
      bias_ada_.AddVec(1.0, bias_grad_square_);

      double linearity_mean_scale = linearity_ada_.Sum() / (linearity_ada_.NumCols() * linearity_ada_.NumRows());
      double bias_mean_scale = bias_ada_.Sum() / bias_ada_.Dim();

      linearity_grad_square_.CopyFromMat(linearity_ada_);
      bias_grad_square_.CopyFromVec(bias_ada_);
      linearity_grad_square_.Scale(1.0/linearity_mean_scale);
      bias_grad_square_.Scale(1.0/bias_mean_scale);

      linearity_grad_square_.InvertElements();
      linearity_grad_.MulElements(linearity_grad_square_);
      bias_grad_.InvertElements();
      bias_grad_.MulElements(bias_grad_square_);
    }
    
    // include momentum
    linearity_corr_.Scale(mmt);
    linearity_corr_.AddMat(1.0 - mmt, linearity_grad_, kNoTrans);
    bias_corr_.Scale(mmt);
    bias_corr_.AddVec(1.0 - mmt, bias_grad_);

    // l2 regularization
    if (l2 != 0.0) {
      linearity_.AddMat(-lr*l2*num_frames_, linearity_);
      if (ref_layer_ != NULL) {
        const AffineTransform* af_component = dynamic_cast<const AffineTransform*> (ref_layer_);
        linearity_.AddMat(lr*l2*num_frames_, af_component->GetLinearity());
      }
    }
    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames_, lr);
    }
    // update
    linearity_.AddMat(-lr, linearity_corr_);
    bias_.AddVec(-lr_bias, bias_corr_);
    // max-norm
    if (max_norm_ > 0.0) {
      CuMatrix<BaseFloat> lin_sqr(linearity_);
      lin_sqr.MulElements(linearity_);
      CuVector<BaseFloat> l2(OutputDim());
      l2.AddColSumMat(1.0, lin_sqr, 0.0);
      l2.ApplyPow(0.5); // we have per-neuron L2 norms
      CuVector<BaseFloat> scl(l2);
      scl.Scale(1.0/max_norm_);
      scl.ApplyFloor(1.0);
      scl.InvertElements();
      linearity_.MulRowsVec(scl); // shink to sphere!
    }
  }

  /// Accessors to the component parameters
  const CuVectorBase<BaseFloat>& GetBias() const {
    return bias_;
  }

  void SetBias(const CuVectorBase<BaseFloat>& bias) {
    KALDI_ASSERT(bias.Dim() == bias_.Dim());
    bias_.CopyFromVec(bias);
  }

  const CuMatrixBase<BaseFloat>& GetLinearity() const {
    return linearity_;
  }

  void SetLinearity(const CuMatrixBase<BaseFloat>& linearity) {
    KALDI_ASSERT(linearity.NumRows() == linearity_.NumRows());
    KALDI_ASSERT(linearity.NumCols() == linearity_.NumCols());
    linearity_.CopyFromMat(linearity);
  }

  const CuVectorBase<BaseFloat>& GetBiasCorr() const {
    return bias_corr_;
  }

  const CuMatrixBase<BaseFloat>& GetLinearityCorr() const {
    return linearity_corr_;
  }

  const BaseFloat GetLearnRate() const { return learn_rate_coef_; }
  const BaseFloat GetBiasLearnRate() const { return bias_learn_rate_coef_; }
  const int32 GetNumFrames() const {return num_frames_;}

 protected:
  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  CuMatrix<BaseFloat> linearity_grad_;
  CuVector<BaseFloat> bias_grad_;

  CuMatrix<BaseFloat> linearity_grad_square_;
  CuVector<BaseFloat> bias_grad_square_;

  CuMatrix<BaseFloat> linearity_ada_;
  CuVector<BaseFloat> bias_ada_;

  BaseFloat learn_rate_coef_;
  BaseFloat bias_learn_rate_coef_;
  BaseFloat max_norm_;   // If > 0, this is the maximum amount of parameter change (in L2 norm)
                         // that we allow per minibatch.  This was introduced in order to
                         // control instability.  Instead of the exact L2 parameter change,
                         // for efficiency purposes we limit a bound on the exact change.
                         // The limit is applied via a constant <= 1.0 for each minibatch,
                         // A suitable value might be, for example, 10 or so; larger if there are
                         // more parameters.

  int32 num_frames_;
};

// This is an idea Dan is trying out, a little bit like
// preconditioning the update with the Fisher matrix, but the
// Fisher matrix has a special structure.
// [note: it is currently used in the standard recipe].
class AffineTransformPreconditioned: public AffineTransform {
 public:
  AffineTransformPreconditioned(int32 dim_in, int32 dim_out): AffineTransform(dim_in, dim_out), alpha_(0.1) {}
  
  void CopyAffineTransform(const AffineTransform& trans_component, double max_norm, double alpha) {
    linearity_ = trans_component.GetLinearity();
    bias_ = trans_component.GetBias();
    learn_rate_coef_ = trans_component.GetLearnRate();
    bias_learn_rate_coef_ = GetBiasLearnRate();
    num_frames_ = GetNumFrames();
    alpha_ = alpha;
    max_norm_ = max_norm;
  }
  
  LayerType GetType() const { return l_Affine_Transform_Precond; }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coefs
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
      ExpectToken(is, binary, "<BiasLearnRateCoef>");
      ReadBasicType(is, binary, &bias_learn_rate_coef_);
    }
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<MaxNorm>");
      ReadBasicType(is, binary, &max_norm_);
    }
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<Alpha>");
      ReadBasicType(is, binary, &alpha_);
    }
    // weights
    linearity_.Read(is, binary);
    bias_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);
  }
  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<MaxNorm>");
    WriteBasicType(os, binary, max_norm_);
    WriteToken(os, binary, "<Alpha>");
    WriteBasicType(os, binary, alpha_);
    // weights
    linearity_.Write(os, binary);
    bias_.Write(os, binary);

  }
//  void Init(BaseFloat learning_rate,
//            int32 input_dim, int32 output_dim,
//            BaseFloat param_stddev, BaseFloat bias_stddev,
//            BaseFloat alpha, BaseFloat max_change);
//  void Init(BaseFloat learning_rate, BaseFloat alpha,
//            BaseFloat max_change, std::string matrix_filename);
  
  void InitData(std::istream &is) {
    // define options
    float param_range = 0.02;
    float bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1;
    float learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
    float max_norm = 0.0;
    float alpha = 0.1;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
			/**/ if (token == "<ParamRange>") ReadBasicType(is, false, &param_range);
      else if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
      else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef);
      else if (token == "<MaxNorm>") ReadBasicType(is, false, &max_norm);
      else if (token == "<Alpha>") ReadBasicType(is, false, &alpha);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws; // eat-up whitespace
    }

    //
    // initialize
    //
    Matrix<BaseFloat> mat(output_dim_, input_dim_);
    for (int32 r=0; r<output_dim_; r++) {
      for (int32 c=0; c<input_dim_; c++) {
        mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
      }
    }
    linearity_ = mat;
    //
    Vector<BaseFloat> vec(output_dim_);
    for (int32 i=0; i<output_dim_; i++) {
      // +/- 1/2*bias_range from bias_mean:
      vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range; 
    }
    bias_ = vec;
    //
    learn_rate_coef_ = learn_rate_coef;
    bias_learn_rate_coef_ = bias_learn_rate_coef;
    max_norm_ = max_norm;
    alpha_ = alpha;
  }

  std::string Info() const {
		
    std::string ref_str = "";
    std::ostringstream ostr;
    ostr << "\n  alpha " << alpha_ << "  max_norm " << max_norm_ << "  learn_rate_coef " << learn_rate_coef_
         << " bias_learn_rate_coef_ " << bias_learn_rate_coef_;
    return std::string("\n  linearity") + MomentStatistics(linearity_) +
           "\n  bias" + MomentStatistics(bias_) + ostr.str() + ref_str;

  }
  std::string InfoGradient() const {
    return std::string("\n  linearity_grad") + MomentStatistics(linearity_corr_) + 
           ", lr-coef " + ToString(learn_rate_coef_) +
           ", max-norm " + ToString(max_norm_) +
           "\n  bias_grad" + MomentStatistics(bias_corr_) + 
           ", lr-coef " + ToString(bias_learn_rate_coef_) +
           ", alpha " + ToString(alpha_);
  }
 

 protected:
  KALDI_DISALLOW_COPY_AND_ASSIGN(AffineTransformPreconditioned);
  BaseFloat alpha_;
    /// The following function is only called if max_change_ > 0.  It returns the
  /// greatest value alpha <= 1.0 such that (alpha times the sum over the
  /// row-index of the two matrices of the product the l2 norms of the two rows
  /// times learning_rate_)
  /// is <= max_change.

  BaseFloat GetScalingFactor(const CuMatrix<BaseFloat> &in_value_precon,
                             const CuMatrix<BaseFloat> &out_deriv_precon) {
    static int scaling_factor_printed = 0;

    KALDI_ASSERT(in_value_precon.NumRows() == out_deriv_precon.NumRows());
    CuVector<BaseFloat> in_norm(in_value_precon.NumRows()),
        out_deriv_norm(in_value_precon.NumRows());
    in_norm.AddDiagMat2(1.0, in_value_precon, kNoTrans, 0.0);
    out_deriv_norm.AddDiagMat2(1.0, out_deriv_precon, kNoTrans, 0.0);
    // Get the actual l2 norms, not the squared l2 norm.
    in_norm.ApplyPow(0.5);
    out_deriv_norm.ApplyPow(0.5);
    BaseFloat sum = learn_rate_coef_ * VecVec(in_norm, out_deriv_norm);
    // sum is the product of norms that we are trying to limit
    // to max_value_.
    KALDI_ASSERT(sum == sum && sum - sum == 0.0 &&
                 "NaN in backprop");
    KALDI_ASSERT(sum >= 0.0);
    if (sum <= max_norm_) return 1.0;
    else {
      BaseFloat ans = max_norm_ / sum;
      if (scaling_factor_printed < 10) {
        KALDI_LOG << "Limiting step size to " << max_norm_
                  << " using scaling factor " << ans << ", for component index " ;
        scaling_factor_printed++;
      }
      return ans;
    }
  }

  void Update(const CuMatrixBase<BaseFloat> &in_value, const CuMatrixBase<BaseFloat> &out_deriv) {

    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;

    in_value_temp.Resize(in_value.NumRows(),
                         in_value.NumCols() + 1, kUndefined);
    in_value_temp.Range(0, in_value.NumRows(),
                        0, in_value.NumCols()).CopyFromMat(in_value);

    // Add the 1.0 at the end of each row "in_value_temp"
    in_value_temp.Range(0, in_value.NumRows(),
                        in_value.NumCols(), 1).Set(1.0);

    in_value_precon.Resize(in_value_temp.NumRows(),
                           in_value_temp.NumCols(), kUndefined),
    out_deriv_precon.Resize(out_deriv.NumRows(),
                            out_deriv.NumCols(), kUndefined);
    // each row of in_value_precon will be that same row of
    // in_value, but multiplied by the inverse of a Fisher
    // matrix that has been estimated from all the other rows,
    // smoothed by some appropriate amount times the identity
    // matrix (this amount is proportional to \alpha).
    PreconditionDirectionsAlphaRescaled(in_value_temp, alpha_, &in_value_precon);
    PreconditionDirectionsAlphaRescaled(out_deriv, alpha_, &out_deriv_precon);

    BaseFloat minibatch_scale = 1.0;

    if (max_norm_ > 0.0)
      minibatch_scale = GetScalingFactor(in_value_precon, out_deriv_precon);


    CuSubMatrix<BaseFloat> in_value_precon_part(in_value_precon,
                                                0, in_value_precon.NumRows(),
                                                0, in_value_precon.NumCols() - 1);
    // this "precon_ones" is what happens to the vector of 1's representing
    // offsets, after multiplication by the preconditioner.
    precon_ones.Resize(in_value_precon.NumRows());

    precon_ones.CopyColFromMat(in_value_precon, in_value_precon.NumCols() - 1);

    BaseFloat local_lrate = minibatch_scale * lr;
    BaseFloat local_lrate_bias = minibatch_scale * lr_bias;

    bias_.AddMatVec(-local_lrate_bias, out_deriv_precon, kTrans,
                           precon_ones, 1.0);
    linearity_.AddMatMat(-local_lrate, out_deriv_precon, kTrans,
                             in_value_precon_part, kNoTrans, 1.0);

  }

  CuMatrix<BaseFloat> in_value_temp;
  CuMatrix<BaseFloat> in_value_precon;
  CuMatrix<BaseFloat> out_deriv_precon;
  CuVector<BaseFloat> precon_ones;

};




class LogisticAffine : public AffineTransform {
  public:
  LogisticAffine(int32 dim_in, int32 dim_out) 
    : AffineTransform(dim_in, dim_out) { 
    KALDI_ASSERT(dim_out == 2);
  }
  ~LogisticAffine()
  { }

  Layer* Copy() const { return new LogisticAffine(*this); }
  LayerType GetType() const { return l_Logistic_Affine; }
  
  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // compute gradient
    linearity_grad_.AddMatMat(1.0, diff, kTrans, input, kNoTrans, 0.0);
    bias_grad_.AddRowSumMat(1.0, diff, 0.0);

    linearity_ada_.AddMat(1.0, linearity_grad_, kNoTrans);
    bias_ada_.AddVec(1.0, bias_grad_);

    num_frames_ = input.NumRows();
    Update();
  }

  void Update()  {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum < 0 ? 0 - opts_.momentum : opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch

    // include momentum
    linearity_corr_.Scale(mmt);
    linearity_corr_.AddMat(1.0 - mmt, linearity_grad_, kNoTrans);
    bias_corr_.Scale(mmt);
    bias_corr_.AddVec(1.0 - mmt, bias_grad_);

    // l2 regularization
    if (l2 != 0.0) {
      linearity_.AddMat(-lr*l2*num_frames_, linearity_);
      if (ref_layer_ != NULL) {
        const AffineTransform* af_component = dynamic_cast<const AffineTransform*> (ref_layer_);
        linearity_.AddMat(lr*l2*num_frames_, af_component->GetLinearity());
      }
    }
    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames_, lr);
    }
    // update
    linearity_.AddMat(-lr, linearity_corr_);
    bias_.AddVec(-lr_bias, bias_corr_);
    // max-norm
    if (max_norm_ > 0.0) {
      CuMatrix<BaseFloat> lin_sqr(linearity_);
      lin_sqr.MulElements(linearity_);
      CuVector<BaseFloat> l2(OutputDim());
      l2.AddColSumMat(1.0, lin_sqr, 0.0);
      l2.ApplyPow(0.5); // we have per-neuron L2 norms
      CuVector<BaseFloat> scl(l2);
      scl.Scale(1.0/max_norm_);
      scl.ApplyFloor(1.0);
      scl.InvertElements();
      linearity_.MulRowsVec(scl); // shink to sphere!
    }
    CuSubVector<BaseFloat> weight_row0(linearity_, 0);
    weight_row0.Set(0);
    CuSubVector<BaseFloat> bias_0(bias_, 0, 1);
    bias_0.Set(0);
  }

  void InitData(std::istream &is) {
    // define options
    AffineTransform::InitData(is);
    CuSubVector<BaseFloat> weight_row0(linearity_, 0);
    weight_row0.Set(0);
    CuSubVector<BaseFloat> bias_0(bias_, 0, 1);
    bias_0.Set(0);
  }

};

} // namespace eesen

#endif

