#ifndef CAFFE_ACCURACY_LAYER_HPP_
#define CAFFE_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class DetectionAccuracyLayer : public LossLayer<Dtype> {
 public:
  explicit DetectionAccuracyLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionAccuracy"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 
  /// @brief Not implemented -- DetectionAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  
  int cls_num_;
  int coord_num_;
  int field_h_;
  int field_w_;
  int downsample_rate_;
  int top_k_;
  bool objectness_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
