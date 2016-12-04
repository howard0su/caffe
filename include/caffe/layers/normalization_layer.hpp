#pragma once

namespace caffe
{
	/**
	 * * @brief Normalizes input.
	 * */
	template <typename Dtype>
		class NormalizationLayer: public Layer<Dtype> {
			public:
				explicit NormalizationLayer(const LayerParameter& param)
					: Layer<Dtype>(param) {}
				virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
						const vector<Blob<Dtype>*>& top);

				virtual inline const char* type() const { return "Normalization"; }
				virtual inline int ExactNumBottomBlobs() const { return 1; }
				virtual inline int ExactNumTopBlobs() const { return 1; }

			protected:
				virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
						const vector<Blob<Dtype>*>& top);
				virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
						const vector<Blob<Dtype>*>& top);
				virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
						const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
				virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
						const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

				Blob<Dtype> sum_multiplier_, norm_, squared_;
		};
}
