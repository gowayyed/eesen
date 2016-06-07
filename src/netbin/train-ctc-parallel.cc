// netbin/train-ctc-parallel.cc

// Copyright 2015   Yajie Miao, Hang Su, Mohammad Gowayyed

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

#include "net/train-opts.h"
#include "net/net.h"
#include "net/ctc-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "gpucompute/cuda-device.h"
#include "net/communicator.h"
#include "util/text-utils.h"

int main(int argc, char *argv[]) {
  using namespace eesen;
  typedef eesen::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of CTC training by SGD.\n"
        "The updates are done per-utternace and by processing multiple utterances in parallel.\n"
        "\n"
        "Usage: train-ctc-parallel [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        "train-ctc-parallel scp:feature.scp ark:labels.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NetTrainOptions trn_opts;  // training options
    trn_opts.Register(&po); 

    bool binary = true, 
				crossvalidate = false;
    po.Register("binary", &binary, "Write model  in binary mode");

		bool block_softmax = false;
		po.Register("block-softmax", &block_softmax, "Whether to use block-softmax or not (default is false). Note that you have to pass this parameter even if the provided model contains a BlockSoftmax layer.");

		bool include_langid = false;
    po.Register("include-langid", &include_langid, "Whether to include the langid in the input");

    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (no backpropagation)");

    int32 num_sequence = 5;
    po.Register("num-sequence", &num_sequence, "Number of sequences processed in parallel");

    double frame_limit = 100000;
    po.Register("frame-limit", &frame_limit, "Max number of frames to be processed");

    int32 report_step=100;
    po.Register("report-step", &report_step, "Step (number of sequences) for status reporting");

    std::string use_gpu="yes";
//    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    int32 num_jobs = 1;
    po.Register("num-jobs", &num_jobs, "Number subjobs in multi-GPU mode");

    int32 job_id = 1;
    po.Register("job-id", &job_id, "Subjob id in multi-GPU mode");

    int32 utts_per_avg = 500;
    po.Register("utts-per-avg", &utts_per_avg, "Number of utterances to process per average (default is 250)");

    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace eesen;
    typedef eesen::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Net net;
		net.Read(model_filename);
    net.SetTrainOptions(trn_opts);

    eesen::int64 total_frames = 0;

    // Initialize feature and labels readers
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader targets_reader(targets_rspecifier);

    // Initialize CTC optimizer
    Ctc ctc;
    ctc.SetReportStep(report_step);
    CuMatrix<BaseFloat> net_out, obj_diff, obj_diff_block;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    std::vector< Matrix<BaseFloat> > feats_utt(num_sequence);  // Feature matrix of every utterance
    std::vector< std::vector<int> > labels_utt(num_sequence);  // Label vector of every utterance
		

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0, avg_count = 0;


    std::vector<int> block_softmax_dims(0);
    if(block_softmax)
		{
      block_softmax_dims = net.GetBlockSoftmaxDims();
		}

		int32 feat_dim = net.InputDim(); // adding a one hot vector for now

    while (1) {

      std::vector<int> frame_num_utt;
      int32 sequence_index = 0, max_frame_num = 0; 

      for ( ; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        // Check that we have targets
        KALDI_LOG << "processing utterance " << utt;
        if (!targets_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing targets";
          num_no_tgt_mat++;
          continue;
        }
				
        // Get feature / target pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        std::vector<int32> targets = targets_reader.Value(utt);

        if (max_frame_num < mat.NumRows()) max_frame_num = mat.NumRows();
        feats_utt[sequence_index] = mat;
        labels_utt[sequence_index] = targets;
        frame_num_utt.push_back(mat.NumRows());
        sequence_index++;
        // If the total number of frames reaches frame_limit, then stop adding more sequences, regardless of whether
        // the number of utterances reaches num_sequence or not.
        if (frame_num_utt.size() == num_sequence || frame_num_utt.size() * max_frame_num > frame_limit) {
            feature_reader.Next(); break;
        }
      }
      int32 cur_sequence_num = frame_num_utt.size();
      
      // Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
      Matrix<BaseFloat> feat_mat_host(cur_sequence_num * max_frame_num, feat_dim, kSetZero);
			Matrix<BaseFloat> given(cur_sequence_num * max_frame_num, 1, kSetZero); // only used when conditioningi
			// KALDI_LOG << "BEFORE START";	
			// KALDI_LOG << feat_dim;

			if(net.IsConditioning()){
				given.Resize(cur_sequence_num * max_frame_num, net.GetConditionInDim());
				Vector<BaseFloat> giv(net.GetConditionInDim(), kSetZero);
				Vector<BaseFloat> oneVec(1, kSetZero);
        oneVec.ReplaceValue(0, 1);
        for (int s = 0; s < cur_sequence_num; s++) {
          int startIdx = 0;
          int bl = 1000;
          for(int i = 0; i < block_softmax_dims.size(); i++) {
            if(labels_utt[s].size() > 0 && labels_utt[s][0] >= startIdx && labels_utt[s][0] < startIdx + block_softmax_dims[i]){
              bl = i;
            }
            startIdx += block_softmax_dims[i];
          }

          Matrix<BaseFloat> mat_tmp = feats_utt[s];
          for (int r = 0; r < frame_num_utt[s]; r++) {
						giv.Range(bl, 1).CopyFromVec(oneVec);
						given.Row(r*cur_sequence_num + s).CopyFromVec(giv);
                
					}
        }
			}
			// KALDI_LOG << "START";	
			Vector<BaseFloat> feat(feat_dim, kSetZero);

			if(include_langid) {
				Vector<BaseFloat> oneVec(1, kSetZero);
				oneVec.ReplaceValue(0, 1);
				for (int s = 0; s < cur_sequence_num; s++) {
					int startIdx = 0;
					int bl = 1000;
          for(int i = 0; i < block_softmax_dims.size(); i++) {
						if(labels_utt[s].size() > 0 && labels_utt[s][0] >= startIdx && labels_utt[s][0] < startIdx + block_softmax_dims[i]){
							bl = i;
						}
						startIdx += block_softmax_dims[i];
					}
	
	        Matrix<BaseFloat> mat_tmp = feats_utt[s];
					for (int r = 0; r < frame_num_utt[s]; r++) {
						feat.Range(0, mat_tmp.NumCols()).CopyFromVec(mat_tmp.Row(r));
						// we get the index of this language
						
						feat.Range(mat_tmp.NumCols() + bl, 1).CopyFromVec(oneVec);
						feat_mat_host.Row(r*cur_sequence_num + s).CopyFromVec(feat);
					}
				}
			} else {
			 //KALDI_LOG << feats_utt[0].NumCols();
  		//	KALDI_LOG << feats_utt[0].NumRows();
			//KALDI_LOG << feat_mat_host.Row(0*cur_sequence_num + 0).NumCols();
	     for (int s = 0; s < cur_sequence_num; s++) {
//				KALDI_LOG << "B 1";
	       Matrix<BaseFloat> mat_tmp = feats_utt[s];
//				 KALDI_LOG << "B 2";

	       for (int r = 0; r < frame_num_utt[s]; r++) {
//					 KALDI_LOG << "B 3";
//					 KALDI_LOG << feat_mat_host.Row(r*cur_sequence_num + s).Dim();
//					 KALDI_LOG <<	mat_tmp.Row(r).Dim();
	         feat_mat_host.Row(r*cur_sequence_num + s).CopyFromVec(mat_tmp.Row(r));
	       }
//				 KALDI_LOG << "B 4";

	      }
			}      

      // Set the original lengths of utterances before padding
      net.SetSeqLengths(frame_num_utt);
			// KALDI_LOG << "BEFORE ANYTHING";
			// Propagation and CTC training
			if(net.IsConditioning()){
				net.PropagateCond(CuMatrix<BaseFloat>(feat_mat_host), CuMatrix<BaseFloat>(given), &net_out);
			} else {
	      net.Propagate(CuMatrix<BaseFloat>(feat_mat_host), &net_out);
			}

			// KALDI_LOG << "END OF PROPAGATION";

			// I moved the Resize outside the EvalParallel for the block softmax to be convenient 
      obj_diff.Resize(net_out.NumRows(), net_out.NumCols());
			obj_diff.Set(0);
			
      if(block_softmax && block_softmax_dims.size() > 0) {	
        int startIdx = 0; 
        for(int i = 0; i < block_softmax_dims.size(); i++) {
          // we need to get the submatrix that corresponds to the current block	
					std::vector< std::vector<int> > labels_utt_block(cur_sequence_num);
					std::vector<int> frame_num_utt_block(cur_sequence_num);
					// for now, we assume that the original labels use the whole index, so we need to change them to be relative to the current softmax
					int nonzero_seq = 0;
					
					for(int s = 0; s < cur_sequence_num; s++){
					// we need to check if this sequence belongs to this block	
						if(labels_utt[s].size() > 0 && labels_utt[s][0] >= startIdx && labels_utt[s][0] < startIdx + block_softmax_dims[i]){
								frame_num_utt_block[s] = frame_num_utt[s];
								for(int r = 0; r < labels_utt[s].size(); r++){
									labels_utt_block[s].push_back(labels_utt[s][r] - startIdx);
								}
								nonzero_seq++;
						}else{
							frame_num_utt_block[s] = 0;
						}
					}
					if(nonzero_seq > 0)
					{
						CuSubMatrix<BaseFloat> net_out_block = net_out.ColRange(startIdx, block_softmax_dims[i]);
						CuSubMatrix<BaseFloat> obj_diff_block = obj_diff.ColRange(startIdx, block_softmax_dims[i]);
						ctc.EvalParallel(frame_num_utt_block, net_out_block, labels_utt_block, &obj_diff_block);
					  // Error rates
					  ctc.ErrorRateMSeq(frame_num_utt_block, net_out_block, labels_utt_block);
					}
					startIdx += block_softmax_dims[i];
				}
      } else {
        ctc.EvalParallel(frame_num_utt, net_out, labels_utt, &obj_diff);
        // Error rates
        ctc.ErrorRateMSeq(frame_num_utt, net_out, labels_utt);
      }
      // Backward pass
      if (!crossvalidate) {
				if(!net.IsConditioning()){
	        net.Backpropagate(obj_diff, NULL);
				} else {
					net.BackpropagateCond(obj_diff, NULL);	
				}
	      if (num_jobs != 1 && (num_done + cur_sequence_num) / utts_per_avg != num_done / utts_per_avg) {
		      comm_avg_weights(net, job_id, num_jobs, avg_count, target_model_filename);
			    avg_count++;
			  }
			}
           
      num_done += cur_sequence_num;
      total_frames += feat_mat_host.NumRows();
      
      if (feature_reader.Done()) break; // end loop of while(1)
    }
		
    if (num_jobs != 1) {
      if (!crossvalidate) {
        comm_avg_weights(net, job_id, num_jobs, avg_count, target_model_filename);
        std::string avg_model_name = comm_avg_model_name(target_model_filename, avg_count);
        rename(avg_model_name.c_str(), target_model_filename.c_str());
      }
      std::string base_done_filename = crossvalidate ? model_filename + ".cv" : target_model_filename + ".tr";
      comm_touch_done(ctc, job_id, num_jobs, base_done_filename);
      avg_count++;
      KALDI_LOG << "Total average operations: " << avg_count;
    }
    
    // Print statistics of gradients when training finishes 
    if (!crossvalidate) {
      KALDI_LOG << net.InfoGradient();
    }
		
    if (!crossvalidate) {
      net.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no targets, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";
    KALDI_LOG << ctc.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
