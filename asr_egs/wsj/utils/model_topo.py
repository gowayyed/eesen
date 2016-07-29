#!/usr/bin/env python

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)
#						2015			 Mohammad Gowayyed (Carnegie Mellon University)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import sys, re

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--","").replace("-", "_");
        args[key] = arg_elements[2*i+1]
    return args


if __name__ == '__main__':

    """
    Python script to generate the network topology. Parameters:
    ------------------
    --input-feat-dim : int
        Dimension of the input features
        Required.
    --lstm-layer-num : int
        Number of LSTM layers
        Required.
    --lstm-cell-dim : int
        Number of memory cells in LSTM. For the bi-directional case, this is the number of cells
        in either the forward or the backward sub-layer.
        Required.
    --target-num : int
        Number of labels as the targets
        Required.
		--block-softmax-dims
				Generate <BlockSoftmax> with dims D1:D2:D3 [default: %default] (copied from Kaldi)
    --param-range : float
        Range to randomly draw the initial values of model parameters. For example, setting it to
        0.1 means model parameters are drawn uniformly from [-0.1, 0.1]
        Optional. By default it is set to 0.1.
    --lstm-type : string
        Type of LSTMs. Optional. Either "bi" (bi-directional) or "uni" (uni-directional). By default,
        "bi" (bi-directional).
    --fgate-bias-init : float
        Initial value of the forget-gate bias. Not specifying this option means the forget-gate bias
        will be initialized randomly, in the same way as the other parameters. 
		--preconditioned : string
				"yes" if you want to make the affined (for now) preconditioned
		--preconditioned-bilstm : string
        "yes" if you want to make the bilsm preconditioned

    """


    # parse arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)

    # these 4 arguments are mandatory
    input_feat_dim=int(arguments['input_feat_dim'])
    lstm_layer_num=int(arguments['lstm_layer_num'])
    lstm_cell_dim=int(arguments['lstm_cell_dim'])
    target_num=int(arguments['target_num'])

    # by default, the range of the parameters is set to 0.1; however, you can change it by specifying "--param-range"
    # this means for initialization, model parameters are drawn uniformly from the interval [-0.1, 0.1]
    param_range='0.1'
    if arguments.has_key('param_range'):
        param_range = arguments['param_range']
    if 'preconditioned_bilstm' in arguments.keys() and arguments['preconditioned_bilstm'] == 'yes':
      model_type = '<BiLstmParallelPreconditioned>'
    else:
      model_type = '<BiLstmParallel>'   # by default
    if arguments.has_key('lstm_type') and arguments['lstm_type'] == 'uni':
        model_type = '<LstmParallel>'

    print '<Nnet>'
    lstm_comm = ' <ParamRange> ' + param_range + ' <LearnRateCoef> 1.0 <MaxGrad> 50.0'

    # add the option to set the initial value of the forget-gate bias
    if arguments.has_key('fgate_bias_init'):
        lstm_comm = lstm_comm + ' <FgateBias> ' + arguments['fgate_bias_init']

    actual_cell_dim = 2*lstm_cell_dim
    if model_type == '<LstmParallel>':
        actual_cell_dim = lstm_cell_dim 

    # the first layer takes input features
    print model_type + ' <InputDim> ' + str(input_feat_dim) + ' <CellDim> ' + str(actual_cell_dim) + lstm_comm
    # the following bidirectional LSTM layers
    for n in range(1, lstm_layer_num):
         print model_type + ' <InputDim> ' + str(actual_cell_dim) + ' <CellDim> ' + str(actual_cell_dim) + lstm_comm
    # the final affine-transform and softmax layer
    if 'preconditioned' in arguments.keys() and arguments['preconditioned'] == 'yes':
      print '<AffineTransformPrecond> <InputDim> ' + str(actual_cell_dim) + ' <OutputDim> ' + str(target_num) + ' <ParamRange> ' + param_range
    else:
      print '<AffineTransform> <InputDim> ' + str(actual_cell_dim) + ' <OutputDim> ' + str(target_num) + ' <ParamRange> ' + param_range
		
    if arguments.has_key('block_softmax_dims') and arguments['block_softmax_dims'] != "":
			ddims = map(int, re.split("[,:]", arguments['block_softmax_dims']))
			assert(sum(ddims) == target_num )
			print '<BlockSoftmax> <InputDim> ' + str(target_num) + ' <OutputDim> ' + str(target_num) + ' <BlockDims> ' + arguments['block_softmax_dims']
    else:
      print '<Softmax> <InputDim> ' + str(target_num) + ' <OutputDim> ' + str(target_num)
    print '</Nnet>'
