"""
 Authors: Shanmuga Venkatachalam, Harideep Nair
"""

### Importing libraries ###

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


### Active Dendrite Layer with SNL Response Function ###

    # Implements a single NeuTNN layer with multiple neurons, dendrites, and segments.
    # Segments are assumed to be distal, while the proximal input is simply used as enable during weight update.
    # Capable of online continuous learning via STDP/R-STDP.
    
    # Args: numDistalInputs - Number of distal inputs to a single neuron. Within a neuron, the inputs are shared across all segments.
    #       numSegments     - Number of segments per dendrite per neuron.
    #       numDendrites    - Number of dendrites per neuron.
    #       numNeurons      - Number of neurons in the layer.
    #       threshold       - Excitation threshold for neuron. An output spike is generated when neuron's body potential reaches
    #                         this threshold.
    #       wMax            - Maximum value for synaptic weights.
    #       wInit           - Type of initialization for synaptic weights. Supports 'zero', 'uniform', and 'normal' initializations.
    #       rewardEn        - Enable reinforcement for STDP (R-STDP).
    #       device          - Device type (e.g. cpu, cuda).
    #
    # Input: data           - Input data to the layer of shape [numNeurons, numDistalInputs].
    #
    # __call__ is the forward function whereas "update" performs STDP update in either unsupervised or supervised fashion.

class activeDendriteLayer_SNL(nn.Module):
    def __init__(self,
                 numDistalInputs: int,
                 numSegments: int,
                 numDendrites: int,
                 numNeurons: int,
                 threshold: int,
                 wMax: int = 8,
                 wInit: str = "uniform",
                 rewardEn: bool = True,
                 device="cpu"
                ):

        super(activeDendriteLayer_SNL, self).__init__()

        self.numDistalInputs = numDistalInputs
        self.numSegments = numSegments
        self.numDendrites = numDendrites
        self.numNeurons = numNeurons
        self.threshold = threshold
        self.wMax = wMax
        self.wInit = wInit
        self.rewardEn = rewardEn
        self.device = device

        self.totalSegments = self.numNeurons*self.numDendrites*self.numSegments

        if self.wInit       == "zero":
            self.weights   = nn.Parameter(torch.zeros(self.totalSegments,
                                                        self.numDistalInputs), requires_grad=False)

        elif self.wInit     == "uniform":
            self.weights   = nn.Parameter(torch.randint(low = 0, high = self.wMax + 1,
                                            size=(self.totalSegments,
                                                self.numDistalInputs)).type(torch.FloatTensor),
                                            requires_grad=False)

        elif self.wInit     == "normal":
            self.weights   = nn.Parameter(torch.round(( (self.wMax+1)/2 + \
                                                        torch.randn(self.totalSegments, self.numDistalInputs))\
                                                        .clamp_(0,self.wMax)), requires_grad=False)


    ########################################### Inference function ###########################################
    def __call__(self, data):
        # Data is of shape [self.numNeurons, self.numDistalInputs]

        inputData = data.unsqueeze(1).repeat(1,self.numDendrites*self.numSegments,1)\
                        .reshape(self.totalSegments,self.numDistalInputs)

        # Applying weights
        weightedInput = inputData * self.weights

        # Weighted input summation - SNL
        summedOutput = torch.sum(weightedInput, dim=1)

        zeroCondition = summedOutput < self.threshold
        summedOutput[zeroCondition] = 0

        # Winner Take All inhibition across segments
        reshapedSummedOutput = summedOutput.reshape(-1,self.numSegments)

        self.wtaOutput = torch.zeros(self.numNeurons*self.numDendrites, self.numSegments,
                                     dtype=reshapedSummedOutput.dtype, device=self.device)

        maxval, indices = torch.max(reshapedSummedOutput, dim=1)

        self.wtaOutput.scatter_(1, indices.unsqueeze(1), maxval.unsqueeze(1))

        outputData = (self.wtaOutput>0).reshape(-1).unsqueeze(1).repeat(1,self.numDistalInputs)

        return inputData, outputData


    ########################################### Update function ###########################################
    def update(self, inputData, inputLabels, outputData, weights, capture, search, backoff, wBase, rewardEn, frac=0.25):

        if rewardEn:
            captureVal = (inputData == 1) * (outputData == 1) * (inputLabels == 1) * capture + (inputData == 0) * (outputData == 1) * (inputLabels == 0) * capture * frac
            backoffVal = (inputData == 0) * (outputData == 1) * (inputLabels == 1) * backoff + (inputData == 1) * (outputData == 1) * (inputLabels == 0) * backoff * frac
            searchCondition  = (inputData == 1) * (outputData == 0) * (inputLabels == 1)
        else:
            captureVal = (inputData == 1) * (outputData == 1) * capture
            backoffVal = (inputData == 0) * (outputData == 1) * backoff
            searchCondition  = (inputData == 1) * (outputData == 0)

        # Capture and Backoff
        weights += captureVal - backoffVal

        # Search
        weights[searchCondition] = torch.maximum(weights[searchCondition],
                                                 torch.minimum(weights[searchCondition] + search,
                                                               torch.tensor([wBase], device=self.device)))

        return weights.clamp_(0, self.wMax)


### Active Dendrite Layer with RNL Response Function ###

    # Implements a single NeuTNN layer with multiple neurons, dendrites, and segments.
    # Segments are assumed to be distal, while the proximal input is simply used as enable during weight update.
    # Capable of online continuous learning via STDP/R-STDP.
    
    # Args: numDistalInputs - Number of distal inputs to a single neuron. Within a neuron, the inputs are shared across all segments.
    #       numSegments     - Number of segments per dendrite per neuron.
    #       numDendrites    - Number of dendrites per neuron.
    #       numNeurons      - Number of neurons in the layer.
    #       threshold       - Excitation threshold for neuron. An output spike is generated when neuron's body potential reaches
    #                         this threshold.
    #       wMax            - Maximum value for synaptic weights.
    #       wInit           - Type of initialization for synaptic weights. Supports 'zero', 'uniform', and 'normal' initializations.
    #       rewardEn        - Enable reinforcement for STDP (R-STDP).
    #       device          - Device type (e.g. cpu, cuda).
    #
    # Input: data           - Input data to the layer of shape [numNeurons, numDistalInputs].
    #
    # __call__ is the forward function whereas "update" performs STDP update in either unsupervised or supervised fashion.

class activeDendriteLayer_RNL(nn.Module):
    def __init__(self,
                 numDistalInputs: int,
                 numSegments: int,
                 numDendrites: int,
                 numNeurons: int,
                 threshold: int,
                 wMax: int = 7,
                 wInit: str = "zero",
                 rewardEn: bool = True,
                 device="cpu"
                ):

        super(activeDendriteLayer_RNL, self).__init__()

        self.numDistalInputs = numDistalInputs
        self.numSegments = numSegments
        self.numDendrites = numDendrites
        self.numNeurons = numNeurons
        self.threshold = threshold
        self.wMax = wMax
        self.wInit = wInit
        self.rewardEn = rewardEn
        self.device = device

        self.totalSegments = self.numNeurons*self.numDendrites*self.numSegments

        if self.wInit       == "zero":
            self.weights   = nn.Parameter(torch.zeros(self.totalSegments,
                                                      self.numDistalInputs), requires_grad=False)

        elif self.wInit     == "uniform":
            self.weights   = nn.Parameter(torch.randint(low = 0, high = self.wMax + 1,
                                          size=(self.totalSegments,
                                                self.numDistalInputs)).type(torch.FloatTensor),
                                          requires_grad=False)

        elif self.wInit     == "normal":
            self.weights   = nn.Parameter(torch.round(( (self.wMax+1)/2 + \
                                                        torch.randn(self.totalSegments, self.numDistalInputs))\
                                                      .clamp_(0,self.wMax)), requires_grad=False)

        self.increasingNum = torch.arange(1, self.wMax+1).repeat(self.totalSegments,self.numDistalInputs,1).to(device)


    ########################################### Inference function ###########################################
    def __call__(self, data):
        # Data is of shape [self.numNeurons, self.numDistalInputs]

        inputData = data.unsqueeze(1).repeat(1,self.numDendrites*self.numSegments,1)\
                        .reshape(self.totalSegments,self.numDistalInputs)

        # Applying weights
        weightedInput = inputData * self.weights

        rnlResponse = torch.minimum(weightedInput.unsqueeze(2).repeat(1,1,self.wMax), self.increasingNum)

        # Weighted input summation - RNL
        summedOutput = torch.sum(rnlResponse, dim=1)

        zeroCondition = summedOutput < self.threshold
        summedOutput[zeroCondition] = 0
        thresholdCrossed = summedOutput != 0

        detectOutputTime = thresholdCrossed & (torch.cumsum(thresholdCrossed, 1) == 1)
        rnlOutputTimes = torch.max(detectOutputTime, dim=1)[1]
        zeroIndices = torch.sum(zeroCondition,dim=1) == self.wMax
        rnlOutputTimes[zeroIndices] = self.wMax

        # Winner Take All inhibition across segments
        reshapedRnlOutput = rnlOutputTimes.reshape(-1,self.numSegments)

        self.wtaOutput = torch.ones(self.numNeurons*self.numDendrites, self.numSegments,
                                     dtype=reshapedRnlOutput.dtype, device=self.device) * self.wMax

        minval, indices = torch.min(reshapedRnlOutput, dim=1)

        self.wtaOutput.scatter_(1, indices.unsqueeze(1), minval.unsqueeze(1))

        outputData = (self.wtaOutput < self.wMax).reshape(-1).unsqueeze(1).repeat(1,self.numDistalInputs)

        return inputData, outputData


    ########################################### Update function ###########################################
    def update(self, inputData, inputLabels, outputData, weights, capture, search, backoff, wBase, rewardEn, frac=0.25):

        if rewardEn:
            captureVal = (inputData == 1) * (outputData == 1) * (inputLabels == 1) * capture + (inputData == 0) * (outputData == 1) * (inputLabels == 0) * capture * frac
            backoffVal = (inputData == 0) * (outputData == 1) * (inputLabels == 1) * backoff + (inputData == 1) * (outputData == 1) * (inputLabels == 0) * backoff * frac
            searchCondition  = (inputData == 1) * (outputData == 0) * (inputLabels == 1)
        else:
            captureVal = (inputData == 1) * (outputData == 1) * capture
            backoffVal = (inputData == 0) * (outputData == 1) * backoff
            searchCondition  = (inputData == 1) * (outputData == 0)

        # Capture and Backoff
        weights += captureVal - backoffVal

        # Search
        weights[searchCondition] = torch.maximum(weights[searchCondition],
                                                 torch.minimum(weights[searchCondition] + search,
                                                               torch.tensor([wBase], device=self.device)))

        return weights.clamp_(0, self.wMax)


### TNN Column Layer###

    # Implements a single TNN layer with multiple TNN columns arranged as a 2D array of size [rows,cols].
    # Each column consists of an excitatory column with 'q' neurons and 'p' synapses per neuron, followed by lateral inhibition.
    # Capable of online continuous learning via STDP/R-STDP.
    
    # Args: input_size - Size of input 2D data (e.g. input image size in a dataset). Can be given as (height,width) tuple
    #                    or integer, in which case both height and width will be set as that integer.
    #       rf_size    - Size of 2D receptive field. Can be given as (height,width) tuple or integer, in which case both
    #                    height and width will be set as that integer.
    #       stride     - Stride for sliding the receptive field across the input 2D data.
    #       nprev      - Number of neurons per column in the previous layer. For first layer, channels added due to filtering
    #                    can be accounted for using nprev. E.g. for OnOff filter, nprev = 2. Else, if there is no filtering,
    #                    nprev = 1 for first layer.
    #       neurons    - Number of neurons per column in the current layer.
    #       theta      - Excitation threshold for neuron. An output spike is generated when neuron's body potential reaches
    #                    this threshold.
    #       timeres    - Bit resolution for input spiketimes.
    #       wres       - Bit resolution for synaptic weights.
    #       w_init     - Type of initialization for synaptic weights. Supports 'zero', 'half', 'random uniform' and
    #                    'random normal' initializations.
    #       ntype      - Type of neuron response function. Supports step-no-leak and ramp-no-leak response functions.
    #       ramp       - Slope of ramp for ramp-no-leak response function. Default is 1.
    #       k          - Maximum number of winners to allow uninhibited.
    #       stoch      - Decides how correlated or uncorrelated weight updates will be, for a column's synapses. Could
    #                    be low or high.
    #       reward_en  - Enable reinforcement for STDP (R-STDP).
    #       corners    - Enable taking only pixels at the four corners of an RF.
    #       data       - Input 4D data of shape [batch_size,channels,height,width].
    # Returns 1) LI output spiketimes consisting of upto k non-null spikes that can be directly used as input to next layer.
    #            Shape is [batch_size,self.q(channels),self.rows(height),self.cols(width)].
    #         2) Re-dimensionalized input spiketimes that can be directly used to compute STDP weight updates.
    #            Shape is [batch_size,self.rows,self.cols,self.q,self.p].
    #         3) Re-dimensionalized LI output spiketimes that can be directly used to compute STDP weight updates.
    #            Shape is [batch_size,self.rows,self.cols,self.q,self.p].

class TNNColumnLayer(nn.Module):
    def __init__(self, inputsize, rfsize, stride, nprev, neurons, theta, timeres=3, wres=3,\
                 w_init="half", ntype="rnl", ramp=1, k=1, stoch="low", reward_en=0, dilated=0, dilation_step=2):
        super(TNNColumnLayer, self).__init__()
        if not isinstance(inputsize,tuple):
            inputsize      = (inputsize,inputsize)
        if not isinstance(rfsize,tuple):
            rfsize         = (rfsize,rfsize)
        
        self.rfsize        = rfsize
        self.stride        = stride
        self.rows          = ((inputsize[0]-rfsize[0])//stride) + 1
        self.cols          = ((inputsize[1]-rfsize[1])//stride) + 1
        
        self.p             = rfsize[0]*rfsize[1]*nprev
        self.dilated       = dilated
        self.dilation_step = dilation_step
        if dilated == 1:
            self.p         = int(np.ceil(rfsize[0]/dilation_step) * np.ceil(rfsize[1]/dilation_step) * nprev)
        
        self.q             = neurons
        self.num           = self.rows*self.cols*self.q
        self.theta         = theta
        self.wres          = wres
        self.ntype         = ntype
        self.ramp          = ramp
        self.k             = k
        
        max_tin            = 2**timeres-1
        if self.ntype   == "snl":
            self.time      = max_tin + 1
        elif self.ntype == "rnl":
            self.time      = max_tin + (2**self.wres-1)
        
        # Synaptic weight initialization
        self.wmax          = 2**self.wres - 1
        if w_init       == "zero":
            self.weights   = nn.Parameter(torch.zeros(self.num, self.p), requires_grad=False)
        elif w_init     == "half":
            self.weights   = nn.Parameter((self.wmax/2)*torch.ones(self.num, self.p),\
                                          requires_grad=False)
        elif w_init     == "uniform":
            self.weights   = nn.Parameter(torch.randint(low=0, high=self.wmax+1, \
                                          size=(self.num, self.p)).type(torch.FloatTensor),\
                                          requires_grad=False)
        elif w_init     == "normal":
            self.weights   = nn.Parameter(torch.round(((self.wmax+1)/2+torch.randn(self.num, \
                                                                              self.p)\
                                                      ).clamp_(0,self.wmax)\
                                                     ),\
                                          requires_grad=False)
        
        if reward_en == 0:
            self.stdp      = STDP_Deterministic(wres)
        else:
            self.stdp      = RSTDP_Deterministic(wres)
        
        self.constidx      = torch.arange(self.num).unsqueeze(1).repeat(1, self.p)
        self.constdec      = torch.arange(self.p, 0, -1).repeat(self.num,1)
        self.constidx2     = torch.arange(self.num)
        self.consttheta    = torch.ones(self.num, self.p) * self.theta
        self.constw        = torch.arange(self.wmax).repeat(self.num,1)
        self.constw2       = torch.arange(self.wmax, 0, -1).repeat(self.num,1)
    
    def __call__(self, data):
        sliced_data                              = data.unfold(0, self.rfsize[0], self.stride)\
                                                       .unfold(1, self.rfsize[1],self.stride)
        if self.dilated == 1:
            sliced_data                          = sliced_data[:,:,:,0:self.dilation_step:,0:self.dilation_step:]
        input_spiketimes                         = sliced_data.unsqueeze(2).repeat(1,1,self.q,1,1,1)\
                                                                           .reshape(self.num,self.p)
        
        eff_weights                              = self.weights.clone()
        nullinput_idx                            = input_spiketimes == float('Inf')
        eff_weights[nullinput_idx]               = 0
        
        sorted_times, sorted_idx                 = torch.sort(input_spiketimes, dim=1)
        sorted_weights                           = eff_weights[self.constidx.to(data.device), sorted_idx]
        
        self.pot                                 = torch.cumsum(sorted_weights, dim=1)
        lt_pot                                   = torch.lt(self.pot,self.consttheta.to(data.device))
        self.pot[lt_pot]                         = 0
        self.pot[torch.logical_not(lt_pot)]      = 1
        
        idxmul                                   = self.pot * self.constdec.to(data.device)
        indices                                  = torch.argmax(idxmul, 1)
        
        ec_times                                 = sorted_times[self.constidx2.to(data.device), indices]
        nullspike_idx                            = (torch.sum(self.pot, dim=1) == 0)
        ec_times[nullspike_idx]                  = float('Inf')
        
        if self.ntype == "rnl":
            nonnull_idx                          = torch.logical_not(nullspike_idx)
            rnl_times_orig                       = ec_times[nonnull_idx]
            pruned_num                           = rnl_times_orig.shape[0]
            
            rnl_times                            = rnl_times_orig.unsqueeze(1).repeat(1,self.wmax)\
                                                       + self.constw.to(data.device)[0:pruned_num]
            rnl_times                            = rnl_times.unsqueeze(2).repeat(1,1,self.p)
            
            in_times                             = input_spiketimes[nonnull_idx].unsqueeze(1).repeat(1,self.wmax,1)
            resp_unnorm                          = rnl_times + 1 - in_times
            resp_weights                         = self.weights[nonnull_idx].unsqueeze(1).repeat(1,self.wmax,1)
            resp                                 = torch.minimum(resp_unnorm, resp_weights)
            resp                                 = torch.maximum(resp,torch.zeros(resp.shape).to(data.device))
            respsum                              = torch.sum(resp, dim=2)
            
            lt_resp                              = respsum  < self.theta
            respsum[lt_resp]                     = 0
            respsum[torch.logical_not(lt_resp)]  = 1
            
            idxmul2                              = respsum * self.constw2.to(data.device)[0:pruned_num]
            indices                              = torch.argmax(idxmul2, 1)
            ec_times[nonnull_idx]                = rnl_times_orig + indices
        
        self.ec_spiketimes                       = ec_times.reshape(-1,self.q)
        
        ### Lateral Inhibition
        self.li_spiketimes                       = float('Inf')*torch.ones(self.rows*self.cols,self.q)\
                                                                                            .to(data.device)
        sort_times, sort_idx                     = torch.sort(self.ec_spiketimes, dim=1)
        sort_times                               = sort_times.type(self.li_spiketimes.dtype)
        
        if self.k == 1:
            self.li_spiketimes.scatter_(1, sort_idx[:,0].unsqueeze(1), sort_times[:,0].unsqueeze(1))
        else:
            self.li_spiketimes.scatter_(1, sort_idx[:,:self.k], sort_times[:,:self.k])
        
        output_spiketimes_stdp                   = self.li_spiketimes.reshape(-1).unsqueeze(1).repeat(1,self.p)
        output_spiketimes_nextlayer              = self.li_spiketimes.reshape(self.rows,self.cols,self.q)
        
        return output_spiketimes_nextlayer, input_spiketimes, output_spiketimes_stdp


### TNN Voter-Tally Layer - Low threshold voter only ###

    # Implements a TNN layer with multiple TNN voters with single threshold and a final tally.
    # Each voter is connected to a single column in the previous layer.
    # Each voter generates upto 'q' votes from 'p' inputs (outputs of last layer of TNN columns).
    # 'q' is equal to the number of classes or labels. 'p' is equal to the number of neurons per column in the previous layer.
    # Capable of online continuous learning via R-STDP.
    # Final tally block finds the label with maximum vote and selects that as the prediction. In case of ties, winning
    # label with the highest index is chosen.
    
    # Args: rows         - Number of rows of TNN columns/voters, when the previous layer of TNN columns or the current layer of
    #                      TNN voters is viewed as a 2D array.
    #       cols         - Number of cols of TNN columns/voters, when the previous layer of TNN columns or the current layer of
    #                      TNN voters is viewed as a 2D array.
    #       nprev        - Number of neurons per column in the previous layer.
    #       num_classes  - Number of output classes or labels.
    #       thetav       - Threshold for RSTDP update.
    #       tau_eff      - Reduced maximum effective spiketime to decrease computation.
    #       wres         - Bit resolution for synaptic weights.
    #       w_init       - Type of initialization for synaptic weights. Supports 'zero', 'half', 'random uniform' and
    #                      'random normal' initializations.
    #       input_spikes - Input spiketimes from the previous layer of TNN columns. 
    #                      Shape is [batch_size,self.q,self.rows,self.cols].
    # Returns 1) One-hot final prediction. Shape is [batch_size,self.q].
    #         2) Re-dimensionalized input spiketimes of all TNN columns that can be directly used in R-STDP.
    #            Shape is [batch_size,self.rows,self.cols,self.p,self.q,self.tau_eff+1].
    #         3) Re-dimensionalized output votes of all TNN voters.
    #            Shape is [batch_size,self.rows,self.cols,self.p,self.q,self.tau_eff+1].

class TNNVoterTallyLayer(nn.Module):
    def __init__(self, rows, cols, nprev, num_classes, thetav, tau_eff=3, wres=3, w_init="half"):
        super(TNNVoterTallyLayer, self).__init__()
        self.rows          = rows
        self.cols          = cols
        self.p             = nprev
        self.q             = num_classes
        self.tau_eff       = tau_eff
        
        self.wmax          = 2**wres - 1
        
        if w_init       == "zero":
            self.weights   = nn.Parameter(torch.zeros(self.rows*self.cols*self.p,self.q,self.tau_eff+1),\
                                          requires_grad=False)
        elif w_init     == "half":
            self.weights   = nn.Parameter((self.wmax/2)*torch.ones(self.rows*self.cols*self.p,self.q,\
                                                                   self.tau_eff+1),requires_grad=False)
        elif w_init     == "uniform":
            self.weights   = nn.Parameter(torch.randint(low=0, high=self.wmax+1, \
                                                        size=(self.rows*self.cols*self.p,self.q,self.tau_eff+1))\
                                          .type(torch.FloatTensor), requires_grad=False)
        elif w_init     == "normal":
            self.weights   = nn.Parameter(torch.round(((self.wmax+1)/2+torch.randn(self.rows*self.cols*self.p,\
                                                                                   self.q,self.tau_eff+1))\
                                                      .clamp_(0,self.wmax)), requires_grad=False)
        
        self.stdp          = RSTDP_Voter(wres, thetav)
        
        self.const         = torch.arange(self.tau_eff+1).repeat(self.rows*self.cols*self.p,self.q,1)
    
    def __call__(self, input_spikes):
        self.votes                      = torch.zeros(self.rows*self.cols*self.p,self.q,self.tau_eff+1)\
                                                                                        .to(input_spikes.device)

        voter_in                        = input_spikes.unsqueeze(3).repeat(1,1,1,self.q).reshape(-1,self.q)
        
        cond1                           = (voter_in >= self.tau_eff)
        cond2                           = (voter_in != float('Inf'))
        voter_in[cond1 * cond2]         = self.tau_eff

        voter_in                        = voter_in.unsqueeze(2).repeat(1,1,self.tau_eff+1)
        voter_in                        = self.const.to(input_spikes.device) - voter_in
        voter_in[voter_in!=0]           = -1
        voter_in                        += 1

        sel_weights                     = voter_in * self.weights
        cond3                           = (sel_weights >= (self.wmax/2))
        self.votes[cond3]               = 1
        votes                           = self.votes.permute(1,0,2).reshape(self.q,-1)

        self.prediction                 = torch.zeros(self.q).to(input_spikes.device)
        self.tally                      = torch.sum(votes,dim=1)
        self.prediction.scatter_(0,torch.argmax(self.tally,dim=0,keepdim=True),torch.ones(self.q)\
                                                                                 .to(input_spikes.device))
        
        return self.prediction, voter_in, self.votes


### TNN Voter-Tally Layer - Dual threshold voter ###

    # Implements a TNN layer with multiple TNN voters with dual threshold and a final tally.
    # Each voter is connected to a single column in the previous layer.
    # Each voter generates upto 'q' votes from 'p' inputs (outputs of last layer of TNN columns).
    # 'q' is equal to the number of classes or labels. 'p' is equal to the number of neurons per column in the previous layer.
    # Capable of online continuous learning via R-STDP.
    # Final tally block finds the label with maximum vote and selects that as the prediction. In case of ties, winning
    # label with the highest index is chosen.
    
    # Args: rows         - Number of rows of TNN columns/voters, when the previous layer of TNN columns or the current layer of
    #                      TNN voters is viewed as a 2D array.
    #       cols         - Number of cols of TNN columns/voters, when the previous layer of TNN columns or the current layer of
    #                      TNN voters is viewed as a 2D array.
    #       nprev        - Number of neurons per column in the previous layer.
    #       num_classes  - Number of output classes or labels.
    #       thetav_lo    - Low threshold for RSTDP update.
    #       thetav_hi    - High threshold for RSTDP update.
    #       tau_eff      - Reduced maximum effective spiketime to decrease computation.
    #       wres         - Bit resolution for synaptic weights.
    #       w_init       - Type of initialization for synaptic weights. Supports 'zero', 'half', 'random uniform' and
    #                      'random normal' initializations.
    #       input_spikes - Input spiketimes from the previous layer of TNN columns. 
    #                      Shape is [batch_size,self.q,self.rows,self.cols].
    # Returns 1) One-hot final prediction. Shape is [batch_size,self.q].
    #         2) Re-dimensionalized input spiketimes of all TNN columns that can be directly used in R-STDP.
    #            Shape is [batch_size,self.rows,self.cols,self.p,self.q,self.tau_eff+1].
    #         3) Re-dimensionalized output votes of all TNN voters.
    #            Shape is [batch_size,self.rows,self.cols,self.p,self.q,self.tau_eff+1].

class DualTNNVoterTallyLayer(nn.Module):
    def __init__(self, rows, cols, nprev, num_classes, thetav_lo, thetav_hi, tau_eff=3, wres=3, w_init="half"):
        super(DualTNNVoterTallyLayer, self).__init__()
        self.rows          = rows
        self.cols          = cols
        self.p             = nprev
        self.q             = num_classes
        self.tau_eff       = tau_eff
        
        self.wmax          = 2**wres - 1
        
        if w_init       == "zero":
            self.weights   = nn.Parameter(torch.zeros(2,self.rows*self.cols*self.p,self.q,self.tau_eff+1),\
                                          requires_grad=False)
        elif w_init     == "half":
            self.weights   = nn.Parameter((self.wmax/2)*torch.ones(2,self.rows*self.cols*self.p,self.q,\
                                                                   self.tau_eff+1),requires_grad=False)
        elif w_init     == "uniform":
            self.weights   = nn.Parameter(torch.randint(low=0, high=self.wmax+1, \
                                                        size=(2,self.rows*self.cols*self.p,self.q,self.tau_eff+1))\
                                          .type(torch.FloatTensor), requires_grad=False)
        elif w_init     == "normal":
            self.weights   = nn.Parameter(torch.round(((self.wmax+1)/2+torch.randn(2,self.rows*self.cols*self.p,\
                                                                                   self.q,self.tau_eff+1))\
                                                      .clamp_(0,self.wmax)), requires_grad=False)
        
        self.stdp          = DualRSTDP_Voter(wres, thetav_lo, thetav_hi)
        
        self.const         = torch.arange(self.tau_eff+1).repeat(2,self.rows*self.cols*self.p,self.q,1)
    
    def __call__(self, input_spikes):
        self.votes                     = torch.zeros(2,self.rows*self.cols*self.p,self.q,self.tau_eff+1)\
                                                                                        .to(input_spikes.device)
        
        voter_in                       = input_spikes.unsqueeze(3).repeat(1,1,1,self.q).reshape(-1,self.q)\
                                                            .unsqueeze(0).repeat(2,1,1)

        cond1                          = (voter_in >= self.tau_eff)
        cond2                          = (voter_in != float('Inf'))
        voter_in[cond1 * cond2]        = self.tau_eff

        voter_in                       = voter_in.unsqueeze(3).repeat(1,1,1,self.tau_eff+1)
        voter_in                       = self.const.to(input_spikes.device) - voter_in
        voter_in[voter_in!=0]          = -1
        voter_in                       += 1

        sel_weights                    = voter_in * self.weights
        cond3                          = (sel_weights >= (self.wmax/2))
        self.votes[cond3]              = 1
        votes                          = self.votes.permute(2,0,1,3).reshape(self.q,-1)

        self.prediction                 = torch.zeros(self.q).to(input_spikes.device)
        self.tally                      = torch.sum(votes,dim=1)
        self.prediction.scatter_(0,torch.argmax(self.tally,dim=0,keepdim=True),torch.ones(self.q)\
                                                                                        .to(input_spikes.device))
        
        return self.prediction, voter_in[0,:,:,:], self.votes


### Unsupervised Spike Timing Dependent Plasticity (STDP) - Deterministic Version with Fixed Point Updates ###

    # Implements 4 separate cases (capture, minus, search, backoff) depending on the presence or absence
    # of input and output spikes, and their timing relationships.
    # Performs unit infrequent updates based on Bernoulli random variables (BRVs).
    
    # Args: wres          - Bit resolution for weights.
    #       ec_in         - Tensor of input spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       li_out        - Tensor of LI's output spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       weights       - Tensor of synaptic weights with shape [self.rows,self.cols,self.q,self.p].
    #       capture       - Update for 'capture' case.
    #       search        - Update for 'search' case.
    #       backoff       - Update for 'backoff' and 'minus' cases.
    # Returns a tensor of STDP-updated synaptic weights of shape [self.rows,self.cols,self.q,self.p].

class STDP_Deterministic():
    def __init__(self, wres):
        self.wmax           = 2**(wres)-1
        
    def __call__(self, ec_in, li_out, weights, capture, search, backoff):
        
        # Case 1 (capture)
        weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)*(weights>=self.wmax/2)] \
                        += capture*capture
        weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)*(weights<self.wmax/2)] \
                        += capture*capture/2
        
        # Case 2 (minus)
        weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)*(weights<self.wmax/2)] \
                       -= backoff*backoff
        weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)*(weights>=self.wmax/2)] \
                       -= backoff*backoff/2
        
        # Case 3 (search)
        weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] \
                       += search
        
        # Case 4 (backoff)
        weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))*(weights<self.wmax/2)] \
                       -= backoff*backoff
        weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))*(weights>=self.wmax/2)] \
                       -= backoff*backoff/2
        
        return nn.Parameter(weights.clamp_(0, self.wmax), requires_grad=False)


### Spike Timing Dependent Plasticity (STDP) with Reward for TNN Column - Deterministic Version with Fixed Point Updates ###

# Adds partial reinforcement to baseline unsupervised STDP.
# If the winner neuron's index matches the desired index corresponding to the assigned label, then proceed with
# conventional STDP cases except for search.
# Else if the winner neuron does not correspond to the assigned label, do reverse STDP for capture case with
# backoff probability. Backoff and minus cases are not required. Search is executed as usual.

    # Args: wres          - Bit resolution for weights.
    #       reward        - Reward signal to guide STDP to learn the desired output label assignments.
    #                       When the winning neuron index matches the desired label, reward is '1' -> perform conventional STDP.
    #                       When there is no winning neuron, reward is '0' -> just perform search.
    #                       When the winning neuron index doesn't match the desired label, reward is '-1' -> perform anti-STDP.
    #       ec_in         - Tensor of input spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       li_out        - Tensor of LI's output spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       weights       - Tensor of synaptic weights with shape [self.rows,self.cols,self.q,self.p].
    #       capture       - Update for 'capture' case.
    #       search        - Update for 'search' case.
    #       backoff       - Update for 'backoff'and 'minus' cases.
    # Returns a tensor of RSTDP-updated synaptic weights of shape [self.rows,self.cols,self.q,self.p].

class RSTDP_Deterministic():
    def __init__(self, wres):
        self.wmax           = 2**(wres)-1
        
    def __call__(self, reward, ec_in, li_out, weights, capture, search, backoff):
        batch_size      = ec_in.shape[0]
        weights         = weights.unsqueeze(0).repeat(batch_size,1,1,1,1)
        
        if reward == 1:
            # Case 1 (capture)
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)*(weights>=self.wmax/2)] \
                           += backoff*backoff
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)*(weights<self.wmax/2)] \
                           += backoff*backoff/2

            # Case 2 (minus)
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)*(weights<self.wmax/2)] \
                           -= backoff*backoff
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)*(weights>=self.wmax/2)] \
                           -= backoff*backoff/2

            # Case 4 (backoff)
            weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))*(weights<self.wmax/2)] \
                           -= backoff*backoff
            weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))*(weights>=self.wmax/2)] \
                           -= backoff*backoff/2

        elif reward == 0:

            # Case 3 (search)
            weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] \
                           += search

        elif reward == -1:
            # Case 1 (capture)
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)*(weights<self.wmax/2)] \
                           -= backoff*backoff
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)*(weights>=self.wmax/2)] \
                           -= backoff*backoff/2

            # Case 3 (search)
            weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] \
                           += search
        
        weights = torch.sum(weights,dim=0)/batch_size
        return nn.Parameter(weights.clamp_(0, self.wmax), requires_grad=False)
    

### Spike Timing Dependent Plasticity (STDP) with Reward for TNN Voter - Low threshold voter only ###

# In presence of input spike at input 'i' at time 't' and correct label 'j', weight_ijk is increased by (1-thetav).
# In presence of input spike at input 'i' at time 't' and incorrect label 'j', weight_ijk is decreased by thetav.

    # Args: wres          - Bit resolution for weights.
    #       thetav        - Threshold for RSTDP update.
    #       target        - Tensor of input spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       voter_in      - Re-dimensionalized output spiketimes from last layer columns.
    #                       Shape is [batch_size,self.rows,self.cols,self.p,self.q,self.tau_eff+1].
    #       weights       - Voter weights with shape [self.rows,self.cols,self.p,self.q,self.tau_eff+1].
    # Returns updated voter weights of shape [self.rows,self.cols,self.p,self.q,self.tau_eff+1].

class RSTDP_Voter():
    def __init__(self, wres, thetav):
        self.wmax           = 2**(wres)-1
        self.thetav         = thetav
        
    def __call__(self, target, voter_in, weights):
        b, c, d           = voter_in.shape
        constant          = torch.arange(c).to(target.device)
        
        target            = constant - target.squeeze().repeat(c)
        target[target!=0] = -1
        target            += 1
        target            = target.unsqueeze(0).repeat(b,1).unsqueeze(2).repeat(1,1,d)

        weights           -= voter_in * self.thetav
        weights           += voter_in * target

        return nn.Parameter(weights.clamp_(0, self.wmax), requires_grad=False)


### Spike Timing Dependent Plasticity (STDP) with Reward for TNN Voter - Dual threshold voter ###

# In presence of input spike at input 'i' at time 't' and correct label 'j', weight_ijk is increased by (1-thetav).
# In presence of input spike at input 'i' at time 't' and incorrect label 'j', weight_ijk is decreased by thetav.

    # Args: wres          - Bit resolution for weights.
    #       thetav_lo     - Low threshold for RSTDP update.
    #       thetav_hi     - High threshold for RSTDP update.
    #       target        - Tensor of input spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       voter_in      - Re-dimensionalized output spiketimes from last layer columns.
    #                       Shape is [batch_size,self.rows,self.cols,self.p,self.q,self.tau_eff+1].
    #       weights       - Voter weights with shape [self.rows,self.cols,self.p,self.q,self.tau_eff+1].
    # Returns updated voter weights of shape [self.rows,self.cols,self.p,self.q,self.tau_eff+1].

class DualRSTDP_Voter():
    def __init__(self, wres, thetav_lo, thetav_hi):
        self.wmax           = 2**(wres)-1
        self.thetav_lo      = thetav_lo
        self.thetav_hi      = thetav_hi
        
    def __call__(self, target, voter_in, weights):
        b, c, d           = voter_in.shape
        constant          = torch.arange(c).to(target.device)
        
        target            = constant - target.squeeze().repeat(c)
        target[target!=0] = -1
        target            += 1
        target            = target.unsqueeze(0).repeat(b,1).unsqueeze(2).repeat(1,1,d)
        
        weights[0,:,:,:] -= voter_in * self.thetav_lo
        weights[0,:,:,:] += voter_in * target
        weights[1,:,:,:] -= voter_in * self.thetav_hi
        weights[1,:,:,:] += voter_in * target
        
        return nn.Parameter(weights.clamp_(0, self.wmax), requires_grad=False)


### Unsupervised Spike Timing Dependent Plasticity (STDP) - Stochastic Version with Integer Updates ###

    # Implements 4 separate cases (capture, minus, search, backoff) depending on the presence or absence
    # of input and output spikes, and their timing relationships.
    # Performs unit infrequent updates based on Bernoulli random variables (BRVs).
    
    # Args: wres          - Bit resolution for weights.
    #       stochasticity - Decides how correlated or uncorrelated weight updates will be, for a column's synapses. Could
    #                       be low or high.
    #       ec_in         - Tensor of input spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       li_out        - Tensor of LI's output spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       weights       - Tensor of synaptic weights with shape [self.rows,self.cols,self.q,self.p].
    #       rvcapture     - BRV for 'capture' case.
    #       rvsearch      - BRV for 'search' case.
    #       rvbackoff     - BRV for 'backoff' and 'minus' cases.
    #       rvmin         - BRV for enforcing a minimum probability of update.
    #       rvstickup     - BRV for sticking the weights towards wmax. Helps in generating a bimodal weight distribution.
    #       rvstickdown   - BRV for sticking the weights towards 0. Helps in generating a bimodal weight distribution.
    # Returns a tensor of STDP-updated synaptic weights of shape [self.rows,self.cols,self.q,self.p].

class STDP_Stochastic():
    def __init__(self, wres, stochasticity="low"):
        self.wmax           = 2**(wres)-1
        self.stoch          = stochasticity
        
    def __call__(self, ec_in, li_out, weights, rvcapture, rvsearch, rvbackoff, rvmin, rvstickup, rvstickdown):
        batch_size      = ec_in.shape[0]
        weights         = weights.unsqueeze(0).repeat(batch_size,1,1,1,1)
        
        # Low stochasticity - All Bernoulli random variables are shared across the entire column.
        if self.stoch == "low":
            # Case 1 (capture)
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                           += rvcapture * torch.max(rvmin, \
                              rvstickup[weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)].long()])
            
            # Case 2 (minus)
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)] \
                           -= rvbackoff * torch.max(rvmin, \
                              rvstickdown[weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)].long()])
            
            # Case 3 (search)
            weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] \
                           += rvsearch
            
            # Case 4 (backoff)
            weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))] \
                           -= rvbackoff * torch.max(rvmin, \
                              rvstickdown[weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))].long()])
        
        # High stochasticity - Each synapse has a separate Bernoulli random variable associated with it.  
        elif self.stoch == "high":
            
            # Case 1 (capture)
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                           += rvcapture[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                            * torch.max(rvmin[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)], \
                              torch.diagonal(rvstickup[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                              [:,weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)].long()],0))
            
            # Case 2 (minus)
            weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)] \
                           -= rvcapture[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)] \
                            * torch.max(rvmin[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)], \
                              torch.diagonal(rvstickdown[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)] \
                              [:,weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)].long()],0))
            
            # Case 3 (search)
            weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] \
                           += rvsearch[(ec_in!=float('Inf'))*(li_out==float('Inf'))]
            
            # Case 4 (backoff)
            weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))] \
                           -= rvbackoff[(ec_in==float('Inf'))*(li_out!=float('Inf'))] \
                            * torch.max(rvmin[(ec_in==float('Inf'))*(li_out!=float('Inf'))], \
                              torch.diagonal(rvstickdown[(ec_in==float('Inf'))*(li_out!=float('Inf'))] \
                              [:,weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))].long()],0))
        
        weights = torch.round(torch.sum(weights,dim=0)/batch_size)
        return nn.Parameter(weights.clamp_(0, self.wmax), requires_grad=False)


### Spike Timing Dependent Plasticity (STDP) with Reward for TNN Column - Stochastic Version with Integer Updates ###

# Adds partial reinforcement to baseline unsupervised STDP.
# If the winner neuron's index matches the desired index corresponding to the assigned label, then proceed with
# conventional STDP cases except for search.
# Else if the winner neuron does not correspond to the assigned label, do reverse STDP for capture case with
# backoff probability. Backoff and minus cases are not required. Search is executed as usual.

    # Args: wres          - Bit resolution for weights.
    #       stochasticity - Decides how correlated or uncorrelated weight updates will be, for a column's synapses. Could
    #                       be low or high.
    #       reward        - Reward signal to guide STDP to learn the desired output label assignments.
    #                       When the winning neuron index matches the desired label, reward is '1' -> perform conventional STDP.
    #                       When there is no winning neuron, reward is '0' -> just perform search.
    #                       When the winning neuron index doesn't match the desired label, reward is '-1' -> perform anti-STDP.
    #       ec_in         - Tensor of input spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       li_out        - Tensor of LI's output spiketimes with shape [batch_size,self.rows,self.cols,self.q,self.p].
    #       weights       - Tensor of synaptic weights with shape [self.rows,self.cols,self.q,self.p].
    #       rvcapture     - BRV for 'capture' case.
    #       rvsearch      - BRV for 'search' case.
    #       rvbackoff     - BRV for 'backoff'and 'minus' cases.
    #       rvmin         - BRV for enforcing a minimum probability of update.
    #       rvstickup     - BRV for sticking the weights towards wmax. Helps in generating a bimodal weight distribution.
    #       rvstickdown   - BRV for sticking the weights towards 0. Helps in generating a bimodal weight distribution.
    # Returns a tensor of RSTDP-updated synaptic weights of shape [self.rows,self.cols,self.q,self.p].

class RSTDP_Stochastic():
    def __init__(self, wres, stochasticity="low"):
        self.wmax           = 2**(wres)-1
        self.stoch          = stochasticity
        
    def __call__(self, reward, ec_in, li_out, weights, rvcapture, rvsearch, rvbackoff, rvmin, rvstickup, rvstickdown):
        batch_size      = ec_in.shape[0]
        weights         = weights.unsqueeze(0).repeat(batch_size,1,1,1,1)
        
        # Low stochasticity - All Bernoulli random variables are shared across the entire column.
        if self.stoch == "low":
            
            if reward == 1:
                # Case 1 (capture)
                weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                               += rvbackoff * torch.max(rvmin, rvstickup \
                                  [weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)].long()])
                
                # Case 2 (minus)
                weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)] \
                               -= rvbackoff * torch.max(rvmin, rvstickdown \
                                  [weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)].long()])
                
                # Case 4 (backoff)
                weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))] \
                               -= rvbackoff * torch.max(rvmin, rvstickdown \
                                  [weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))].long()])
                
            elif reward == 0:
                
                # Case 3 (search)
                weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] \
                               += rvsearch
                
            elif reward == -1:
                # Case 1 (capture)
                weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                               -= rvbackoff * torch.max(rvmin, rvstickdown \
                                  [weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)].long()])

                # Case 3 (search)
                weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] \
                               += rvsearch
                
        # High stochasticity - Each synapse has a separate Bernoulli random variable associated with it.  
        elif self.stoch == "high":
            
            if reward == 1:
                # Case 1 (capture)
                weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                               += rvbackoff[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                                * torch.max(rvmin[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)], \
                                  torch.diagonal(rvstickup[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                                  [:,weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)].long()],0))

                # Case 2 (minus)
                weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)] \
                               -= rvbackoff[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)] \
                                * torch.max(rvmin[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)], \
                                  torch.diagonal(rvstickdown[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)] \
                                  [:,weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in>li_out)].long()],0))

                # Case 4 (backoff)
                weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))] \
                               -= rvbackoff[(ec_in==float('Inf'))*(li_out!=float('Inf'))] \
                                * torch.max(rvmin[(ec_in==float('Inf'))*(li_out!=float('Inf'))], \
                                  torch.diagonal(rvstickdown[(ec_in==float('Inf'))*(li_out!=float('Inf'))] \
                                  [:,weights[(ec_in==float('Inf'))*(li_out!=float('Inf'))].long()],0))
                
            elif reward == 0:

                # Case 3 (search)
                weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] \
                               += rvsearch[(ec_in!=float('Inf'))*(li_out==float('Inf'))]
                
            elif reward == -1:
                # Case 1 (capture)
                weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                               -= rvbackoff[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                                * torch.max(rvmin[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)], \
                                  torch.diagonal(rvstickdown[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)] \
                                  [:,weights[(ec_in!=float('Inf'))*(li_out!=float('Inf'))*(ec_in<=li_out)].long()],0))

                # Case 3 (search)
                weights[(ec_in!=float('Inf'))*(li_out==float('Inf'))] \
                               += rvsearch[(ec_in!=float('Inf'))*(li_out==float('Inf'))]
        
        weights = torch.round(torch.sum(weights,dim=0)/batch_size)
        return nn.Parameter(weights.clamp_(0, self.wmax), requires_grad=False)