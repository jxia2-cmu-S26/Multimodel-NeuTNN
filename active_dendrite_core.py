"""
Core NeuTNN active dendrite layer (SNL).

Extracted from activeDendrite_snl.py so IMU/EEG experiments can import the layer
without pulling in MNIST loaders or training scripts.
"""

import torch
import torch.nn as nn


class activeDendriteLayer(nn.Module):
    """
    Single NeuTNN layer: multiple neurons, dendrites, segments; distal segments;
    STDP / R-STDP via update().

    Input to __call__: data of shape [numNeurons, numDistalInputs].
    """

    def __init__(
        self,
        numDistalInputs: int,
        numSegments: int,
        numDendrites: int,
        numNeurons: int,
        threshold: int,
        wMax: int = 8,
        wInit: str = "uniform",
        rewardEn: bool = True,
        device="cpu",
    ):
        super(activeDendriteLayer, self).__init__()

        self.numDistalInputs = numDistalInputs
        self.numSegments = numSegments
        self.numDendrites = numDendrites
        self.numNeurons = numNeurons
        self.threshold = threshold
        self.wMax = wMax
        self.wInit = wInit
        self.rewardEn = rewardEn
        self.device = device

        self.totalSegments = self.numNeurons * self.numDendrites * self.numSegments

        if self.wInit == "zero":
            self.weights = nn.Parameter(
                torch.zeros(self.totalSegments, self.numDistalInputs), requires_grad=False
            )
        elif self.wInit == "uniform":
            self.weights = nn.Parameter(
                torch.randint(
                    low=0,
                    high=self.wMax + 1,
                    size=(self.totalSegments, self.numDistalInputs),
                ).type(torch.FloatTensor),
                requires_grad=False,
            )
        elif self.wInit == "normal":
            self.weights = nn.Parameter(
                torch.round(
                    ((self.wMax + 1) / 2 + torch.randn(self.totalSegments, self.numDistalInputs)).clamp_(
                        0, self.wMax
                    )
                ),
                requires_grad=False,
            )

    def __call__(self, data):
        inputData = (
            data.unsqueeze(1)
            .repeat(1, self.numDendrites * self.numSegments, 1)
            .reshape(self.totalSegments, self.numDistalInputs)
        )

        weightedInput = inputData * self.weights
        summedOutput = torch.sum(weightedInput, dim=1)

        zeroCondition = summedOutput < self.threshold
        summedOutput[zeroCondition] = 0

        reshapedSummedOutput = summedOutput.reshape(-1, self.numSegments)

        self.wtaOutput = torch.zeros(
            self.numNeurons * self.numDendrites,
            self.numSegments,
            dtype=reshapedSummedOutput.dtype,
            device=self.device,
        )

        maxval, indices = torch.max(reshapedSummedOutput, dim=1)
        self.wtaOutput.scatter_(1, indices.unsqueeze(1), maxval.unsqueeze(1))

        outputData = (self.wtaOutput > 0).reshape(-1).unsqueeze(1).repeat(1, self.numDistalInputs)

        return inputData, outputData

    def update(self, inputData, inputLabels, outputData, weights, capture, search, backoff, wBase, rewardEn, frac=0.25):
        if rewardEn:
            captureVal = (
                (inputData == 1) * (outputData == 1) * (inputLabels == 1) * capture
                + (inputData == 0) * (outputData == 1) * (inputLabels == 0) * capture * frac
            )
            backoffVal = (
                (inputData == 0) * (outputData == 1) * (inputLabels == 1) * backoff
                + (inputData == 1) * (outputData == 1) * (inputLabels == 0) * backoff * frac
            )
            searchCondition = (inputData == 1) * (outputData == 0) * (inputLabels == 1)
        else:
            captureVal = (inputData == 1) * (outputData == 1) * capture
            backoffVal = (inputData == 0) * (outputData == 1) * backoff
            searchCondition = (inputData == 1) * (outputData == 0)

        weights += captureVal - backoffVal

        weights[searchCondition] = torch.maximum(
            weights[searchCondition],
            torch.minimum(
                weights[searchCondition] + search,
                torch.tensor([wBase], device=self.device),
            ),
        )

        return weights.clamp_(0, self.wMax)
