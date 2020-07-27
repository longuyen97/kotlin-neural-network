package de.longuyen.neuronalnetwork.losses

import org.nd4j.linalg.api.ndarray.INDArray

interface LossFunction {
    fun forward(yTrue: INDArray, yPrediction: INDArray) : INDArray

    fun backward(yTrue: INDArray, yPrediction: INDArray) : INDArray
}