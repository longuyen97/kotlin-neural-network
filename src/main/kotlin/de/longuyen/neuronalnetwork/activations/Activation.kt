package de.longuyen.neuronalnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray

interface Activation{
    fun forward(x: INDArray) : INDArray

    fun backward(x: INDArray): INDArray
}