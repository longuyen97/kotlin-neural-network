package de.longuyen.core

import org.nd4j.linalg.api.ndarray.INDArray

abstract class Model {
    abstract fun train(trainingData : Pair<INDArray, INDArray>, hyperparameters: Map<String, Double>) : MutableList<Double>
    abstract fun forwardPropagate(x: INDArray) : Map<String, INDArray>
    abstract fun backwardPropagate(x: INDArray, y: INDArray, caches: Map<String, INDArray>)
}