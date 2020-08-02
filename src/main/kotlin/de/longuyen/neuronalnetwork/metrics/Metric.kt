package de.longuyen.neuronalnetwork.metrics

import org.nd4j.linalg.api.ndarray.INDArray

abstract class Metric {
    abstract fun compute(yTrue: INDArray, yPrediction: INDArray) : Double;
}