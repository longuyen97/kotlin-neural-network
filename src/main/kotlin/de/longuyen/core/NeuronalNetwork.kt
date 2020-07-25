package de.longuyen.core

import org.nd4j.linalg.api.ndarray.INDArray

class NeuronalNetwork : Model() {
    override fun train(trainingData: Pair<INDArray, INDArray>,  hyperparameters: Map<String, Double>): MutableList<Double> {
        TODO("Not yet implemented")
    }

    override fun forwardPropagate(x: INDArray): Map<String, INDArray> {
        TODO("Not yet implemented")
    }

    override fun backwardPropagate(x: INDArray, y: INDArray, caches: Map<String, INDArray>) {
        TODO("Not yet implemented")
    }

}