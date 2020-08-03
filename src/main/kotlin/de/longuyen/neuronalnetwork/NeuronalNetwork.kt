package de.longuyen.neuronalnetwork

import org.nd4j.linalg.api.ndarray.INDArray

abstract class NeuronalNetwork {
    /**
     * Adjusting model's parameters with the given data set.
     * @param X training features
     * @param Y training target
     * @param x validating features
     * @param y validating target
     * @param epochs how many epochs should the training take
     * @param verbose should the model print the current metric?
     * @param batchSize for faster training
     * @return history of the training with "val-loss", "train-loss", "val-metric", "train-metric"
     */
    abstract fun train(
        X: INDArray,
        Y: INDArray,
        x: INDArray,
        y: INDArray,
        epochs: Long,
        batchSize: Long,
        verbose: Boolean = true
    ): Map<String, DoubleArray>


    /**
     * Adjusting model's parameters with the given data set.
     * @param X training features
     * @param Y training target
     * @param x validating features
     * @param y validating target
     * @param epochs how many epochs should the training take
     * @param verbose should the model print the current metric?
     * @return history of the training with "val-loss", "train-loss", "val-metric", "train-metric"
     */
    abstract fun train(
        X: INDArray,
        Y: INDArray,
        x: INDArray,
        y: INDArray,
        epochs: Long,
        verbose: Boolean = true
    ): Map<String, DoubleArray>


    /**
     * Forward propagation and return the output of the last layer
     * @param x features
     * @return last output of the network
     */
    abstract fun inference(x: INDArray): INDArray
}