package de.longuyen.neuronalnetwork

import de.longuyen.neuronalnetwork.activations.Activation
import de.longuyen.neuronalnetwork.losses.LossFunction
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable


/**
 * Vectorized deep neuronal network for supervised machine learning.
 *
 * @param layers Notate how many input each layer of the network will become. The first element of the array is the number
 * of the dataset's features. The last element of the array can be interpreted as the output of the network.
 *
 * @param hiddenActivation Each output of the hidden layers will be scaled with this function
 *
 * @param lastActivation Last output of the last hidden layer will be scaled with this function
 *
 * @param lossFunction Determined how the result of the network should be evaluated against a target.
 */
class NeuronalNetwork(private val layers: IntArray, private val hiddenActivation: Activation, private val lastActivation: Activation, private val lossFunction: LossFunction) : Serializable {
    /*
     * Intern parameters of the network
     */
    private val weights = mutableMapOf<String, INDArray>()

    /*
     * Indicate how many hidden layers there are
     */
    private val hiddenCount = layers.size - 1

    init {
        // Initialize each layer of the network with random values
        // TODO implement better algorithms for the initialization's process
        for (i in 1 until layers.size) {
            // i - 1 rows, i columns
            // Initialize weights
            weights["W$i"] = Nd4j.rand(*intArrayOf(layers[i], layers[i - 1])).castTo(DataType.DOUBLE)
            // Initialize biases
            weights["b$i"] = Nd4j.zeros(*intArrayOf(layers[i], 1)).castTo(DataType.DOUBLE)
        }
    }

    /**
     * Adjusting model's parameters with the given data set.
     * @param X training features
     * @param Y training target
     * @param x validating features
     * @param y validating target
     */
    fun train(X: INDArray, Y: INDArray, x: INDArray, y: INDArray, learningRate: Double, epochs: Long, verbose: Boolean = true): Pair<MutableList<Double>, MutableList<Double>> {
        val validationLosses = mutableListOf<Double>()
        val trainingLosses = mutableListOf<Double>()

        for (epoch in 0 until epochs) {
            // Forward propagation
            val cache = forward(X)

            // Backward propagation
            backward(X, Y, cache, learningRate)

            // Calculate losses on training and validating dataset
            val yPrediction = inference(x)
            val trainingLoss = lossFunction.forward(yTrue = Y, yPrediction = cache["A$hiddenCount"]!!)
            val validationLoss = lossFunction.forward(yTrue = y, yPrediction = yPrediction)

            if(verbose) {
                println("Epoch $epoch - Training loss ${(trainingLoss.element() as Double).toInt()} - Validation Loss ${(validationLoss.element() as Double).toInt()}")
            }

            validationLosses.add(validationLoss.element() as Double)
            trainingLosses.add(trainingLoss.element() as Double)
        }

        return Pair(trainingLosses, validationLosses)
    }

    /**
     * Forward propagation and return the output of the last layer
     */
    fun inference(x: INDArray): INDArray {
        return forward(x)["A$hiddenCount"]!!
    }

    /**
     * Forward propagation
     * @param x features
     * return the output of each layer and its corresponding scaled version
     */
    private fun forward(x: INDArray): MutableMap<String, INDArray> {
        val cache = mutableMapOf<String, INDArray>()

        // Calculate the output of the first layer manually
        cache["Z1"] = (weights["W1"]!!.mmul(x)).add(weights["b1"]!!)
        cache["A1"] = hiddenActivation.forward(cache["Z1"]!!)

        // Calculate the output of the next hidden layers atuomatically
        for (i in 2 until layers.size - 1) {
            cache["Z$i"] = (weights["W$i"]!!.mmul(cache["A${i - 1}"])).add(weights["b$i"]!!)
            cache["A$i"] = hiddenActivation.forward(cache["Z$i"]!!)
        }

        // Calculate the output of the last layer manually so I scale the output with another activation function
        cache["Z${layers.size - 1}"] = (weights["W${layers.size - 1}"]!!.mmul(cache["A${layers.size - 2}"])).add(weights["b${layers.size - 1}"]!!)
        cache["A${layers.size - 1}"] = lastActivation.forward(cache["Z${layers.size - 1}"]!!)

        return cache
    }

    /**
     * Backward propagation
     * @param x training features
     * @param y training target
     * @param cache the cached output of each layer, unscaled and scaled
     * @param learningRate the learning rate for gradient descent
     */
    private fun backward(x: INDArray, y: INDArray, cache: MutableMap<String, INDArray>, learningRate: Double) {
        val grads = mutableMapOf<String, INDArray>()

        // Calculate gradients of the loss respected to prediction
        grads["dZ$hiddenCount"] = lossFunction.backward(yTrue = y, yPrediction = cache["A$hiddenCount"]!!)

        // Calculate the gradients of loss respected to the output and the activated output
        for (i in 1 until hiddenCount) {
            grads["dA${hiddenCount - i}"] = (weights["W${hiddenCount - i + 1}"]!!.transpose()).mmul(grads["dZ${hiddenCount - i + 1}"]!!)
            grads["dZ${hiddenCount - i}"] = grads["dA${hiddenCount - i}"]!!.mul(hiddenActivation.backward(cache["Z${hiddenCount - i}"]!!))
        }

        // Calculate the gradients of loss respected to weights and biases
        grads["dW1"] = grads["dZ1"]!!.mmul(x.transpose())
        grads["db1"] = grads["dZ1"]!!.sum(true, 1)
        for (i in 2 until layers.size) {
            grads["dW$i"] = grads["dZ$i"]!!.mmul(cache["A${i - 1}"]!!.transpose())
            grads["db$i"] = grads["dZ$i"]!!.sum(true, 1)
        }

        // Adjusting parameters respected to the gradients
        for (i in 2 until layers.size) {
            weights["W$i"] = weights["W$i"]!!.sub(grads["dW$i"]!!.mul(learningRate))
            weights["b$i"] = weights["b$i"]!!.sub(grads["db$i"]!!.mul(learningRate))
        }
    }
}
