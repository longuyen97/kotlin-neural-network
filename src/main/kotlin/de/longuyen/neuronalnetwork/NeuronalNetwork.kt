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
    private val parameters = mutableMapOf<String, INDArray>()

    /*
     * Bias for each layer of the network
     */
    private val biases = mutableMapOf<String, INDArray>()

    /*
     * Indicate how many hidden layers there are
     */
    private val hiddenCount = layers.size - 1

    init {
        // Initialize each layer of the network with random values
        // TODO implement better algorithms for the initialization's process
        for (i in 1 until layers.size) {
            // i - 1 rows, i columns
            parameters["W$i"] = Nd4j.rand(*intArrayOf(layers[i], layers[i - 1])).castTo(DataType.DOUBLE)
            biases["b$i"] = Nd4j.zeros(*intArrayOf(layers[i], 1)).castTo(DataType.DOUBLE)
        }
    }

    /**
     *
     */
    fun train(X: INDArray, Y: INDArray, x: INDArray, y: INDArray, learningRate: Double, epochs: Long, verbose: Boolean = true): Pair<MutableList<Double>, MutableList<Double>> {
        val validationLosses = mutableListOf<Double>()
        val trainingLosses = mutableListOf<Double>()
        for (epoch in 0 until epochs) {
            val cache = forward(X)
            backward(X, Y, cache, learningRate)
            val yPrediction = inference(x)
            val trainingLoss = lossFunction.forward(Y, cache.second["A$hiddenCount"]!!)
            val validationLoss = lossFunction.forward(y, yPrediction)

            if(verbose) {
                println("Epoch $epoch - Training loss ${(trainingLoss.element() as Double).toInt()} - Validation Loss ${(validationLoss.element() as Double).toInt()}")
            }

            validationLosses.add(validationLoss.element() as Double)
            trainingLosses.add(trainingLoss.element() as Double)
        }

        return Pair(trainingLosses, validationLosses)
    }

    /**
     *
     */
    fun inference(x: INDArray): INDArray {
        return forward(x).second["A$hiddenCount"]!!
    }

    /**
     *
     */
    private fun forward(x: INDArray): Pair<MutableMap<String, INDArray>, MutableMap<String, INDArray>> {
        val logits = mutableMapOf<String, INDArray>()
        val activations = mutableMapOf<String, INDArray>()

        logits["Z1"] = (parameters["W1"]!!.mmul(x)).add(biases["b1"]!!)
        activations["A1"] = hiddenActivation.forward(logits["Z1"]!!)

        for (i in 2 until layers.size - 1) {
            logits["Z$i"] = (parameters["W$i"]!!.mmul(activations["A${i - 1}"])).add(biases["b$i"]!!)
            activations["A$i"] = hiddenActivation.forward(logits["Z$i"]!!)
        }
        logits["Z${layers.size - 1}"] = (parameters["W${layers.size - 1}"]!!.mmul(activations["A${layers.size - 2}"])).add(biases["b${layers.size - 1}"]!!)
        activations["A${layers.size - 1}"] = lastActivation.forward(logits["Z${layers.size - 1}"]!!)

        return Pair(logits, activations)
    }

    /**
     *
     */
    private fun backward(x: INDArray, y: INDArray, cache: Pair<MutableMap<String, INDArray>, MutableMap<String, INDArray>>, learningRate: Double) {
        val logits = cache.first
        val activations = cache.second
        val activationGrads = mutableMapOf<String, INDArray>()
        val logitGrads = mutableMapOf<String, INDArray>()
        val parameterGrads = mutableMapOf<String, INDArray>()
        val biasGrads = mutableMapOf<String, INDArray>()

        logitGrads["dZ$hiddenCount"] = lossFunction.backward(y, activations["A$hiddenCount"]!!)

        for (i in 1 until hiddenCount) {
            val activationGrad =
                (parameters["W${hiddenCount - i + 1}"]!!.transpose()).mmul(logitGrads["dZ${hiddenCount - i + 1}"]!!)
            activationGrads["dA${hiddenCount - i}"] = activationGrad


            val logitGrad = activationGrads["dA${hiddenCount - i}"]!!.mul(hiddenActivation.backward(logits["Z${hiddenCount - i}"]!!))
            logitGrads["dZ${hiddenCount - i}"] = logitGrad
        }

        parameterGrads["dW1"] = logitGrads["dZ1"]!!.mmul(x.transpose())
        biasGrads["db1"] = logitGrads["dZ1"]!!.sum(true, 1)
        for (i in 2 until layers.size) {
            parameterGrads["dW$i"] = logitGrads["dZ$i"]!!.mmul(activations["A${i - 1}"]!!.transpose())
            biasGrads["db$i"] = logitGrads["dZ$i"]!!.sum(true, 1)
        }

        for (i in 2 until layers.size) {
            parameters["W$i"] = parameters["W$i"]!!.sub(parameterGrads["dW$i"]!!.mul(learningRate))
            biases["b$i"] = biases["b$i"]!!.sub(biasGrads["db$i"]!!.mul(learningRate))
        }
    }
}
