package de.longuyen

import de.longuyen.trainer.dMae
import de.longuyen.trainer.dRelu
import de.longuyen.trainer.mae
import de.longuyen.trainer.relu
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun main() {
    val layers = longArrayOf(10, 5, 2, 1)

    val parameters = mutableMapOf<String, INDArray>()
    val biases = mutableMapOf<String, INDArray>()
    val features = Nd4j.rand(*longArrayOf(10, 2))
    val target = Nd4j.rand(*longArrayOf(1, 2))
    for (i in 1 until layers.size) {
        // i - 1 rows, i columns
        parameters["W$i"] = Nd4j.rand(*longArrayOf(layers[i], layers[i - 1]))
        biases["b$i"] = Nd4j.zeros(*longArrayOf(layers[i], 1))
    }

    val hiddens = layers.size - 1

    val logits = mutableMapOf<String, INDArray>()
    val activations = mutableMapOf<String, INDArray>()
    val activationGrads = mutableMapOf<String, INDArray>()
    val logitGrads = mutableMapOf<String, INDArray>()
    val parameterGrads = mutableMapOf<String, INDArray>()
    val biasGrads = mutableMapOf<String, INDArray>()


    for (epoch in 0..10) {
        logits["Z1"] = (parameters["W1"]!!.mmul(features)).add(biases["b1"]!!)
        activations["A1"] = relu(logits["Z1"]!!)

        for (i in 2 until layers.size - 1) {
            logits["Z$i"] = (parameters["W$i"]!!.mmul(activations["A${i - 1}"])).add(biases["b$i"]!!)
            activations["A$i"] = relu(logits["Z$i"]!!)
        }
        logits["Z${layers.size - 1}"] = (parameters["W${layers.size - 1}"]!!.mmul(activations["A${layers.size - 2}"])).add(biases["b${layers.size - 1}"]!!)
        activations["A${layers.size - 1}"] = logits["Z${layers.size - 1}"]!!.dup()

        logitGrads["dZ$hiddens"] = dMae(target, activations["A$hiddens"]!!)
        val loss = mae(target, activations["A$hiddens"]!!)

        for (i in 1 until hiddens) {
            val activationGrad =
                (parameters["W${hiddens - i + 1}"]!!.transpose()).mmul(logitGrads["dZ${hiddens - i + 1}"]!!)
            activationGrads["dA${hiddens - i}"] = activationGrad


            val logitGrad = activationGrads["dA${hiddens - i}"]!!.mul(dRelu(logits["Z${hiddens - i}"]!!))
            logitGrads["dZ${hiddens - i}"] = logitGrad
        }

        parameterGrads["dW1"] = logitGrads["dZ1"]!!.mmul(features.transpose())
        biasGrads["db1"] = logitGrads["dZ1"]!!.sum(true, 1)
        for(i in 2 until layers.size){
            parameterGrads["dW$i"] = logitGrads["dZ$i"]!!.mmul(activations["A${i - 1}"]!!.transpose())
            biasGrads["db$i"] = logitGrads["dZ$i"]!!.sum(true, 1)
        }

        val learningRate = 0.001
        for(i in 2 until layers.size) {
            parameters["W$i"] = parameters["W$i"]!!.sub(parameterGrads["dW$i"]!!.mul(learningRate))
            biases["b$i"] = biases["b$i"]!!.sub(biasGrads["db$i"]!!.mul(learningRate))
        }

        println("Epoch $epoch - Loss $loss")
    }
}








