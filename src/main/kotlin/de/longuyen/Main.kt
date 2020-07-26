package de.longuyen

import de.longuyen.trainer.dMae
import de.longuyen.trainer.dRelu
import de.longuyen.trainer.relu
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun main() {
    val layers = longArrayOf(10, 5, 2, 1)

    val parameters = mutableMapOf<String, INDArray>()
    val biases = mutableMapOf<String, INDArray>()
    for (i in 1 until layers.size){
        // i - 1 rows, i columns
        parameters["W$i"] = Nd4j.rand(*longArrayOf(layers[i], layers[i - 1]))
        biases["b$i"] = Nd4j.zeros(*longArrayOf(layers[i], 1))
    }
    val logits = mutableMapOf<String, INDArray>()
    val activations = mutableMapOf<String, INDArray>()

    val features = Nd4j.rand(*longArrayOf(10, 2))
    val target = Nd4j.rand(*longArrayOf(1, 2))
    logits["Z1"] = (parameters["W1"]!!.mmul(features))
    activations["A1"] = relu(logits["Z1"]!!)

    for(i in 2 until layers.size - 1){
        logits["Z$i"] = (parameters["W$i"]!!.mmul(activations["A${i - 1}"])).add(biases["b$i"]!!)
        activations["A$i"] = relu(logits["Z$i"]!!)
    }
    logits["Z${layers.size - 1}"] = (parameters["W${layers.size - 1}"]!!.mmul(activations["A${layers.size - 2}"])).add(biases["b${layers.size - 1}"]!!)
    activations["A${layers.size - 1}"] = logits["Z${layers.size - 1}"]!!.dup()

    val activationGrads = mutableMapOf<String, INDArray>()
    val logitGrads = mutableMapOf<String, INDArray>()

    logitGrads["dZ${layers.size - 1}"] = dMae(target, activations["A${layers.size - 1}"]!!)

    for(i in 1 until layers.size - 1){
        val activationGrad = (parameters["W${layers.size - 1}"]!!.transpose()).mmul(logitGrads["dZ${layers.size - i}"]!!)
        activationGrads["dA${layers.size - i - 1}"] = activationGrad


        val logitGrad = activationGrads["dA${layers.size - 1 - i}"]!!.mmul(dRelu(logits["Z${layers.size - 1 - i}"]!!))
        logitGrads["dZ${layers.size - 1 - i}"] = logitGrad
    }
}