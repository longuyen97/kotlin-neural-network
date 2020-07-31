package de.longuyen.neuronalnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable

class NoActivation : Activation(), Serializable {
    override fun forward(x: INDArray): INDArray {
        return x.dup()
    }

    override fun backward(x: INDArray): INDArray {
        return Nd4j.ones(*x.shape())
    }
}