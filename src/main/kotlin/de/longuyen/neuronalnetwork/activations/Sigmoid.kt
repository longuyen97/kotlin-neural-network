package de.longuyen.neuronalnetwork.activations

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

class Sigmoid : Activation(){
    override fun forward(x: INDArray): INDArray {
        return Nd4j.ones(*x.shape()).div(Transforms.exp(x, true).add(1))
    }

    override fun backward(x: INDArray): INDArray {
        val e = this.forward(x)
        return e.mul(Nd4j.ones(*x.shape()).sub(e))
    }
}