package de.longuyen.neuronalnetwork.losses

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

class CrossEntropy : LossFunction() {
    override fun forward(yTrue: INDArray, yPrediction: INDArray): INDArray {
        val loss = (yTrue.mul(Transforms.log(yPrediction)).sum(true, 0)).mul(-1)
        return loss.mean(1)
    }

    override fun backward(yTrue: INDArray, yPrediction: INDArray): INDArray {
        return (yTrue.sub(yPrediction)).div(yTrue.shape()[1])
    }
}