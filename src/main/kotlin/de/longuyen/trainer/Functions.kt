package de.longuyen.trainer

import org.nd4j.linalg.activations.impl.ActivationReLU
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

fun relu(x: INDArray): INDArray {
    return ActivationReLU().getActivation(x, false).castTo(x.dataType())
}

fun dRelu(x: INDArray) : INDArray {
    val input: Array<DoubleArray> = x.toDoubleMatrix()
    for(yi in input.indices){
        for(xi in input[yi].indices){
            if(input[yi][xi] > 0){
                input[yi][xi] = 1.0
            }else{
                input[yi][xi] = 0.0
            }
        }
    }
    return Nd4j.createFromArray(input).castTo(x.dataType())
}

fun mae(yTrue: INDArray, yPrediction: INDArray) : INDArray {
    val diff = yPrediction.sub(yTrue)
    val absolute = Transforms.abs(diff)
    return absolute.sum(true, 0).mean(true, 0).castTo(yTrue.dataType())
}

fun dMae(yTrue: INDArray, yPrediction: INDArray) : INDArray {
    val aOutput: Array<DoubleArray> = yPrediction.toDoubleMatrix()
    val aTarget: Array<DoubleArray> = yTrue.toDoubleMatrix()
    val gradients: Array<DoubleArray> = yTrue.toDoubleMatrix()
    for (y in aOutput.indices) {
        for (x in aOutput[y].indices) {
            if (aOutput[y][x] > aTarget[y][x]) {
                gradients[y][x] = 1.0
            } else {
                gradients[y][x] = -1.0
            }
        }
    }
    return Nd4j.createFromArray(gradients).castTo(yTrue.dataType())
}






