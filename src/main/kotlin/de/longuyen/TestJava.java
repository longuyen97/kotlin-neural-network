package de.longuyen;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestJava {
    public static void main(String[] args) {
        INDArray output = Nd4j.rand(5, 5);
        INDArray target = Nd4j.rand(5, 5);
        double [][] aOutput = output.toDoubleMatrix();
        double [][] aTarget = target.toDoubleMatrix();
        double [][] gradients = target.toDoubleMatrix();
        for(int y = 0; y < aOutput.length; y++) {
            for(int x = 0; x < aOutput.length; x++){
                if(aOutput[y][x] > aTarget[y][x]){
                    gradients[y][x] = 1;
                }else{
                    gradients[y][x] = -1;
                }
            }
        }
    }
}
