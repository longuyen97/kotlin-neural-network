package de.longuyen

import org.nd4j.linalg.factory.Nd4j
import java.io.FileOutputStream
import java.io.ObjectOutputStream


fun main() {
    val x = Nd4j.zeros(4, 4)
    println(x)

    val ranks = x.rank()
    println(ranks)

    val shape = x.shape()
    shape.forEach { print("$it ") }
    println()

    val length = x.length()
    println(length)

    val dt = x.dataType()
    println(dt)

    FileOutputStream("target/test.ser").use{
        val os = ObjectOutputStream(it)
        os.writeObject(x)
    }
}