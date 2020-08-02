package de.longuyen.gui

import de.longuyen.neuronalnetwork.NeuronalNetwork
import de.longuyen.neuronalnetwork.activations.LeakyRelu
import de.longuyen.neuronalnetwork.activations.Softmax
import de.longuyen.neuronalnetwork.initializers.ChainInitializer
import de.longuyen.neuronalnetwork.losses.CrossEntropy
import de.longuyen.neuronalnetwork.optimizers.MomentumGradientDescent
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import java.awt.BorderLayout
import java.awt.Color
import java.awt.Dimension
import java.awt.Image
import java.awt.event.ActionEvent
import java.awt.event.ActionListener
import java.awt.image.BufferedImage
import java.io.ObjectInputStream
import java.util.*
import javax.swing.*
import javax.swing.event.ChangeEvent
import javax.swing.event.ChangeListener


fun resize(img: BufferedImage, newW: Int, newH: Int): BufferedImage {
    val tmp: Image = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH)
    val dimg = BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB)
    val g2d = dimg.createGraphics()
    g2d.drawImage(tmp, 0, 0, null)
    g2d.dispose()
    return dimg
}

class Controller(width: Int, height: Int) {
    private val view = View()
    private val clearButton = JButton("Clear")
    private val processButton = JButton("Process")
    private val thicknessStat = JLabel("10")
    private val resultLabel = JLabel("", SwingConstants.CENTER)
    private val thicknessSlider = JSlider(JSlider.HORIZONTAL, 10, 50, 10)
    private lateinit var model: NeuronalNetwork

    init {
        ObjectInputStream(this.javaClass.getResourceAsStream("/models/neuronalnetwork.ser")).use {
            model = it.readObject() as NeuronalNetwork
        }
        val mainFrame = JFrame("Neuronal network interaction")
        val mainFrameContainer = mainFrame.contentPane
        mainFrameContainer.layout = BorderLayout()
        mainFrameContainer.add(view, BorderLayout.CENTER)
        mainFrameContainer.add(resultLabel, BorderLayout.SOUTH)
        val panel = JPanel()
        thicknessSlider.majorTickSpacing = 25
        thicknessSlider.paintTicks = true
        thicknessSlider.preferredSize = Dimension(40, 40)
        val thicknessChangeListener = ChangeListener { e: ChangeEvent? ->
            thicknessStat.text = String.format("%s", thicknessSlider.value)
            view.setThickness(thicknessSlider.value)
        }
        thicknessSlider.addChangeListener(thicknessChangeListener)
        val mouseClickedActionListener = ActionListener { event: ActionEvent ->
            if (event.source === clearButton) {
                view.clear()
            } else if (event.source === processButton) {
                val image = resize(view.getImage(), 28, 28)
                val nativeImageMatrix = arrayOfNulls<DoubleArray>(28)
                for (y in 0 until image.height) {
                    val doubleArray = DoubleArray(28)
                    for (x in 0 until image.width) {
                        val color = Color(image.getRGB(x, y))
                        doubleArray[x] = 255.0 - ((color.red + color.green + color.blue) / 3.0)
                    }
                    nativeImageMatrix[y] = doubleArray
                }
                val ndarray =
                    ((Nd4j.createFromArray(nativeImageMatrix).reshape(intArrayOf(784, 1)).castTo(DataType.DOUBLE)).div(
                        255.0
                    )).castTo(DataType.DOUBLE)
                val result = Nd4j.argMax(model.inference(ndarray), 0)
                println(result)
            }
        }
        processButton.addActionListener(mouseClickedActionListener)
        clearButton.addActionListener(mouseClickedActionListener)
        panel.add(processButton)
        panel.add(clearButton)
        panel.add(thicknessSlider)
        panel.add(thicknessStat)
        mainFrameContainer.add(panel, BorderLayout.NORTH)
        mainFrame.isVisible = true
        mainFrame.setSize(width, height + 11)
        mainFrame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    }
}
