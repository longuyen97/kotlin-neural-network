package de.longuyen.gui

import java.awt.*
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.awt.event.MouseMotionAdapter
import java.awt.image.BufferedImage
import java.awt.image.ColorModel
import java.awt.image.WritableRaster
import javax.swing.JComponent


class View : JComponent() {
    private var x1 = 0
    private var y1 = 0
    private var x2 = 0
    private var y2 = 0
    private val globalGraphics: Graphics2D
    private val image = BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB)

    fun getImage() : BufferedImage {
        val cm: ColorModel = image.colorModel
        val isAlphaPremultiplied = cm.isAlphaPremultiplied
        val raster: WritableRaster = image.copyData(null)
        return BufferedImage(cm, raster, isAlphaPremultiplied, null)
    }

    override fun paintComponent(g1: Graphics) {
        g1.drawImage(image, 0, 0, null)
    }

    fun clear() {
        globalGraphics.paint = Color.white
        globalGraphics.fillRect(0, 0, 500, 500)
        globalGraphics.paint = Color.black
        repaint()
    }

    fun setImage(image: BufferedImage){
        this.image.data = image.data
        repaint()
    }

    fun setThickness(thick: Int) {
        globalGraphics.stroke = BasicStroke(thick.toFloat())
    }

    init {
        background = Color.WHITE
        isDoubleBuffered = false
        globalGraphics = image.graphics as Graphics2D
        globalGraphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        clear()
        this.print(globalGraphics)
        addMouseListener(object : MouseAdapter() {
            override fun mousePressed(e: MouseEvent) {
                x2 = e.x
                y2 = e.y
            }
        })
        addMouseMotionListener(object : MouseMotionAdapter() {
            override fun mouseDragged(e: MouseEvent) {
                x1 = e.x
                y1 = e.y
                globalGraphics.drawLine(x2, y2, x1, y1)
                repaint()
                x2 = x1
                y2 = y1
            }
        })
    }
}