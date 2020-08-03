package de.longuyen.gui

import java.awt.Image
import java.awt.image.BufferedImage


fun resize(img: BufferedImage, newW: Int, newH: Int): BufferedImage {
    val tmp: Image = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH)
    val dimg = BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB)
    val g2d = dimg.createGraphics()
    g2d.drawImage(tmp, 0, 0, null)
    g2d.dispose()
    return dimg
}