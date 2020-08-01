package de.longuyen.gui

import javax.swing.SwingUtilities
import javax.swing.UIManager
import javax.swing.UnsupportedLookAndFeelException


@Throws(
    ClassNotFoundException::class,
    UnsupportedLookAndFeelException::class,
    InstantiationException::class,
    IllegalAccessException::class
)
fun main() {
    for (info in UIManager.getInstalledLookAndFeels()) {
        if ("Nimbus" == info.name) {
            UIManager.setLookAndFeel(info.className)
            break
        }
    }
    SwingUtilities.invokeLater { Controller(500, 500) }
}
