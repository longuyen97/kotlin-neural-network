package de.longuyen.ui;

import javax.swing.*;

public class Paint {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new Draw(500, 500));
    }
}
