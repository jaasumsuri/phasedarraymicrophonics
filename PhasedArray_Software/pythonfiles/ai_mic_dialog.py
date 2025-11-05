#######################################
#
#      Phased Array Microphonics
#  AI Microphone Configuration Dialog
#
#       Author : Joe Do
#       Date : 11/4/2025
#
#######################################

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSpinBox, QLineEdit, QPushButton, QCheckBox,
                               QGroupBox, QFormLayout, QScrollArea, QWidget)
from PySide6.QtCore import Qt


class AIMicConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Microphone Configurations")
        self.setMinimumSize(500, 400)
        self.setModal(True)
        
        # Store phantom mic positions: {mic_index: (x, y, z)}
        self.phantom_positions = {}
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Enable/Disable checkbox
        self.enable_checkbox = QCheckBox("Enable AI Phantom Microphones")
        self.enable_checkbox.setChecked(False)
        self.enable_checkbox.stateChanged.connect(self.on_enable_changed)
        main_layout.addWidget(self.enable_checkbox)
        
        # Configuration group (initially disabled)
        self.config_group = QGroupBox("Phantom Mic Configuration")
        self.config_group.setEnabled(False)
        config_layout = QVBoxLayout()
        
        # Number of phantom mics
        num_mics_layout = QHBoxLayout()
        num_mics_layout.addWidget(QLabel("Number of Phantom Mics:"))
        self.num_mics_spinbox = QSpinBox()
        self.num_mics_spinbox.setMinimum(1)
        self.num_mics_spinbox.setMaximum(10)
        self.num_mics_spinbox.setValue(1)
        self.num_mics_spinbox.valueChanged.connect(self.update_position_fields)
        num_mics_layout.addWidget(self.num_mics_spinbox)
        num_mics_layout.addStretch()
        config_layout.addLayout(num_mics_layout)
        
        # Scroll area for position fields
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.position_container = QWidget()
        self.position_layout = QFormLayout(self.position_container)
        self.position_layout.setLabelAlignment(Qt.AlignLeft)
        self.scroll_area.setWidget(self.position_container)
        config_layout.addWidget(self.scroll_area)
        
        self.config_group.setLayout(config_layout)
        main_layout.addWidget(self.config_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)
        
        # Initialize position fields
        self.position_fields = {}  # {mic_index: {'x': QLineEdit, 'y': QLineEdit, 'z': QLineEdit}}
        self.update_position_fields()
    
    def on_enable_changed(self, state):
        """Enable/disable the configuration group based on checkbox state"""
        enabled = self.enable_checkbox.isChecked()
        self.config_group.setEnabled(enabled)
        if not enabled:
            # Clear positions when disabled
            self.phantom_positions = {}
    
    def update_position_fields(self):
        """Dynamically create/remove position input fields based on number of phantom mics"""
        num_mics = self.num_mics_spinbox.value()
        
        # Remove excess fields if number decreased
        while len(self.position_fields) > num_mics:
            mic_index = max(self.position_fields.keys())
            if mic_index in self.position_fields:
                # Remove from layout
                fields = self.position_fields[mic_index]
                self.position_layout.removeRow(fields['label'])
                # Delete the label widget
                fields['label'].deleteLater()
                del self.position_fields[mic_index]
                if mic_index in self.phantom_positions:
                    del self.phantom_positions[mic_index]
        
        # Add new fields if number increased
        for mic_index in range(len(self.position_fields), num_mics):
            label = QLabel(f"Phantom Mic #{mic_index + 1} Position:")
            x_edit = QLineEdit()
            x_edit.setPlaceholderText("X")
            y_edit = QLineEdit()
            y_edit.setPlaceholderText("Y")
            z_edit = QLineEdit()
            z_edit.setPlaceholderText("Z")
            
            # Set default values if they exist
            if mic_index in self.phantom_positions:
                x, y, z = self.phantom_positions[mic_index]
                x_edit.setText(str(x))
                y_edit.setText(str(y))
                z_edit.setText(str(z))
            
            # Layout for X, Y, Z inputs
            coords_layout = QHBoxLayout()
            coords_layout.addWidget(QLabel("X:"))
            coords_layout.addWidget(x_edit)
            coords_layout.addWidget(QLabel("Y:"))
            coords_layout.addWidget(y_edit)
            coords_layout.addWidget(QLabel("Z:"))
            coords_layout.addWidget(z_edit)
            coords_layout.addStretch()
            
            self.position_fields[mic_index] = {
                'label': label,
                'x': x_edit,
                'y': y_edit,
                'z': z_edit
            }
            
            # Add to form layout
            self.position_layout.addRow(label, coords_layout)
    
    def get_configuration(self):
        """Returns the current configuration as a dictionary"""
        enabled = self.enable_checkbox.isChecked()
        num_mics = self.num_mics_spinbox.value()
        
        # Collect positions
        positions = []
        if enabled:
            for mic_index in range(num_mics):
                if mic_index in self.position_fields:
                    fields = self.position_fields[mic_index]
                    try:
                        x = float(fields['x'].text()) if fields['x'].text() else 0.0
                        y = float(fields['y'].text()) if fields['y'].text() else 0.0
                        z = float(fields['z'].text()) if fields['z'].text() else 0.0
                        positions.append((x, y, z))
                    except ValueError:
                        positions.append((0.0, 0.0, 0.0))
                else:
                    positions.append((0.0, 0.0, 0.0))
        
        return {
            'enabled': enabled,
            'num_phantom_mics': num_mics if enabled else 0,
            'positions': positions if enabled else []
        }
    
    def set_configuration(self, config):
        """Set the dialog configuration from a dictionary"""
        if config is None:
            return
        
        self.enable_checkbox.setChecked(config.get('enabled', False))
        if config.get('enabled', False):
            num_mics = config.get('num_phantom_mics', 1)
            self.num_mics_spinbox.setValue(num_mics)
            positions = config.get('positions', [])
            for i, pos in enumerate(positions[:num_mics]):
                if i in self.position_fields:
                    x, y, z = pos
                    self.position_fields[i]['x'].setText(str(x))
                    self.position_fields[i]['y'].setText(str(y))
                    self.position_fields[i]['z'].setText(str(z))
