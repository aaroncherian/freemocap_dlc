import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QSlider, QWidget, \
    QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MainWindow(QMainWindow):
    def __init__(self, mediapipe_data_3d, dlc_data_3d):
        super().__init__()
        self.setWindowTitle('Spliced Data Viewer')

        self.layout = QVBoxLayout()

        plot_widget = ScatterPlot3DWidget(mediapipe_data_3d, dlc_data_3d)
        # plot_widget.set_layout(self.layout)

        self.setCentralWidget(plot_widget)


class ScatterPlot3DWidget(QWidget):
    def __init__(self, mediapipe_data_3d, dlc_data_3d, parent=None):
        super(ScatterPlot3DWidget, self).__init__(parent)
        self.mediapipe_data_3d = mediapipe_data_3d
        self.dlc_data_3d = dlc_data_3d

        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Slider Layout
        self.slider_layout = QHBoxLayout()
        self.layout.addLayout(self.slider_layout)

        # Frame label
        self.frame_label = QLabel("Frame 0")
        self.slider_layout.addWidget(self.frame_label)

        # 3D Scatter Plot
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.layout.addWidget(self.canvas)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(mediapipe_data_3d) - 1)
        self.slider.valueChanged.connect(self.update_plot)
        self.slider_layout.addWidget(self.slider)

        # Axis range
        self.ax_range = 900
        self.mean_x = np.nanmean(mediapipe_data_3d[:, :, 0])
        self.mean_y = np.nanmean(mediapipe_data_3d[:, :, 1])
        self.mean_z = np.nanmean(mediapipe_data_3d[:, :, 2])

    def update_plot(self, value):
        # Clear the current plot
        self.ax.cla()

        # Update frame label
        self.frame_label.setText(f"Frame {value}")

        # Get the data for the selected frame
        mediapipe_frame_data = self.mediapipe_data_3d[value, :, :]
        dlc_frame_data = self.dlc_data_3d[value, :, :]

        # Update the scatter plot with the new data
        self.ax.scatter(mediapipe_frame_data[:, 0], mediapipe_frame_data[:, 1], mediapipe_frame_data[:, 2],
                        color='blue')
        self.ax.scatter(dlc_frame_data[:, 0], dlc_frame_data[:, 1], dlc_frame_data[:, 2], color='red')

        # Set equal axes based on mean for mediapipe data
        self.ax.set_xlim([self.mean_x - self.ax_range, self.mean_x + self.ax_range])
        self.ax.set_ylim([self.mean_y - self.ax_range, self.mean_y + self.ax_range])
        self.ax.set_zlim([self.mean_z - self.ax_range, self.mean_z + self.ax_range])

        # Redraw the plot
        self.canvas.draw()

# if __name__ == '__main__':

#     path_to_spliced_data = r"D:\steen_pantsOn_gait_3_cameras\output_data\mediapipe_postprocessed_3d_xyz.npy"

#     # spliced_data = np.load(path_to_spliced_data)
#     # path_to_spliced_data  = r'D:\steen_pantsOn_gait_3_cameras\output_data\raw_data\mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
#     spliced_data = np.load(path_to_spliced_data)[:,0:25,:]


#     app = QApplication([])
#     win = MainWindow(spliced_data)
#     win.show()
#     app.exec()
