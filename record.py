#!/usr/bin/env python

import numpy as np
import os
import shutil
import mss
import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigCanvas

from PIL import ImageTk, Image

import sys
from pynput import keyboard  # For global keyboard input

# -------------
# Configurable Settings
# -------------
# Polling interval (ms) when not recording
time_idle_ms   = 10
# Polling interval (ms) when recording
record_interval_ms = 10
# Enable keypress markers on plot
enable_key_markers = True

IMAGE_SIZE = (320, 240)
IMAGE_TYPE = ".png"

PY3_OR_LATER = sys.version_info[0] >= 3

if PY3_OR_LATER:
    import tkinter as tk
    import tkinter.ttk as ttk
    import tkinter.messagebox as tkMessageBox
else:
    import Tkinter as tk
    import ttk
    import tkMessageBox

from utils import Screenshot

class MainWindow:
    """ Main frame of the application """

    def __init__(self):
        self.root = tk.Tk()
        self.sct = mss.mss()

        self.root.title('Data Acquisition')
        self.root.geometry("660x325")
        self.root.resizable(False, False)

        # Keyboard state (0/1)
        self.keyboard_state = {'forward':0, 'left':0, 'right':0, 'brake':0}

        # Start global keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.keyboard_listener.start()

        # GUI setup
        self.create_main_panel()

        # Timer settings
        self.idle_rate   = time_idle_ms
        self.sample_rate = record_interval_ms
        self.rate        = self.idle_rate
        self.recording   = False
        self.t           = 0
        self.pause_timer = False

        # Start loop
        self.on_timer()
        self.root.mainloop()

    def create_main_panel(self):
        top_half = tk.Frame(self.root)
        top_half.pack(side=tk.TOP, expand=True, padx=5, pady=5)
        tk.Label(self.root, text="(UI paused while recording)").pack(side=tk.TOP, padx=5)

        bottom_half = tk.Frame(self.root)
        bottom_half.pack(side=tk.LEFT, padx=5, pady=10)

        # Image panel
        self.img_panel = tk.Label(top_half, image=ImageTk.PhotoImage('RGB', size=IMAGE_SIZE))
        self.img_panel.pack(side=tk.LEFT, padx=5)

        # Plot
        self.init_plot()
        self.PlotCanvas = FigCanvas(self.fig, master=top_half)
        self.PlotCanvas.get_tk_widget().pack(side=tk.RIGHT, padx=5)

        # Directory entry + button
        textframe = tk.Frame(bottom_half, width=332, height=15)
        textframe.pack(side=tk.LEFT)
        textframe.pack_propagate(0)

        self.outputDirVar = tk.StringVar()
        self.txt_outputDir = tk.Entry(textframe, textvariable=self.outputDirVar, width=100)
        self.txt_outputDir.pack(side=tk.LEFT)
        self.outputDirVar.set('samples/' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

        self.record_btn = ttk.Button(bottom_half, text='Record', command=self.on_btn_record)
        self.record_btn.pack(side=tk.LEFT, padx=5)

    def init_plot(self):
        self.plotMem  = 50
        self.plotData = [[0]*4 for _ in range(self.plotMem)]
        self.fig  = Figure(figsize=(4,3), dpi=80)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_ylim(0,1)

    def on_timer(self):
        self.poll()
        if not self.recording:
            self.draw()
        if not self.pause_timer:
            self.root.after(self.rate, self.on_timer)

    def poll(self):
        self.img = self.take_screenshot()
        if self.recording:
            self.save_data()
            self.t += 1
        self.update_plot()

    def take_screenshot(self):
        sct_img = self.sct.grab({
            'top':Screenshot.OFFSET_Y,
            'left':Screenshot.OFFSET_X,
            'width':Screenshot.SRC_W,
            'height':Screenshot.SRC_H
        })
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

    def update_plot(self):
        self.plotData.append(list(self.keyboard_state.values()))
        self.plotData.pop(0)

    def save_data(self):
        os.makedirs(self.outputDir, exist_ok=True)
        img_file = f"{self.outputDir}/img_{self.t}{IMAGE_TYPE}"
        self.img.save(img_file)
        line = img_file + ',' + ','.join(map(str,self.keyboard_state.values())) + '\n'
        self.outfile.write(line)
        self.outfile.flush()

    def draw(self):
        # show image
        self.img.thumbnail(IMAGE_SIZE, Image.ANTIALIAS)
        self.img_panel.img = ImageTk.PhotoImage(self.img)
        self.img_panel['image'] = self.img_panel.img

        # plot state lines
        x = np.asarray(self.plotData)
        self.axes.clear()
        cols = ['r','b','g','k']
        labels = ['Forward','Left','Right','Brake']
        for i,c,l in zip(range(4), cols, labels):
            self.axes.plot(range(self.plotMem), x[:,i], color=c, label=l)

        # overlay keypress markers
        if enable_key_markers:
            diffs = np.diff(x, axis=0, prepend=[[0,0,0,0]])
            for i,color in enumerate(cols):
                presses = np.where(diffs[:,i] == 1)[0]
                for idx in presses:
                    self.axes.axvline(idx, color=color, linestyle='--', alpha=0.5)

        self.axes.legend()
        self.PlotCanvas.draw()

    def on_btn_record(self):
        self.pause_timer = True
        if self.recording:
            self.recording = False
            self.record_btn['text'] = 'Record'
            self.rate = self.idle_rate
            self.outfile.close()
        else:
            if self.start_recording():
                self.recording = True
                self.t = 0
                self.record_btn['text'] = 'Stop'
                self.rate = self.sample_rate
                os.makedirs(self.outputDir, exist_ok=True)
                self.outfile = open(f"{self.outputDir}/data.csv", 'a')
        self.pause_timer = False
        self.on_timer()

    def start_recording(self):
        if not self.outputDirVar.get():
            tkMessageBox.showerror('Error','Specify Output Directory', parent=self.root)
            return False
        self.outputDir = self.outputDirVar.get()
        if os.path.exists(self.outputDir):
            if tkMessageBox.askyesno('Warning','Overwrite existing?', parent=self.root): shutil.rmtree(self.outputDir)
            else:
                self.txt_outputDir.focus_set()
                return False
        return True

    def on_key_press(self, key):
        try:
            if key==keyboard.Key.shift:  self.keyboard_state['forward']=1
            elif key==keyboard.Key.left: self.keyboard_state['left']=1
            elif key==keyboard.Key.right:self.keyboard_state['right']=1
            elif key==keyboard.Key.ctrl: self.keyboard_state['brake']=1
        except: pass

    def on_key_release(self, key):
        try:
            if key==keyboard.Key.shift:  self.keyboard_state['forward']=0
            elif key==keyboard.Key.left: self.keyboard_state['left']=0
            elif key==keyboard.Key.right:self.keyboard_state['right']=0
            elif key==keyboard.Key.ctrl: self.keyboard_state['brake']=0
        except: pass

if __name__=='__main__':
    MainWindow()
