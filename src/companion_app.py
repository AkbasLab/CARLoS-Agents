import tkinter as tk
from tkinter import ttk
from src.companion_layout_selector import LayoutSelectorApp
from src.companion_layout_creator import LayoutCreatorApp
from src.companion_vehicle_editor import VehicleEditorApp
from src.companion_app_help import HelpFrame
import sys

class CARLOSCompanion(tk.Tk):
    def __init__(self, folder_path="src\\layouts", *args, **kwargs):    
        super().__init__(*args, **kwargs)

        self.title("CARLOS Companion App")
        self.folder_path = folder_path

        self.configure_global_font(font_family="Consolas", font_size=9)

        self.menu_frame = tk.Frame(self, background="dark gray")
        self.menu_frame.pack(
            fill=tk.BOTH,
            side=tk.TOP,
            expand=True,
        )

        button_container = tk.Frame(self.menu_frame, background="dark gray")
        button_container.pack(anchor="center")

        self.container = tk.Frame(self)
        self.container.pack(expand=True, fill=tk.BOTH, side=tk.TOP)

        selector_button = tk.Button(
            button_container,
            text="Go to Layout Selector",
            command=lambda: self.show_frame(LayoutSelectorApp),
            width=30,
        )
        selector_button.grid(row=0, column=0, padx=20, pady=10)

        creator_button = tk.Button(
            button_container,
            text="Go to Layout Creator",
            command=lambda: self.show_frame(LayoutCreatorApp),
            width=30,
        )
        creator_button.grid(row=0, column=1, padx=20, pady=10)

        vehicle_button = tk.Button(
            button_container,
            text="Go to Vehicle Editor",
            command=lambda: self.show_frame(VehicleEditorApp),
            width=30,
        )
        vehicle_button.grid(row=0, column=2, padx=20, pady=10)

        help_button = tk.Button(
            button_container,
            text="Help",
            command=lambda: self.show_frame(HelpFrame),
            width=30,
        )
        help_button.grid(row=0, column=3, padx=40, pady=10)

        self.frames = {}
        for F in (LayoutSelectorApp, LayoutCreatorApp, VehicleEditorApp, HelpFrame):
            frame = None
            if F == LayoutSelectorApp:
                frame = F(self.container, folder_path=self.folder_path)
            else:
                frame = F(self.container)
            self.frames[F] = frame
            frame.grid(row=1, column=0, sticky="nsew")

        self.show_frame(LayoutCreatorApp) 
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_global_font(self, font_family="Consolas", font_size=10):
        default_font = (font_family, font_size)
        self.option_add("*Font", default_font)
        style = ttk.Style()
        style.configure("TButton", font=default_font) 
        style.configure("TLabel", font=default_font) 

    def show_frame(self, frame_class):
        frame = self.frames[frame_class]
        if isinstance(frame, LayoutSelectorApp):
            frame.refresh_files()
        frame.tkraise()

    def on_close(self):
        self.quit()  
        self.destroy()  
        sys.exit()  

if __name__ == "__main__":
    app = CARLOSCompanion()
    app.mainloop()
