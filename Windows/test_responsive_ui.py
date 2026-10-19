import threading
import tkinter as tk
import unittest
from offline_pipeline import OfflineStudio


class ResponsiveTests(unittest.TestCase):
    def test_selectors_fit_and_commands_remain_inside_window(self):
        root=tk.Tk();root.attributes('-alpha',0)
        try:
            app=OfflineStudio.__new__(OfflineStudio);app.root=root
            app.ip=tk.StringVar();app.status=tk.StringVar(value='Escaneo de objetos')
            app.quality=tk.DoubleVar(value=.05);app.params={};app.cancel=threading.Event()
            app.build_ui()
            for width,height in [(680,520),(800,600),(1024,768),(1380,900)]:
                root.geometry(f'{width}x{height}');root.update()
                for widget in (app.capture_button,app.stop_button,app.result_combo):
                    self.assertGreater(widget.winfo_width(),70)
                    self.assertGreaterEqual(widget.winfo_rootx(),root.winfo_rootx())
                    self.assertLessEqual(widget.winfo_rootx()+widget.winfo_width(),root.winfo_rootx()+width)
                self.assertEqual(app.quality.get(),.05)
                self.assertEqual(app.device.get(),'CUDA')
        finally:root.destroy()


if __name__=='__main__':unittest.main()
