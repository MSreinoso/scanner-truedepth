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
                # No radio/check button may be wider than the panel that holds it.
                app.ui_mode.set('Avanzado'); app.show_mode(); root.update()
                def walk(widget):
                    for child in widget.winfo_children():
                        yield child; yield from walk(child)
                panel=app.advanced.master
                page,notebook=panel,panel.master
                while notebook.winfo_class()!='TNotebook': page,notebook=notebook,notebook.master
                notebook.select(page); root.update()
                self.assertGreaterEqual(panel.winfo_width(),250,f'{width}x{height}')
                for child in walk(panel):
                    if child.winfo_class() in ('TRadiobutton','TCheckbutton','TButton'):
                        self.assertLessEqual(child.winfo_reqwidth(),panel.winfo_width(),child.cget('text'))
                    if child.winfo_class()=='TLabel' and child.winfo_ismapped():
                        self.assertLessEqual(int(str(child.cget('wraplength')) or 0),panel.winfo_width(),child.cget('text'))
                # Wheel over a field (not the canvas) still scrolls the panel.
                canvas=panel.master; canvas.yview_moveto(0); root.update()
                field=next(c for c in walk(panel) if c.winfo_class()=='TEntry' and c.winfo_ismapped())
                before=canvas.yview()[0]
                field.event_generate('<MouseWheel>',delta=-120,rootx=field.winfo_rootx()+3,rooty=field.winfo_rooty()+3,x=3,y=3,warp=False)
                root.update()
                self.assertGreater(canvas.yview()[0],before,f'{width}x{height}')
                app.ui_mode.set('Simple'); app.show_mode()
                self.assertEqual(app.quality.get(),.05)
                self.assertEqual(app.device.get(),'CUDA')
            # A click must reach the command even while poll() refreshes states.
            app.busy=True;app.set_state(app.cancel_button,'normal');root.update()
            calls=[]
            app.cancel_processing=lambda:calls.append(1)
            app.cancel_button.configure(command=lambda:app.cancel_processing())
            app.cancel_button.event_generate('<Enter>');app.cancel_button.event_generate('<ButtonPress-1>',x=5,y=5)
            app.set_state(app.cancel_button,'normal');root.update()
            app.cancel_button.event_generate('<ButtonRelease-1>',x=5,y=5);root.update()
            self.assertEqual(calls,[1])
        finally:root.destroy()


if __name__=='__main__':unittest.main()
