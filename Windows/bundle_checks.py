"""Packaged executable diagnostics without opening network listeners."""
def run(destination):
    import json,sys,traceback
    from pathlib import Path
    result={}
    try:
        import torch,numpy as np
        from gpu_backend import cuda_module,CudaTile
        from neural_depth import DepthDenoiser
        from ai_alignment import NeuralMatcher
        cp=cuda_module();result['cuda']=cp.cuda.runtime.getDeviceCount()
        depth=np.full((32,32),.3,np.float32);rgb=np.full((32,32,3),128,np.uint8)
        tile=CudaTile(16,.001,[-.008,-.008,.292])
        for _ in range(3):tile.integrate(depth,rgb,[120,120,16,16],np.eye(4))
        vertices,_=tile.mesh(2);assert len(vertices)>0
        np.testing.assert_allclose(vertices[:,:,2],.3,atol=1e-6)
        result['cuda_triangles']=len(vertices)
        from cuda_registration import register
        from offline_pipeline import Settings
        ok,pose,fitness,rmse=register(depth+.001,depth,[120,120,16,16],[120,120,16,16],np.eye(4),Settings())
        assert ok and abs(pose[2,3]+.001)<1e-5
        result['cuda_registration']={'fitness':fitness,'rmse':rmse,'translation':pose[:3,3].tolist()}
        output=DepthDenoiser().denoise(depth,rgb);assert np.isfinite(output).all()
        matcher=NeuralMatcher();result['loftr_device']=matcher.device
        import tkinter as tk,threading
        from offline_pipeline import OfflineStudio
        root=tk.Tk();root.withdraw()
        try:
            app=OfflineStudio.__new__(OfflineStudio);app.root=root;app.ip=tk.StringVar()
            app.status=tk.StringVar();app.quality=tk.DoubleVar(value=.5);app.params={};app.cancel=threading.Event()
            app.build_ui();root.update_idletasks()
            assert app.model_choice.get();assert app.reconstruction_mode.get()=='RTX sola'
            result['ui']=True
        finally:root.destroy()
        result.update(ok=True,torch_path=torch.__file__,frozen=bool(getattr(sys,'frozen',False)))
    except Exception:
        result.update(ok=False,error=traceback.format_exc())
    Path(destination).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return 0 if result.get('ok') else 1
