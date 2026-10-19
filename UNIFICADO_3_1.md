# Bichito 3D 3.1

Windows se distribuye como `Bichito3D-Unificado-v3.1.exe`, empaquetado onefile con Python, CUDA runtime/NVRTC, CuPy, PyTorch, LoFTR y el denoiser entrenado. No necesita un Python externo ni un BAT separado. Al abrir, el empaquetador extrae temporalmente sus dependencias; eso requiere espacio y añade tiempo de arranque. El controlador NVIDIA sigue siendo un requisito del sistema.

Elegir RTX sola, RTX + IA o Comparar ambas antes de reconstruir. La comparación ejecuta ambos recorridos sobre la misma sesión y conserva mallas separadas. El selector de resultados determina qué malla se ve y exporta como STL/PLY. RTX sola no ejecuta las redes neuronales; RTX + IA aplica el denoiser seleccionado y, si se activa, LoFTR. Los originales se conservan.

Avanzado incluye un selector de modelo, Añadir modelo y Establecer modelo predeterminado. Los modelos añadidos deben ser checkpoints `residual-unet-v1` con un JSON acompañante que indique arquitectura y SHA256; se valida su state_dict antes de incorporarlos. Modelos de otras arquitecturas necesitan su adaptador: cambiar un nombre no convierte una red incompatible. La selección predeterminada persiste en LOCALAPPDATA/Bichito3D/Models y no depende del directorio temporal del EXE. El modelo incluido sigue siendo experimental: NYU/Kinect con ruido simulado, no validación física TrueDepth.

iOS 3.1 build 5: se eliminó la parada por cinco frames acumulados omitidos, una condición que podía cancelar el arranque durante compresión PNG. Los samples sin calibración se omiten sin convertirlos en errores fatales. Se corrigió el orden de notificación al detener por un error de almacenamiento. Se mantienen la protección térmica crítica, el límite de caché y la preservación de bloques pendientes.

La verificación con iPhone físico sigue pendiente; la compilación y los tests no sustituyen esa medición. El EXE incluye `--diagnostic archivo.json` para verificar CUDA/TSDF/Marching Cubes, carga de ambas redes y construcción de UI sin ocupar puertos de captura.
