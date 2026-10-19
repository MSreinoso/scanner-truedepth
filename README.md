# Bichito 3D 3.0 — entrega y límites

Modo Simple predeterminado: Estándar (0.5 mm), Sketch (2 mm), Alta (0.2 mm), Máxima (0.05 mm). El panel Avanzado conserva resolución exacta, ICP, color, muestreo, memoria y backend. 1:1 no significa precisión física de 0.05 mm. El visor no usa movimiento automático ni orientación IMU para girar.

## Captura y transferencia

RGB: PNG sin pérdida para los píxeles BGRA opacos de la cámara; profundidad: Float32 original, byte shuffle reversible y Zstandard 1.5.7 nivel 1. No se convierte a Float16. Los NaN, ceros con signo y bits de profundidad se prueban con round trip Swift/Python. Referencias exactas entre frames iguales reducen redundancia conservando timestamp, IMU y calibración de cada frame. Los frames parecidos, pero diferentes, no se borran como si fueran idénticos.

WebSocket binario en TCP 50508. Cada mensaje contiene secuencia, SHA-256 y un frame comprimido. Windows valida y escribe/fsync antes de ACK. El iPhone borra solo ese bloque después de comprobar el ACK; la confirmación final valida todos los frames y permite liberar el manifiesto de la sesión. Los reintentos son idempotentes y automáticos (1–15 s). Se conserva compatibilidad de lectura con v2.

La caché de bloques v3 se limita a 180 MB: si la red no drena la cola, la captura se detiene sin sobrescribir datos pendientes. Esto no garantiza que una sesión completa pese menos de 200 MB ni que la compresión alcance siempre 8:1. Durante la captura la PC muestra solo el preview ligero y guarda bloques comprimidos; no reconstruye. Las sesiones v2 antiguas sin confirmar se conservan; esta actualización no las elimina silenciosamente.

## RTX y reconstrucción

Runtime CUDA aislado en `.gpu-runtime`, preparado en este equipo para RTX 5060 Laptop, 8 GB. Abrir `Iniciar_Bichito3D_GPU.bat`, o el botón Abrir modo RTX/IA desde el EXE. El EXE portable incluye CPU; el modo RTX usa el runtime local ya instalado, no un CUDA inexistente en el wheel Open3D de Windows. `Instalar_GPU.bat` permite recrearlo con Python 3.11 y requiere varios GB de descarga/disco en la PC.

Alineación global ICP multiescala en CPU; volumen global grueso para localizar superficie. Refinamiento: TSDF y Marching Cubes con kernels CUDA reales por mosaicos, liberando VRAM entre mosaicos y guardando malla por partes en disco. No se aumenta silenciosamente el voxel fino. El visor usa una muestra limitada, las exportaciones STL/PLY recorren las partes completas. No existe un plazo garantizado para escenas arbitrarias: la cantidad de superficie y los frames siguen afectando tiempo y tamaño del resultado. La lectura de frames aceptados y la alineación global también requieren recursos de CPU/disco.

La RTX 5060 ejecutó una reconstrucción sintética de 3 frames a 0.05 mm: 1.521 mosaicos, 3.800.498 triángulos, aproximadamente 9 segundos en esa escena pequeña. Es una prueba funcional, no una medida de precisión física ni un benchmark extrapolable a cualquier escaneo. El test de una superficie conocida verificó su coordenada Z y la exportación STL en mm.

## RGB e IA

Limpieza conjunta guiada por textura/bordes RGB, opcional y aplicada solo al reconstruir. No se modifica el archivo de mediciones. No se denomina photometric stereo: no hay un sistema de iluminación controlada que justifique ese modelo físico.

LoFTR preentrenado outdoor de Kornia: alineación de características opcional, inferencia local con PyTorch 2.8 CUDA 12.8, semillas de pose rígida robusta y validación final ICP. Los pesos públicos se descargaron por HTTPS, se fijó SHA-256 y se comprobó inferencia en la RTX. Las imágenes no se envían a servicios externos. Pesos de UNet/MobileStereoNet entrenados para el ruido específico de TrueDepth no fueron proporcionados: no se ha inventado ni activado un modelo de limpieza que afirme precisión profesional sin validación.

## Validación

26 pruebas locales aprobadas, incluidas CUDA real, inicialización idempotente del runtime, protocolo, datos Float32/PNG bit a bit, reintentos, persistencia, presets y geometría. Pruebas adicionales: WebSocket con desconexión y reconexión real, reconstrucción fina en la RTX, carga e inferencia LoFTR. CI compila iOS/Windows y prueba el codec Swift contra Python; sus runners sin GPU omiten las pruebas CUDA, que se ejecutaron localmente.

Pendiente: medir calidad, tasa de compresión, temperatura, pérdidas y recuperación con el iPhone real y capturas representativas. Sin esa evidencia no se garantiza que se hayan eliminado todos los cierres ni un acabado profesional.

Referencias: [Zstandard oficial](https://github.com/facebook/zstd), [OpenEXR Float/Half](https://openexr.com/en/latest/TechnicalIntroduction.html), [LoFTR de Kornia](https://kornia.readthedocs.io/en/latest/models/loftr.html), [instalación CuPy](https://docs.cupy.dev/en/stable/install.html).
