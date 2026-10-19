# Limpieza neuronal de profundidad

## Búsqueda de modelos utilizables

Se consultaron las páginas de los autores y repositorios originales:

* [Shabanov et al., 3DV 2020](https://ahanio.github.io/lq2hq/) y [artículo completo](https://arxiv.org/html/2009.04776): sí estudian TrueDepth de iPhone X frente a Kinect v2. La página no ofrece pesos, dataset ni código descargables. El artículo advierte que su red entrenada con personas solo es fiable en regiones de cuerpos humanos. No se reutilizó ni se atribuyó una validación que no tenemos.
* [DeepDepthDenoising, ICCV 2019](https://github.com/VCL3D/DeepDepthDenoising): código MIT y enlace a pesos, pero entrenados para Intel RealSense D415. No constituyen pesos validados para TrueDepth.
* [NYU Depth V2, fuente original](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html): RGB-D público de Kinect, 1.449 pares etiquetados y escenas identificadas. Se descargó directamente el archivo original de 2.972.037.809 bytes, evitando conversiones de profundidad de terceros. Cita: Silberman, Kohli, Hoiem y Fergus, *Indoor Segmentation and Support Inference from RGBD Images*, ECCV 2012.

La búsqueda no demuestra la inexistencia universal de pesos TrueDepth; no se localizaron unos públicos, descargables y verificables adecuados para esta aplicación.

## Implementación entregada

`neural_depth.py`: U-Net residual propia, entrenada desde cero con RGB, profundidad métrica, máscara de validez y detalle local de profundidad. La corrección está limitada a 10 mm, conserva escala métrica y no inventa valores donde falta medición. Solo opera sobre una copia durante la reconstrucción posterior en Windows; no cambia la captura, compresión ni transferencia.

`download_nyu.py` descarga el dataset público original, reanudando descargas incompletas. `train_depth.py` separa escenas completas en entrenamiento, validación y prueba (70/15/15; semilla 2718); el ruido de prueba usa una semilla distinta. Las escenas se escalan al rango cercano. Se añaden ruido gaussiano, ruido correlacionado y cuantización aleatorios. **Es un modelo de ruido sintético, no una calibración medida de TrueDepth.** La supervisión excluye los píxeles donde el Kinect no midió profundidad. La pérdida combina error de profundidad y gradientes para penalizar el borrado de bordes.

Los pesos de mejor validación se seleccionan sin utilizar el conjunto de prueba. `models/depth_denoiser_v1.json` conserva SHA-256 del dataset y pesos, particiones por escena, arquitectura, semillas, pasos, dispositivo, historial y métricas de prueba. `models/depth_denoiser_v1.pt` contiene solo el state_dict; su carga utiliza `weights_only=True` y comprueba integridad.

## Uso

Los pesos entrenados se entregan preparados. Abrir `Iniciar_Bichito3D_GPU.bat`, pestaña Avanzado, activar **IA profundidad experimental**. Funciona también en CPU con PyTorch instalado; el EXE portable sin PyTorch indica abrir el runtime RTX / IA. No se activa silenciosamente un modelo sin validación real del sensor.

Para reproducir: `Entrenar_IA_Profundidad.bat` (descarga de unos 3 GB y entrenamiento local). El código, pesos e informe no necesitan cuentas, datos aportados manualmente ni servicios de inferencia externos. Los datos originales descargados permanecen solo en `.training-data`, fuera del repositorio.

## Alcance de la evaluación

La mejora de error reportada corresponde a escenas NYU apartadas con ruido sintético añadido y reescalado al rango cercano. No mide la exactitud real de TrueDepth ni prueba conservación de detalles de 0.05 mm. Las referencias Kinect tampoco son patrones de metrología. La red queda implementada y entrenada; sigue siendo experimental en TrueDepth hasta disponer de una evaluación con referencias reales del sensor. Las pruebas funcionales comprueban inferencia real, agujeros, integridad, límites de corrección y conservación de originales.
