Versión 3.2 — revisión de captura, RTX e interfaz

Se admite BGRA de cámara con alfa no opaco mediante Zstd exacto, sin descartar el primer fotograma. El error original ya no se sustituye por el mensaje genérico de sesión vacía. Inicio/finalización remotos son órdenes separadas e idempotentes.

La reconstrucción RTX usa ICP proyectivo coarse-to-fine, localización de superficie, TSDF y Marching Cubes en CUDA. La decodificación, rectificación RGB-D y E/S siguen en CPU. La caché de imágenes CUDA está limitada por VRAM disponible y presupuesto. El informe model.json identifica GPU, método y tareas CPU.

Interfaz por pasos, controles reordenados al reducir ventana, selectores de ancho flexible, paneles desplazables y estado del iPhone visible. 0,05 mm predeterminado para objetos generales; no hay segmentación facial. Ese valor es muestreo de reconstrucción, no precisión física garantizada de TrueDepth.

Validación: 36 pruebas locales; ICP recupera desplazamiento conocido; fusión CUDA de plano; interfaz 680x520, 800x600, 1024x768 y 1380x900. Reconstrucción sintética a 0,05 mm: 3/3 frames aceptados, 3.804.362 triángulos, RTX 5060 Laptop. No equivale a un escaneo físico validado.

Pendiente de prueba física: reproducir Iniciar desde ambas interfaces en el iPhone del usuario. El fallo de alfa es una causa posible identificada y cubierta por prueba Swift/Python; no se afirma que sea la causa confirmada en su dispositivo. La nueva versión conservará y mostrará el error concreto si existe otro fallo.
