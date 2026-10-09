#!/usr/bin/env python3
"""
Práctica 1 (RIA) - Seguimiento de un vehículo guía en RoboboSim con RL.

Un solo archivo con varios modos (se elige uno por ejecución):

    python3 p1_entrenar.py --calibrar [--modo acercar|alejar|continuo] [--tilt 90]
    python3 p1_entrenar.py --comprobar
    python3 p1_entrenar.py --aleatorio --episodios 10
    python3 p1_entrenar.py --entrenar --pasos 20000 --semilla 0 --nombre ref

Opciones comunes:
    --config config_p1.json   archivo con los valores que dependen de la calibración
    --sin-simulador           robot real: sin módulo sim, el reinicio lo hace una persona

Decisiones de diseño (a justificar en la memoria):
  * Observación: SÓLO sensores del seguidor (blob verde + última acción).
    La posición del guía/seguidor y la separación real vienen del módulo sim y
    se usan únicamente para la recompensa, la terminación y el registro.
  * Acción: continua, 2 dimensiones (velocidad lineal y angular) por cmd_vel.
  * Actuación por cmd_vel (m/s, rad/s). Si se prefiere move_wheels, basta con
    cambiar _publicar_velocidad(); no hace falta tocar nada más.
  * Algoritmo: SAC (fuera de política, aprovecha mejor cada muestra; cada paso
    cuesta tiempo real).

Todos los valores que dependen de la calibración están en el archivo de
configuración (CONFIG_DEFECTO muestra las claves), no en el código.
"""

import argparse
import csv
import json
import math
import os
import statistics
import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist

# ---------------------------------------------------------------------
# Interfaces de ROS 2 del puente robobo_ros2
# ---------------------------------------------------------------------
# Comprobar con `ros2 interface list | grep -i robobo` y corregir aquí si
# alguno de los nombres no coincide con tu versión del puente.
try:
    from robobo_ros2_interfaces.msg import BlobArray
    from robobo_ros2_interfaces.srv import SetActiveColorBlobs, MoveTilt
    HAY_INTERFACES_BLOB = True
except ImportError:
    BlobArray = SetActiveColorBlobs = MoveTilt = None
    HAY_INTERFACES_BLOB = False

# Interfaces del módulo sim (sólo existen en imágenes recientes del
# contenedor). Sin ellas el entorno funciona igual, pero no puede reiniciar
# la escena ni conocer la verdad del terreno.
try:
    from robobo_ros2_interfaces.msg import RobotLocation
    from robobo_ros2_interfaces.srv import ResetSimulation
    HAY_INTERFACES_SIM = True
except ImportError:
    RobotLocation = ResetSimulation = None
    HAY_INTERFACES_SIM = False

AVISO_SIN_SIM = (
    'No se encuentran las interfaces del módulo sim '
    '(robobo_ros2_interfaces.msg.RobotLocation).\n'
    'Suele ser una de estas dos cosas:\n'
    '  1. La terminal no tiene cargado el espacio de trabajo del puente:\n'
    '         source /opt/ros/robobo/setup.bash\n'
    '  2. La imagen del contenedor es anterior al módulo sim y hay que\n'
    '     reconstruirla.\n'
    'Mientras tanto, usa --sin-simulador (robot repuesto a mano entre episodios).')

# =====================================================================
# Constantes del problema
# =====================================================================

# Espacios de nombres del nodo puente (valores por defecto de robobo_container).
NS_BASE = '/robobo/robot_0/base'
NS_SMARTPHONE = '/robobo/robot_0/smartphone'
NS_SIM = '/robobo/robot_0/sim'
# El guía se conoce lanzando un SEGUNDO contenedor con sólo el módulo sim.
# ASUMIDO: robot_1 es el guía. Si no hay segundo contenedor, el entorno usa
# un sustituto de la recompensa basado en el tamaño del blob (ver _recompensa).
NS_SIM_GUIA = '/robobo/robot_1/sim'

# --- Parámetros del anexo B (geometría y criterios de fallo) ---
SEP_COLISION_M = 0.20          # colisión: distancia entre centros < 0,20 m
BANDA_M = (0.25, 0.50)         # banda objetivo de separación
SEP_PERDIDO_M = 1.00           # guía perdido: separación > 1,0 m
SEP_CENTRO_BANDA_M = 0.375     # centro de la banda

# --- Coeficientes de la recompensa (cada uno se declara en la memoria) ---
R_BANDA = 1.0          # recompensa por paso dentro de la banda objetivo
K_FUERA = 4.0          # penalización por metro (o unidad de tamaño) fuera de la banda
TOPE_FUERA = 2.0       # tope de esa penalización por paso
C_NO_VISTO = 0.5       # penalización por paso sin ver la pelota
C_GIRO = 0.05          # penalización por |w| normalizado (evita zigzag)
C_CAMBIO = 0.05        # penalización por cambio brusco de acción (suavidad)
P_COLISION = 10.0      # penalización terminal por colisión
P_PERDIDO = 10.0       # penalización terminal por perder al guía

# --- Parámetros de la calibración del sensor ---
V_PULSO, PASO_CAL_S = 0.04, 0.2   # m/s, s  -> ~8 mm por pulso de avance
W_GIRO, GIRO_S = 0.3, 0.3         # rad/s, s -> ~5° por pulso de giro
N_GIRO = 4                        # pulsos de giro hacia cada lado
N_RUIDO = 20                      # lecturas en reposo para medir el ruido
SEP_PARADA_M = 0.18               # al acercarse, parar antes de embestir
SEP_MAX_M = 1.5                   # al alejarse, tope de seguridad
SEP_REFERENCIA = (0.20, 0.25, 0.375, 0.50, 1.00)

# Espera máxima por una lectura nueva antes de dar el episodio por perdido.
# Los blobs se publican a 5 Hz (cada 0,2 s), así que 3 s es muy holgado.
ESPERA_MAX_S = 3.0

# Valores por defecto de la configuración. SOBRESCRIBIR en config_p1.json con
# lo medido en la calibración (simulador primero, robot real después).
CONFIG_DEFECTO = {
    'color_blob': 'green',
    'tilt': 90.0,            # inclinación de la cámara (grados)
    'size_lejos': 100.0,     # blob.size con el guía muy lejos (límite de visión)
    'size_cerca': 1000.0,    # blob.size con el guía muy cerca (máximo)
    'size_obj': 400.0,       # blob.size a la separación objetivo (0,375 m) -> medir
    'size_tol': 0.10,        # semianchura de la banda en tamaño normalizado (sustituto)
    'size_choque': 0.97,     # tamaño normalizado que se considera "chocando" (sustituto)
    'x_min': 0.0,            # rango de blob.x en la imagen
    'x_max': 100.0,
    'y_min': 0.0,            # rango de blob.y en la imagen
    'y_max': 100.0,
    'v_max': 0.15,           # m/s hacia delante
    'v_min': -0.05,          # m/s hacia atrás
    'w_max': 1.0,            # rad/s
    'paso_s': 0.2,           # espera tras aplicar la acción (< cmd_vel timeout 0,5 s)
    'pasos_max': 150,        # truncamiento (pasos por episodio)
    'n_perdido': 10,         # pasos seguidos sin ver la pelota => guía perdido
}


def cargar_config(ruta):
    """Lee el archivo de configuración y lo mezcla con los valores por defecto."""
    cfg = dict(CONFIG_DEFECTO)
    if ruta and os.path.exists(ruta):
        with open(ruta) as fh:
            cfg.update(json.load(fh))
    elif ruta:
        print('Aviso: no existe {}; se usan los valores por defecto.'.format(ruta))
    return cfg


def llamar_servicio(nodo, tipo, nombre, **campos):
    """Llama a un servicio de ROS 2 y espera la respuesta (haciendo spin)."""
    cli = nodo.create_client(tipo, nombre)
    if not cli.wait_for_service(timeout_sec=2.0):
        raise RuntimeError('Servicio {} no disponible'.format(nombre))
    req = tipo.Request()
    for k, v in campos.items():
        setattr(req, k, v)
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(nodo, fut, timeout_sec=5.0)
    return fut.result()


# =====================================================================
# El entorno
# =====================================================================

class RoboboSeguimientoEnv(gym.Env):
    """El seguidor sigue a un Robobo guía con una pelota verde en RoboboSim.

    Observación (Box, float32, dimensión 7) - SÓLO sensores del seguidor
        [0] visible   1 si el detector devuelve el blob verde, 0 si no
        [1] x         posición horizontal del blob, en [-1, 1] (-1 = izquierda)
        [2] y         posición vertical del blob, en [-1, 1]
        [3] size      tamaño del blob normalizado a [0, 1] (0 lejos, 1 muy cerca)
        [4] x_ult     última x vista (indica por qué lado se perdió la pelota)
        [5] v_prev    velocidad lineal ordenada en el paso anterior, en [-1, 1]
        [6] w_prev    velocidad angular ordenada en el paso anterior, en [-1, 1]

        Si la pelota no se ve: visible=0 y x=y=size=0 (la bandera `visible`
        distingue "no se ve" de "se ve en el centro"). x_ult conserva la última
        x vista. Esto es información de sensor, así que existe en el robot real.

    Acción (Box, float32, dimensión 2), continua
        [0] velocidad lineal en [-1, 1]  (0 = parado; >0 -> [0, v_max]; <0 -> [v_min, 0])
        [1] velocidad angular en [-1, 1] (-> [-w_max, w_max])

    Terminación (terminated)
        colisión | guía perdido (separación > 1 m, o N pasos sin ver la pelota)
    Truncamiento (truncated)
        se alcanzan pasos_max pasos.
    """

    metadata = {'render_modes': []}

    def __init__(self, cfg=None, usar_simulador=True, verbose=False):
        super().__init__()
        self.cfg = cfg if cfg is not None else dict(CONFIG_DEFECTO)
        self.verbose = verbose
        self.usar_simulador = usar_simulador

        if not HAY_INTERFACES_BLOB:
            raise RuntimeError('No se encuentran las interfaces de blobs '
                               '(robobo_ros2_interfaces). Revisa los imports.')
        if usar_simulador and not HAY_INTERFACES_SIM:
            raise RuntimeError(AVISO_SIN_SIM)

        # rclpy.init() sólo puede llamarse una vez por proceso.
        if not rclpy.ok():
            rclpy.init()
        self.nodo = Node('p1_entorno_rl')

        # Publicador y suscripciones se crean UNA vez, aquí (crearlos dentro
        # de step() haría perder las primeras órdenes por el descubrimiento).
        self.pub_vel = self.nodo.create_publisher(Twist, NS_BASE + '/cmd_vel', 10)

        self._blob = None        # mejor blob verde de la última lectura (o None)
        self._n_blob = 0         # nº de mensajes recibidos (incluye los vacíos)
        self._t_blob = 0.0       # instante de llegada del último mensaje
        self.nodo.create_subscription(
            BlobArray, NS_SMARTPHONE + '/color_blobs', self._cb_blobs, 1)

        # Sólo el color verde, y cámara a la inclinación calibrada.
        llamar_servicio(self.nodo, SetActiveColorBlobs,
                        NS_SMARTPHONE + '/set_active_color_blobs',
                        red=False, green=True, blue=False, custom=False)
        llamar_servicio(self.nodo, MoveTilt, NS_BASE + '/move_tilt',
                        angle=float(self.cfg['tilt']), speed=30.0)
        time.sleep(1.0)

        # Canal privilegiado del simulador (no existe en el robot real).
        self.cli_reset = None
        self._pose_yo = None      # (x_mm, z_mm, yaw_deg) del seguidor
        self._pose_guia = None    # idem del guía (requiere 2º contenedor)
        if usar_simulador:
            self.cli_reset = self.nodo.create_client(
                ResetSimulation, NS_SIM + '/reset_simulation')
            if not self.cli_reset.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(
                    'No responde {}/reset_simulation. ¿robobo_container lanzado '
                    'con el módulo sim?'.format(NS_SIM))
            self.nodo.create_subscription(
                RobotLocation, NS_SIM + '/robot_location',
                lambda m: setattr(self, '_pose_yo', self._pose(m)), 1)
            self.nodo.create_subscription(
                RobotLocation, NS_SIM_GUIA + '/robot_location',
                lambda m: setattr(self, '_pose_guia', self._pose(m)), 1)

        # Espacios de observación y acción.
        self.observation_space = spaces.Box(
            low=np.array([0, -1, -1, 0, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Estado interno del episodio.
        self.pasos = 0
        self.sin_ver = 0
        self.x_ult = 0.0
        self.ultima_accion = np.zeros(2, dtype=np.float32)

        # Confirma que el puente está en marcha antes de entrenar.
        self._esperar_lectura_nueva(inicial=True)

    # -----------------------------------------------------------------
    # Comunicación con ROS 2
    # -----------------------------------------------------------------

    @staticmethod
    def _pose(msg):
        """(x, z, yaw) del robot en milímetros y grados (verdad del terreno)."""
        return (msg.position.x, msg.position.z, msg.rotation.y)

    def _cb_blobs(self, msg):
        """Callback de color_blobs; sólo se ejecuta dentro de un spin.

        El detector entrega a lo sumo un blob por imagen, pero se filtra por
        color y se queda con el mayor por si acaso. La lista vacía también
        cuenta como lectura ("pelota no vista").
        """
        verdes = [b for b in msg.blobs if b.color == self.cfg['color_blob']]
        self._blob = max(verdes, key=lambda b: b.size) if verdes else None
        self._n_blob += 1
        self._t_blob = time.time()

    def _vaciar_cola(self):
        """Procesa y descarta los mensajes pendientes.

        Todo lo que haya en cola en este momento se publicó antes o durante la
        aplicación de la acción, así que no se devuelve como observación.
        """
        for _ in range(100):
            antes = self._n_blob
            rclpy.spin_once(self.nodo, timeout_sec=0.0)
            if self._n_blob == antes:
                return

    def _esperar_lectura_nueva(self, inicial=False, republicar=None):
        """Bloquea hasta recibir una lectura POSTERIOR a este instante.

        Es el punto delicado de acoplar Gymnasium con ROS 2: step() debe
        devolver la observación resultante de la acción, pero ROS 2 entrega
        los mensajes cuando el ejecutor los procesa. Devolver la última lectura
        guardada daría una observación anterior a la acción: no falla ningún
        mensaje, el entrenamiento corre y simplemente no aprende.

        `republicar=(v, w)` reenvía la orden mientras se espera, para que el
        temporizador de cmd_vel (0,5 s, tiempo real) no detenga al robot.
        """
        objetivo = self._n_blob + 1
        t0 = time.time()
        while self._n_blob < objetivo:
            if republicar is not None:
                self._publicar_velocidad(*republicar)
            rclpy.spin_once(self.nodo, timeout_sec=0.05)
            if time.time() - t0 > ESPERA_MAX_S:
                raise RuntimeError(
                    'No llegan mensajes de {}/color_blobs. ¿Está en marcha '
                    'robobo_container con el módulo blob?'.format(NS_SMARTPHONE)
                    if inicial else
                    'Se ha interrumpido el flujo de blobs durante el episodio.')

    def _publicar_velocidad(self, v, w):
        """Envía la orden de movimiento (v en m/s, w en rad/s)."""
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_vel.publish(msg)

    def _detener(self):
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    def _reiniciar_escena(self):
        """Reinicia la escena del simulador (llamada a servicio con spin)."""
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)
        if futuro.result() is None or not futuro.result().success:
            raise RuntimeError('El reinicio de la escena ha fallado.')

    # -----------------------------------------------------------------
    # Observación y verdad del terreno
    # -----------------------------------------------------------------

    def _tamano_norm(self):
        """size normalizado a [0, 1] (0 = lejos, 1 = muy cerca); 0 si no se ve."""
        b = self._blob
        if b is None:
            return 0.0
        lejos, cerca = self.cfg['size_lejos'], self.cfg['size_cerca']
        return float(np.clip((b.size - lejos) / (cerca - lejos), 0.0, 1.0))

    def _observacion(self):
        b = self._blob
        if b is None:
            visible, x, y, s = 0.0, 0.0, 0.0, 0.0
        else:
            c = self.cfg
            visible = 1.0
            x = 2.0 * (b.x - c['x_min']) / (c['x_max'] - c['x_min']) - 1.0
            y = 2.0 * (b.y - c['y_min']) / (c['y_max'] - c['y_min']) - 1.0
            x, y = float(np.clip(x, -1, 1)), float(np.clip(y, -1, 1))
            s = self._tamano_norm()
            self.x_ult = x
        return np.array([visible, x, y, s, self.x_ult, *self.ultima_accion],
                        dtype=np.float32)

    def _separacion(self):
        """Separación REAL en metros (distancia entre centros), o None.

        Verdad del terreno: sólo para recompensa, terminación y registro, NUNCA
        para la observación. Aviso: es la distancia euclídea. El anexo B define
        la separación como distancia entre las proyecciones sobre la línea
        central del carril, que requiere la geometría del anillo (la debe
        implementar el evaluador). En rectas coinciden; en curvas difieren.
        """
        if self._pose_yo is None or self._pose_guia is None:
            return None
        return math.hypot(self._pose_yo[0] - self._pose_guia[0],
                          self._pose_yo[1] - self._pose_guia[1]) / 1000.0

    # -----------------------------------------------------------------
    # Recompensa
    # -----------------------------------------------------------------

    def _recompensa(self, sep, visible, accion, anterior, choque, perdido):
        """Recompensa del paso, con cada término por separado.

        1. Banda de separación: +R_BANDA dentro de la banda 0,25-0,50 m; fuera,
           -K_FUERA por metro de distancia a la banda (tope TOPE_FUERA).
           Sin segundo contenedor (sep=None) se usa un sustituto en tamaño del
           blob: banda centrada en size_obj con semianchura size_tol.
        2. Pelota no vista: -C_NO_VISTO por paso.
        3. Suavidad: -C_GIRO*|w| y -C_CAMBIO*|a - a_anterior|.
        4. Terminales: -P_COLISION por colisión, -P_PERDIDO por guía perdido.
        """
        if sep is not None:
            fuera = max(0.0, abs(sep - SEP_CENTRO_BANDA_M)
                        - (BANDA_M[1] - BANDA_M[0]) / 2.0)
        else:
            c = self.cfg
            obj = (c['size_obj'] - c['size_lejos']) / (c['size_cerca'] - c['size_lejos'])
            fuera = max(0.0, abs(self._tamano_norm() - obj) - c['size_tol'])
        r = R_BANDA if fuera == 0.0 else -min(TOPE_FUERA, K_FUERA * fuera)

        if not visible:
            r -= C_NO_VISTO
        r -= C_GIRO * abs(float(accion[1]))
        r -= C_CAMBIO * float(np.abs(accion - anterior).sum())

        if choque:
            r -= P_COLISION
        if perdido:
            r -= P_PERDIDO
        return float(r)

    # -----------------------------------------------------------------
    # Interfaz de Gymnasium
    # -----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._detener()

        if self.cli_reset is not None:
            # Simulador: reinicio de la escena. El seguidor parte parado y el
            # guía sigue su ronda. Todos los episodios empiezan en la misma
            # pose; aleatorizarla con sim/set_robot_location es una variante
            # candidata para la tabla de experimentos.
            self._reiniciar_escena()
            time.sleep(0.5)
        else:
            # Robot real: la persona repone al seguidor tras el episodio.
            input('Coloca el robot detrás del guía y pulsa Intro... ')

        self.pasos = 0
        self.sin_ver = 0
        self.x_ult = 0.0
        self.ultima_accion = np.zeros(2, dtype=np.float32)

        # Descarta lo acumulado durante el reinicio y espera una lectura nueva.
        self._vaciar_cola()
        self._esperar_lectura_nueva()
        return self._observacion(), {}

    def step(self, accion):
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)
        anterior = self.ultima_accion
        self.ultima_accion = accion

        # 1. Acción normalizada -> unidades físicas (0 equivale a parado).
        c = self.cfg
        v = accion[0] * c['v_max'] if accion[0] >= 0 else -accion[0] * c['v_min']
        w = float(accion[1]) * c['w_max']
        t_accion = time.time()
        self._publicar_velocidad(v, w)

        # 2. Dejar que la acción surta efecto (sin spin: los mensajes se
        #    acumulan y se descartan a continuación).
        time.sleep(c['paso_s'])

        # 3. Vaciar la cola y esperar una lectura posterior a la acción,
        #    reenviando la orden para que cmd_vel no caduque.
        self._vaciar_cola()
        self._esperar_lectura_nueva(republicar=(v, w))

        # Comprobación explícita del requisito técnico del enunciado: la
        # lectura devuelta llegó DESPUÉS de aplicar la acción.
        latencia = self._t_blob - t_accion
        if latencia <= 0.0:
            raise RuntimeError('La observación es anterior a la acción (latencia {:.3f} s).'
                               .format(latencia))

        # 4. Evaluar el resultado.
        self.pasos += 1
        visible = self._blob is not None
        self.sin_ver = 0 if visible else self.sin_ver + 1
        sep = self._separacion()

        if sep is not None:
            choque = sep < SEP_COLISION_M
            perdido = sep > SEP_PERDIDO_M or self.sin_ver >= c['n_perdido']
        else:   # sin verdad del terreno: sustituto basado en el blob
            choque = self._tamano_norm() >= c['size_choque']
            perdido = self.sin_ver >= c['n_perdido']

        recompensa = self._recompensa(sep, visible, accion, anterior, choque, perdido)
        terminated = bool(choque or perdido)
        truncated = bool(self.pasos >= c['pasos_max'])
        if terminated or truncated:
            self._detener()

        causa = ('colision' if choque else 'perdido' if perdido
                 else 'tiempo' if truncated else 'ninguna')
        info = {'causa': causa, 'separacion': sep, 'visible': visible,
                'latencia_obs': latencia, 'pose_yo': self._pose_yo,
                'pose_guia': self._pose_guia}

        if self.verbose:
            print('paso {:3d} vis={} sep={} v={:+.3f} w={:+.3f} r={:+.3f} {}'.format(
                self.pasos, int(visible),
                '--' if sep is None else '{:.3f}'.format(sep), v, w, recompensa,
                '' if causa == 'ninguna' else causa.upper()))

        return self._observacion(), recompensa, terminated, truncated, info

    def close(self):
        try:
            self._detener()
            self.nodo.destroy_node()
        except Exception:
            pass
        # No se llama a rclpy.shutdown(): otros entornos del proceso lo necesitan.


# =====================================================================
# Modo --calibrar: caracterización del sensor (apartado 5.1)
# =====================================================================

def dif_ang(a, b):
    """Diferencia angular a - b normalizada a [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def calibrar(cfg, modo='acercar', barrido=True, ruido=True, tilt=None,
             dist_inicial_m=None, csv_path='calibracion_blob.csv',
             json_path='calibracion.json'):
    """Mide qué ve la cámara: rango, tamaño frente a distancia, ruido, barrido.

    modo='acercar'  mide ruido, barrido lateral y aproximación al guía
                    (parado). Da size/x/y frente a la separación.
    modo='alejar'   retrocede hasta perder la pelota: da el alcance máximo.
    modo='continuo' no mueve el robot: imprime y registra lecturas (robot real,
                    o con otro programa conduciendo al seguidor por rectas y
                    curvas).

    tilt: inclinación de la cámara en grados (repetir con varios valores para
          ver su efecto). Por defecto, el de la configuración.
    dist_inicial_m: separación inicial medida a mano. Sólo se usa sin módulo
          sim (robot real): la separación se estima como dist_inicial_m menos el
          avance ordenado (lazo abierto, orientativo).

    Salida: tabla por pantalla, CSV con todas las lecturas y JSON con las
    constantes para pasar al archivo de configuración.
    """
    if not HAY_INTERFACES_BLOB:
        raise RuntimeError('No se encuentran las interfaces de blobs '
                           '(robobo_ros2_interfaces). Revisa los imports.')
    if tilt is None:
        tilt = cfg['tilt']
    color = cfg['color_blob']
    if not rclpy.ok():
        rclpy.init()
    nodo = Node('p1_calibracion')

    est = {'blob': None, 'n': 0, 'yo': None, 'guia': None}
    avance = [0.0]        # m ordenados (estimación en lazo abierto)
    giro = [0.0]          # grados ordenados (estimación en lazo abierto)
    filas, resumen = [], {'color': color, 'tilt': tilt}

    def cb_blobs(m):
        verdes = [b for b in m.blobs if b.color == color]
        est['blob'] = max(verdes, key=lambda b: b.size) if verdes else None
        est['n'] += 1     # cuenta también las listas vacías (pelota perdida)

    nodo.create_subscription(BlobArray, NS_SMARTPHONE + '/color_blobs', cb_blobs, 1)
    if HAY_INTERFACES_SIM:
        def pose(clave):
            return lambda m: est.__setitem__(
                clave, (m.position.x, m.position.z, m.rotation.y))
        nodo.create_subscription(RobotLocation, NS_SIM + '/robot_location', pose('yo'), 1)
        nodo.create_subscription(RobotLocation, NS_SIM_GUIA + '/robot_location',
                                 pose('guia'), 1)
    pub = nodo.create_publisher(Twist, NS_BASE + '/cmd_vel', 10)

    def lectura_nueva(n_msgs=2):
        """Espera n_msgs mensajes de blobs posteriores a la llamada. Con 2 se
        descarta el primero, que puede ser una imagen anterior al pulso."""
        objetivo = est['n'] + n_msgs
        t0 = time.time()
        while est['n'] < objetivo:
            rclpy.spin_once(nodo, timeout_sec=0.05)
            if time.time() - t0 > ESPERA_MAX_S:
                raise RuntimeError(
                    'No llegan blobs de {}/color_blobs. ¿Está en marcha '
                    'robobo_container con el módulo blob?'.format(NS_SMARTPHONE))

    def mover(v, w, segundos):
        """Publica la orden a 10 Hz durante `segundos` y luego para (cmd_vel
        caduca a los 0,5 s)."""
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        t0 = time.time()
        while time.time() - t0 < segundos:
            pub.publish(msg)
            time.sleep(0.1)
        msg.linear.x = msg.angular.z = 0.0
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.05)

    def girar(sg):    # +1 izquierda, -1 derecha (angular.z > 0 = antihorario)
        mover(0.0, sg * W_GIRO, GIRO_S)
        giro[0] += math.degrees(sg * W_GIRO * GIRO_S)

    def separacion():
        if est['yo'] and est['guia']:          # verdad del terreno, mm -> m
            return math.hypot(est['yo'][0] - est['guia'][0],
                              est['yo'][1] - est['guia'][1]) / 1000.0
        if dist_inicial_m is not None:
            return dist_inicial_m - avance[0]
        return None

    def medir(n_msgs=2):
        lectura_nueva(n_msgs)
        b, yo = est['blob'], est['yo']
        return {'size': b.size if b else 0, 'x': b.x if b else None,
                'y': b.y if b else None, 'sep': separacion(),
                'yaw': yo[2] if yo else giro[0]}

    def fmt(v, spec, ancho):
        return (format(v, spec) if v is not None else '--').rjust(ancho)

    def cabecera():
        print('{:<8s}{:>10s}{:>10s}{:>10s}{:>9s}{:>9s}'.format(
            'etapa', 'size', 'x', 'y', 'sep(m)', 'yaw(°)'))

    def registrar(etapa, d, imprimir=True):
        filas.append(dict(etapa=etapa, **d))
        if imprimir:
            print('{:<8s}{}{}{}{}{}'.format(
                str(etapa), fmt(d['size'], '.6g', 10), fmt(d['x'], '.6g', 10),
                fmt(d['y'], '.6g', 10), fmt(d['sep'], '.3f', 9),
                fmt(d['yaw'], '.1f', 9)))

    def medir_ruido():
        """Lecturas con el robot parado: media y desviación de size, x, y."""
        print('--- ruido en reposo ({} lecturas) ---'.format(N_RUIDO))
        m = []
        for _ in range(N_RUIDO):
            d = medir(1)
            registrar('ruido', d, imprimir=False)
            m.append(d)
        vis = [d for d in m if d['size'] > 0]
        r = {'lecturas': N_RUIDO, 'perdidas': N_RUIDO - len(vis)}
        if len(vis) >= 2:
            for k in ('size', 'x', 'y'):
                v = [d[k] for d in vis]
                r[k] = {'media': statistics.mean(v), 'std': statistics.pstdev(v)}
        resumen['ruido'] = r
        print(json.dumps(r, indent=2))

    def barrer():
        """Gira a ambos lados para ver cómo se mueve el blob en x (y dónde sale
        del campo de visión), y vuelve al centro comprobando la deriva."""
        print('--- barrido lateral: centro, derecha, centro, izquierda, centro ---')
        cabecera()
        c0 = medir()
        registrar('centro', c0)
        centros, lados = [c0], {}
        for nombre, sg in (('der', -1), ('izq', +1)):
            dados, ult = 0, None
            for i in range(1, N_GIRO + 1):
                girar(sg)
                dados += 1
                d = medir()
                registrar('{}{}'.format(nombre, i), d)
                if d['size'] == 0:             # pelota fuera de campo
                    break
                ult = d
            for _ in range(dados):             # volver al medio
                girar(-sg)
            c = medir()
            registrar('centro', c)
            centros.append(c)
            lados[nombre] = ult
        vis_c = [c for c in centros if c['size'] > 0]
        b = {'x_centro': statistics.mean(c['x'] for c in vis_c)}
        if centros[-1]['size'] > 0:
            b['deriva_x_al_volver'] = centros[-1]['x'] - c0['x']
        for nombre, ult in lados.items():
            dy = dif_ang(ult['yaw'], c0['yaw']) if ult else 0.0
            b['x_por_grado_' + nombre] = ((ult['x'] - c0['x']) / dy
                                          if abs(dy) > 1e-6 else None)
        resumen['barrido'] = b
        print(json.dumps(b, indent=2))

    try:
        llamar_servicio(nodo, SetActiveColorBlobs, NS_SMARTPHONE + '/set_active_color_blobs',
                        red=False, green=True, blue=False, custom=False)
        llamar_servicio(nodo, MoveTilt, NS_BASE + '/move_tilt',
                        angle=float(tilt), speed=30.0)
        time.sleep(1.0)

        if modo == 'continuo':
            cabecera()
            while rclpy.ok():                  # Ctrl+C para terminar
                registrar('c', medir(1))
            return

        d0 = medir()
        if d0['size'] == 0:
            print('No se ve la pelota {}. Coloca al seguidor mirándola.'.format(color))
            return
        if ruido:
            medir_ruido()
        if barrido:
            barrer()

        # Aproximación (o alejamiento) a pulsos de 8 mm.
        sentido = 1 if modo == 'acercar' else -1
        print('--- {} ---'.format('aproximación' if sentido > 0 else 'alejamiento'))
        cabecera()
        inicio = len(filas)
        maximo, estancado, perdidos, paso = d0['size'], 0, 0, 0
        while paso < 200:
            mover(sentido * V_PULSO, 0.0, PASO_CAL_S)
            avance[0] += sentido * V_PULSO * PASO_CAL_S
            paso += 1
            d = medir()
            registrar(paso, d)
            perdidos = perdidos + 1 if d['size'] == 0 else 0
            if d['size'] > maximo:
                maximo, estancado = d['size'], 0
            else:
                estancado += 1
            if perdidos >= 5:                  # pelota perdida: fin
                break
            if sentido > 0 and (estancado >= 15 or
                                (d['sep'] is not None and d['sep'] <= SEP_PARADA_M)):
                break                          # size saturado o demasiado cerca
            if sentido < 0 and d['sep'] is not None and d['sep'] >= SEP_MAX_M:
                break
        if sentido > 0:
            mover(-0.06, 0.0, 1.0)             # retirarse

        vis = [f for f in filas[inicio:] if f['size'] > 0]
        con_sep = [f for f in vis if f['sep'] is not None]
        ref = {}
        for r in SEP_REFERENCIA:               # size en las separaciones clave
            f = min(con_sep, key=lambda f: abs(f['sep'] - r)) if con_sep else None
            ref[str(r)] = f['size'] if f and abs(f['sep'] - r) <= 0.02 else None
        resumen['aproximacion'] = {
            'modo': modo,
            'size_inicial': d0['size'],
            'size_max': max(f['size'] for f in vis) if vis else 0,
            'x_rango': [min(f['x'] for f in vis), max(f['x'] for f in vis)] if vis else None,
            'sep_min_visto_m': min(f['sep'] for f in con_sep) if con_sep else None,
            'sep_max_visto_m': max(f['sep'] for f in con_sep) if con_sep else None,
            'fuente_sep': ('simulador' if est['guia'] else
                           'estimada' if dist_inicial_m is not None else None),
            'size_en_separacion_m': ref}
        print('\n' + json.dumps(resumen['aproximacion'], indent=2))
        print('\nCopia a config_p1.json: size_cerca (=size_max), size_lejos, '
              'size_obj (size a 0,375 m),\nx_min/x_max (x_rango) y tilt.')

    except KeyboardInterrupt:
        pass
    finally:
        try:
            mover(0.0, 0.0, 0.1)
        except Exception:
            pass
        if filas:
            with open(csv_path, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=['etapa', 'size', 'x', 'y', 'sep', 'yaw'])
                w.writeheader()
                w.writerows(filas)
        if len(resumen) > 2:
            with open(json_path, 'w') as fh:
                json.dump(resumen, fh, indent=2)
        nodo.destroy_node()


# =====================================================================
# Modo --comprobar: verificación del entorno antes de entrenar
# =====================================================================

def comprobar(cfg, usar_sim, n_pasos=20):
    """Comprueba el entorno de punta a punta y mide el coste por paso.

    Verifica: (1) llegan blobs y la observación está dentro del espacio;
    (2) cada observación es POSTERIOR a la acción (latencia > 0; además
    step() ya lanza un error si no); (3) segundos por paso, dato que hay que
    declarar en la memoria para fijar el presupuesto de entrenamiento.
    La acción usada es [0, 0] (robot parado), que no mueve nada.
    """
    env = RoboboSeguimientoEnv(cfg, usar_simulador=usar_sim, verbose=True)
    try:
        obs, _ = env.reset()
        print('obs inicial:', np.round(obs, 3))
        print('dentro del espacio:', env.observation_space.contains(obs))

        tiempos, lat, vistos, fuera = [], [], 0, 0
        for _ in range(n_pasos):
            t0 = time.time()
            obs, r, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
            tiempos.append(time.time() - t0)
            lat.append(info['latencia_obs'])
            vistos += int(info['visible'])
            fuera += int(not env.observation_space.contains(obs))
            if term or trunc:
                break
        print('\n--- resumen ({} pasos) ---'.format(len(tiempos)))
        print('segundos por paso: media {:.3f}, máx {:.3f}'.format(
            statistics.mean(tiempos), max(tiempos)))
        print('latencia obs - acción: mín {:.3f} s, media {:.3f} s (debe ser > 0)'.format(
            min(lat), statistics.mean(lat)))
        print('pelota vista en {}/{} pasos'.format(vistos, len(tiempos)))
        print('observaciones fuera del espacio:', fuera)
    finally:
        env.close()


# =====================================================================
# Modos --aleatorio y --entrenar (Stable-Baselines3)
# =====================================================================

def crear_env_monitorizado(cfg, usar_sim, nombre):
    """Entorno con Monitor: escribe un CSV por episodio (recompensa, duración y
    causa de terminación), que son los datos de las curvas de aprendizaje."""
    from stable_baselines3.common.monitor import Monitor
    os.makedirs('logs', exist_ok=True)
    env = RoboboSeguimientoEnv(cfg, usar_simulador=usar_sim)
    return Monitor(env, filename=os.path.join('logs', nombre),
                   info_keywords=('causa',))


def aleatorio(cfg, usar_sim, episodios, nombre='aleatorio'):
    """Línea base: política aleatoria en el propio entorno de entrenamiento."""
    env = crear_env_monitorizado(cfg, usar_sim, nombre)
    try:
        for ep in range(episodios):
            env.reset()
            fin, total = False, 0.0
            while not fin:
                _, r, term, trunc, info = env.step(env.action_space.sample())
                total += r
                fin = term or trunc
            print('episodio {}: recompensa {:.2f}, causa {}'.format(ep, total, info['causa']))
    finally:
        env.close()


def entrenar(cfg, usar_sim, pasos, semilla, nombre):
    """Entrena SAC. Se elige un algoritmo fuera de política porque cada paso
    cuesta tiempo real (~0,4 s) y SAC reutiliza cada muestra muchas veces."""
    from stable_baselines3 import SAC
    nombre = '{}_s{}'.format(nombre, semilla)
    env = crear_env_monitorizado(cfg, usar_sim, nombre)
    try:
        model = SAC('MlpPolicy', env, seed=semilla, learning_starts=500,
                    verbose=1, tensorboard_log='tb')
        model.learn(total_timesteps=pasos, tb_log_name=nombre)
        os.makedirs('modelos', exist_ok=True)
        model.save(os.path.join('modelos', nombre))
        with open(os.path.join('modelos', nombre + '_config.json'), 'w') as fh:
            json.dump(cfg, fh, indent=2)           # la config viaja con el modelo
        print('Modelo guardado en modelos/{}.zip'.format(nombre))
    finally:
        env.close()


# =====================================================================
# Programa principal
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument('--calibrar', action='store_true')
    modo.add_argument('--comprobar', action='store_true')
    modo.add_argument('--aleatorio', action='store_true')
    modo.add_argument('--entrenar', action='store_true')
    ap.add_argument('--config', default='config_p1.json')
    ap.add_argument('--sin-simulador', action='store_true',
                    help='robot real: sin módulo sim, reinicio a mano')
    # Opciones de --calibrar
    ap.add_argument('--modo', choices=['acercar', 'alejar', 'continuo'], default='acercar')
    ap.add_argument('--tilt', type=float, default=None)
    ap.add_argument('--dist-inicial', type=float, default=None,
                    help='separación inicial en m (robot real, sin módulo sim)')
    ap.add_argument('--sin-ruido', action='store_true')
    ap.add_argument('--sin-barrido', action='store_true')
    # Opciones de --aleatorio / --entrenar
    ap.add_argument('--episodios', type=int, default=10)
    ap.add_argument('--pasos', type=int, default=20000)
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--nombre', default='ref')
    a = ap.parse_args()

    cfg = cargar_config(a.config)
    usar_sim = not a.sin_simulador

    try:
        if a.calibrar:
            calibrar(cfg, modo=a.modo, barrido=not a.sin_barrido, ruido=not a.sin_ruido,
                     tilt=a.tilt, dist_inicial_m=a.dist_inicial)
        elif a.comprobar:
            comprobar(cfg, usar_sim)
        elif a.aleatorio:
            aleatorio(cfg, usar_sim, a.episodios)
        elif a.entrenar:
            entrenar(cfg, usar_sim, a.pasos, a.semilla, a.nombre)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()