"""Система ОДУ модели кисти и решение задачи Коши."""
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from .activation import ActivationFactory, MovementScenario
from .domain import Finger, Joint, JointType, Muscle, MuscleGroup, IndividualMuscleGroup

class HandModel:
    """
    Основной класс модели кисти

    Атрибуты:
        fingers: словарь пальцев {index: Finger}
        muscle_groups: словарь групп мышц
        individual_muscles: словарь индивидуальных мышц
        coupling_matrix: матрица связей между пальцами
    """

    def __init__(self):
        self.fingers = {}
        self.muscle_groups = {}
        self.individual_muscles = {}
        self.individual_muscle_groups = {}
        self.coupling_matrix = self._create_coupling_matrix()

        self._initialize_fingers()
        self._initialize_muscles()
        self._initialize_muscle_groups()

    def _create_coupling_matrix(self) -> np.ndarray:
        """Создание матрицы связей между пальцами (4x4)"""
        return np.array([
            [1.0, 0.3, 0.2, 0.1],
            [0.3, 1.0, 0.4, 0.2],
            [0.2, 0.4, 1.0, 0.5],
            [0.1, 0.2, 0.5, 1.0]
        ])

    def _initialize_fingers(self):
        """Инициализация пальцев"""
        # Длины фаланг (м) для пальцев 2-5
        lengths = {
            2: (0.045, 0.028, 0.023),
            3: (0.048, 0.030, 0.024),
            4: (0.044, 0.027, 0.022),
            5: (0.038, 0.023, 0.019),
        }

        # Массы фаланг (кг)
        masses = {
            2: (0.012, 0.008, 0.006),
            3: (0.013, 0.009, 0.007),
            4: (0.011, 0.007, 0.005),
            5: (0.009, 0.006, 0.004),
        }

        # Углы плоскостей (рад)
        phi = {
            2: 10.0 * np.pi / 180.0,
            3: 0.0,
            4: -10.0 * np.pi / 180.0,
            5: -20.0 * np.pi / 180.0,
        }

        # Создаем суставы для каждого пальца
        for finger in [2, 3, 4, 5]:
            joints = {
                JointType.MCP: Joint.create_mcp(),
                JointType.PIP: Joint.create_pip(),
                JointType.DIP: Joint.create_dip()
            }

            self.fingers[finger] = Finger(
                index=finger,
                lengths=lengths[finger],
                masses=masses[finger],
                phi=phi[finger],
                joints=joints
            )

    def _initialize_muscles(self):
        """Инициализация мышц с временными заглушками активации"""
        # Временно задаем нулевые активации, они будут заменены при симуляции
        dummy_activation = lambda t: 0.0

        # Общие мышцы
        self.muscles = {
            'FDP': Muscle('FDP', 200.0, dummy_activation, {}),
            'FDS': Muscle('FDS', 160.0, dummy_activation, {}),
            'EDC': Muscle('EDC', 100.0, dummy_activation, {}),
            'EIP': Muscle('EIP', 30.0, dummy_activation, {}),
            'EDM': Muscle('EDM', 25.0, dummy_activation, {}),
            'Lumbricales': Muscle('Lumbricales', 20.0, dummy_activation, {}),
            'Interossei': Muscle('Interossei', 25.0, dummy_activation, {}),
        }

        # Установка плеч сил
        # FDP (сгибает все суставы)
        for finger in [2, 3, 4, 5]:
            self.muscles['FDP'].moment_arms[JointType.MCP] = 0.012
            self.muscles['FDP'].moment_arms[JointType.PIP] = 0.009
            self.muscles['FDP'].moment_arms[JointType.DIP] = 0.007

        # FDS (сгибает MCP и PIP)
        for finger in [2, 3, 4, 5]:
            self.muscles['FDS'].moment_arms[JointType.MCP] = 0.012
            self.muscles['FDS'].moment_arms[JointType.PIP] = 0.010
            self.muscles['FDS'].moment_arms[JointType.DIP] = 0.000

        # EDC (разгибает все суставы)
        for finger in [2, 3, 4, 5]:
            self.muscles['EDC'].moment_arms[JointType.MCP] = 0.010
            self.muscles['EDC'].moment_arms[JointType.PIP] = 0.008
            self.muscles['EDC'].moment_arms[JointType.DIP] = 0.006

        # EIP (только палец 2)
        self.muscles['EIP'].moment_arms[JointType.MCP] = 0.008
        self.muscles['EIP'].moment_arms[JointType.PIP] = 0.006
        self.muscles['EIP'].moment_arms[JointType.DIP] = 0.004

        # EDM (только палец 5)
        self.muscles['EDM'].moment_arms[JointType.MCP] = 0.008
        self.muscles['EDM'].moment_arms[JointType.PIP] = 0.006
        self.muscles['EDM'].moment_arms[JointType.DIP] = 0.004

        # Lumbricales (червеобразные)
        for finger in [2, 3, 4, 5]:
            self.muscles['Lumbricales'].moment_arms[JointType.MCP] = 0.006
            self.muscles['Lumbricales'].moment_arms[JointType.PIP] = -0.003
            self.muscles['Lumbricales'].moment_arms[JointType.DIP] = -0.002

        # Interossei (межкостные)
        for finger in [2, 3, 4, 5]:
            self.muscles['Interossei'].moment_arms[JointType.MCP] = 0.005
            self.muscles['Interossei'].moment_arms[JointType.PIP] = -0.002
            self.muscles['Interossei'].moment_arms[JointType.DIP] = -0.001

    def _initialize_muscle_groups(self):
        """Инициализация групп мышц"""
        # Группы с общим брюшком
        coeffs_FDP = [0.25, 0.30, 0.25, 0.20]  # пальцы 2,3,4,5
        coeffs_FDS = [0.25, 0.30, 0.25, 0.20]
        coeffs_EDC = [0.30, 0.35, 0.25, 0.10]

        self.muscle_groups = {
            'FDP': MuscleGroup('FDP', self.muscles['FDP'], coeffs_FDP),
            'FDS': MuscleGroup('FDS', self.muscles['FDS'], coeffs_FDS),
            'EDC': MuscleGroup('EDC', self.muscles['EDC'], coeffs_EDC),
        }

        # Индивидуальные группы с матрицей связей
        self.individual_muscle_groups = {
            'Lumbricales': IndividualMuscleGroup('Lumbricales', self.muscles['Lumbricales'], self.coupling_matrix),
            'Interossei': IndividualMuscleGroup('Interossei', self.muscles['Interossei'], self.coupling_matrix),
        }

    def set_activation_functions(self, activations: Dict[str, Callable]):
        """
        Установка функций активации для мышц

        Параметры:
            activations: словарь с ключами 'FDP', 'FDS', 'EDC', 'synergist'
        """
        self.muscles['FDP'].activation_func = activations.get('FDP', lambda t: 0.05)
        self.muscles['FDS'].activation_func = activations.get('FDS', lambda t: 0.05)
        self.muscles['EDC'].activation_func = activations.get('EDC', lambda t: 0.10)

        # Для синергистов (Lumbricales, Interossei) используем общую функцию
        synergist_func = activations.get('synergist', lambda t: 0.01)
        self.muscles['Lumbricales'].activation_func = synergist_func
        self.muscles['Interossei'].activation_func = synergist_func

        # EIP и EDM по умолчанию неактивны
        self.muscles['EIP'].activation_func = lambda t: 0.05
        self.muscles['EDM'].activation_func = lambda t: 0.05

    def get_muscle_moments(self, t: float) -> Dict[int, Dict[JointType, float]]:
        """
        Вычисление мышечных моментов для всех пальцев

        Возвращает:
            словарь {finger: {JointType: момент}}
        """
        moments = {finger: {jt: 0.0 for jt in JointType} for finger in self.fingers}

        # Моменты от общих мышц
        for group in self.muscle_groups.values():
            for joint in JointType:
                group_moments = group.get_moments(t, joint)
                for finger, moment in group_moments.items():
                    moments[finger][joint] += moment

        # Моменты от индивидуальных мышц
        for group in self.individual_muscle_groups.values():
            for joint in JointType:
                group_moments = group.get_moments(t, joint)
                for finger, moment in group_moments.items():
                    moments[finger][joint] += moment

        # Дополнительные мышцы (EIP для пальца 2, EDM для пальца 5)
        if 2 in self.fingers:
            moments[2][JointType.MCP] += self.muscles['EIP'].get_moment(t, JointType.MCP)
            moments[2][JointType.PIP] += self.muscles['EIP'].get_moment(t, JointType.PIP)
            moments[2][JointType.DIP] += self.muscles['EIP'].get_moment(t, JointType.DIP)

        if 5 in self.fingers:
            moments[5][JointType.MCP] += self.muscles['EDM'].get_moment(t, JointType.MCP)
            moments[5][JointType.PIP] += self.muscles['EDM'].get_moment(t, JointType.PIP)
            moments[5][JointType.DIP] += self.muscles['EDM'].get_moment(t, JointType.DIP)

        return moments

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Функция динамики для solve_ivp

        y = [theta1_2, theta2_2, theta3_2, omega1_2, omega2_2, omega3_2,
             theta1_3, theta2_3, theta3_3, omega1_3, omega2_3, omega3_3,
             ... для пальцев 4, 5]
        """
        dy = np.zeros_like(y)
        finger_indices = [2, 3, 4, 5]

        # Получаем мышечные моменты
        muscle_moments = self.get_muscle_moments(t)

        for idx, finger in enumerate(finger_indices):
            base = 6 * idx
            finger_obj = self.fingers[finger]

            theta1 = y[base + 0]
            theta2 = y[base + 1]
            theta3 = y[base + 2]
            omega1 = y[base + 3]
            omega2 = y[base + 4]
            omega3 = y[base + 5]

            # Пассивные моменты
            tau_pass1 = finger_obj.joints[JointType.MCP].passive_moment(theta1, omega1)
            tau_pass2 = finger_obj.joints[JointType.PIP].passive_moment(theta2, omega2)
            tau_pass3 = finger_obj.joints[JointType.DIP].passive_moment(theta3, omega3)

            # Гравитационные моменты
            tau_grav1, tau_grav2, tau_grav3 = finger_obj.gravity_moments(theta1, theta2, theta3)

            # Мышечные моменты
            tau_mus1 = muscle_moments[finger][JointType.MCP]
            tau_mus2 = muscle_moments[finger][JointType.PIP]
            tau_mus3 = muscle_moments[finger][JointType.DIP]

            # Суммарные моменты
            tau_total1 = tau_mus1 + tau_grav1 + tau_pass1
            tau_total2 = tau_mus2 + tau_grav2 + tau_pass2
            tau_total3 = tau_mus3 + tau_grav3 + tau_pass3

            # Эффективные моменты инерции
            I_MCP, I_PIP, I_DIP = finger_obj.effective_inertia(theta2)

            eps = 1e-10

            dy[base + 0] = omega1
            dy[base + 1] = omega2
            dy[base + 2] = omega3
            dy[base + 3] = tau_total1 / (I_MCP + eps)
            dy[base + 4] = tau_total2 / (I_PIP + eps)
            dy[base + 5] = tau_total3 / (I_DIP + eps)

        return dy

    def simulate(self, scenario: MovementScenario = MovementScenario.FIST_CLENCH,
                 t_span: Tuple[float, float] = (0, 6.0),
                 t_eval: Optional[np.ndarray] = None,
                 y0: Optional[np.ndarray] = None,
                 **scenario_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск симуляции модели

        Параметры:
            scenario: сценарий движения
            t_span: интервал времени (start, end)
            t_eval: точки времени для оценки
            y0: начальные условия (если None - используются стандартные)
            **scenario_kwargs: параметры для сценария
        """
        # Установка функций активации
        activations = ActivationFactory.create(scenario, **scenario_kwargs)
        self.set_activation_functions(activations)

        # Начальные условия
        if y0 is None:
            theta1_init = 0.2  # рад
            theta2_init = 0.4  # рад
            theta3_init = 0.2  # рад

            y0 = np.zeros(24)
            for i in range(4):
                y0[6 * i + 0] = theta1_init
                y0[6 * i + 1] = theta2_init
                y0[6 * i + 2] = theta3_init

        # Точки времени
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 600)

        # Решение
        print(f"Запуск симуляции: {scenario.value}")
        sol = solve_ivp(self.dynamics, t_span, y0, t_eval=t_eval,
                        method='BDF', rtol=1e-6, atol=1e-8)

        print(f"Решение получено для {len(sol.t)} временных точек")

        return sol.t, sol.y

    def get_finger_angles(self, sol_y: np.ndarray, finger: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Извлечение углов для заданного пальца"""
        finger_map = {2: 0, 3: 1, 4: 2, 5: 3}
        idx = finger_map[finger]
        base = 6 * idx

        theta1 = sol_y[base + 0, :]
        theta2 = sol_y[base + 1, :]
        theta3 = sol_y[base + 2, :]

        return theta1, theta2, theta3

