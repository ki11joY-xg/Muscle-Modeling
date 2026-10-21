"""Структуры суставов, фаланг и мышц. Единицы: SI."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Tuple
import numpy as np

class JointType(Enum):
    """Типы суставов"""
    MCP = "MCP"
    PIP = "PIP"
    DIP = "DIP"


@dataclass
class Joint:
    """
    Класс сустава с пассивными свойствами

    Атрибуты:
        name: название сустава
        k: угловая жесткость (Н·м/рад)
        b: коэффициент демпфирования (Н·м·с/рад)
        theta_neutral: нейтральный угол (рад)
    """
    name: JointType
    k: float = 0.0
    b: float = 0.0
    theta_neutral: float = 0.0

    def passive_moment(self, theta: float, omega: float) -> float:
        """Вычисление пассивного момента в суставе"""
        return -self.k * (theta - self.theta_neutral) - self.b * omega

    @classmethod
    def create_mcp(cls) -> 'Joint':
        return cls(JointType.MCP, k=0.8, b=0.07, theta_neutral=0.2)

    @classmethod
    def create_pip(cls) -> 'Joint':
        return cls(JointType.PIP, k=0.5, b=0.05, theta_neutral=0.4)

    @classmethod
    def create_dip(cls) -> 'Joint':
        return cls(JointType.DIP, k=0.3, b=0.03, theta_neutral=0.2)


@dataclass
class Muscle:
    """
    Класс мышцы

    Атрибуты:
        name: название мышцы
        F_max: максимальная изометрическая сила (Н)
        activation_func: функция активации a(t) -> [0,1]
        moment_arms: словарь {сустав: плечо силы} (м)
    """
    name: str
    F_max: float
    activation_func: Callable[[float], float]
    moment_arms: Dict[JointType, float] = field(default_factory=dict)

    def get_force(self, t: float) -> float:
        """Получить текущую силу мышцы"""
        return self.activation_func(t) * self.F_max

    def get_moment(self, t: float, joint: JointType) -> float:
        """Получить момент в заданном суставе"""
        return self.get_force(t) * self.moment_arms.get(joint, 0.0)

    def set_moment_arm(self, joint: JointType, arm: float) -> 'Muscle':
        """Установить плечо силы для сустава (chainable)"""
        self.moment_arms[joint] = arm
        return self


class MuscleGroup:
    """
    Группа мышц с общим брюшком (например, FDP, FDS, EDC)

    Атрибуты:
        name: название группы
        muscle: базовая мышца группы
        distribution_coeffs: коэффициенты распределения по пальцам [2,3,4,5]
    """

    def __init__(self, name: str, muscle: Muscle, distribution_coeffs: List[float]):
        self.name = name
        self.muscle = muscle
        self.distribution_coeffs = np.array(distribution_coeffs)
        self.finger_indices = [2, 3, 4, 5]

    def get_forces(self, t: float) -> Dict[int, float]:
        """Получить силы для каждого пальца"""
        total_force = self.muscle.get_force(t)
        return {finger: coeff * total_force
                for finger, coeff in zip(self.finger_indices, self.distribution_coeffs)}

    def get_moments(self, t: float, joint: JointType) -> Dict[int, float]:
        """Получить моменты для каждого пальца в заданном суставе"""
        forces = self.get_forces(t)
        arm = self.muscle.moment_arms.get(joint, 0.0)
        return {finger: force * arm for finger, force in forces.items()}


class IndividualMuscleGroup:
    """
    Группа индивидуальных мышц (червеобразные, межкостные)
    С учетом матрицы связей между пальцами
    """

    def __init__(self, name: str, muscle: Muscle, coupling_matrix: np.ndarray):
        self.name = name
        self.muscle = muscle
        self.coupling_matrix = coupling_matrix
        self.finger_indices = [2, 3, 4, 5]
        self.idx_map = {2: 0, 3: 1, 4: 2, 5: 3}

    def get_forces(self, t: float) -> Dict[int, float]:
        """Получить силы с учетом связей между пальцами"""
        base_activation = self.muscle.activation_func(t)
        a_vec = np.array([base_activation] * 4)
        a_real = self.coupling_matrix @ a_vec

        forces = {}
        for finger, idx in self.idx_map.items():
            forces[finger] = a_real[idx] * self.muscle.F_max

        return forces

    def get_moments(self, t: float, joint: JointType) -> Dict[int, float]:
        """Получить моменты для каждого пальца в заданном суставе"""
        forces = self.get_forces(t)
        arm = self.muscle.moment_arms.get(joint, 0.0)
        return {finger: force * arm for finger, force in forces.items()}


@dataclass
class Finger:
    """
    Класс пальца

    Атрибуты:
        index: номер пальца (2-5)
        lengths: длины фаланг (P1, P2, P3) в м
        masses: массы фаланг (кг)
        phi: угол плоскости пальца (рад)
        joints: словарь суставов
    """
    index: int
    lengths: Tuple[float, float, float]
    masses: Tuple[float, float, float]
    phi: float
    joints: Dict[JointType, Joint]

    def __post_init__(self):
        self.inertias = self._compute_inertias()
        self.cm_positions = self._compute_cm_positions()

    def _compute_inertias(self) -> Tuple[float, float, float]:
        """Вычисление моментов инерции (стержневая модель)"""
        return tuple((1.0 / 12.0) * m * l ** 2
                     for m, l in zip(self.masses, self.lengths))

    def _compute_cm_positions(self) -> Tuple[float, float, float]:
        """Положения центров масс от проксимального конца"""
        return tuple(l / 2 for l in self.lengths)

    def effective_inertia(self, theta2: float) -> Tuple[float, float, float]:
        """Эффективные моменты инерции для суставов"""
        l1, l2 = self.lengths[0], self.lengths[1]
        m2, m3 = self.masses[1], self.masses[2]
        i1, i2, i3 = self.inertias

        I_MCP = (i1 + i2 + i3 +
                 m2 * l1 ** 2 +
                 m3 * (l1 ** 2 + l2 ** 2 + 2 * l1 * l2 * np.cos(theta2)))
        I_PIP = i2 + i3 + m3 * l2 ** 2
        I_DIP = i3

        return I_MCP, I_PIP, I_DIP

    def gravity_moments(self, theta1: float, theta2: float, theta3: float) -> Tuple[float, float, float]:
        """Вычисление гравитационных моментов"""
        g = 9.81
        l1, l2, l3 = self.lengths
        m1, m2, m3 = self.masses

        c1, s1 = np.cos(theta1), np.sin(theta1)
        c12, s12 = np.cos(theta1 + theta2), np.sin(theta1 + theta2)
        c123, s123 = np.cos(theta1 + theta2 + theta3), np.sin(theta1 + theta2 + theta3)

        tau1 = -g * (m1 * (l1 / 2) * c1 +
                     m2 * (l1 * c1 + (l2 / 2) * c12) +
                     m3 * (l1 * c1 + l2 * c12 + (l3 / 2) * c123))

        tau2 = -g * (m2 * (l2 / 2) * c12 +
                     m3 * (l2 * c12 + (l3 / 2) * c123))

        tau3 = -g * (m3 * (l3 / 2) * c123)

        return tau1, tau2, tau3

