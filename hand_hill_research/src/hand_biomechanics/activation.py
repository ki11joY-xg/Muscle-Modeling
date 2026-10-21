"""Наборы заданных во времени возбуждений мышц (не динамика активации)."""
from __future__ import annotations
from enum import Enum
from typing import Callable, Dict
import numpy as np

class MovementScenario(Enum):
    """Доступные сценарии движений"""
    FIST_CLENCH = "fist_clench"  # Сжатие в кулак
    FIST_OPEN = "fist_open"  # Разжатие кулака
    ISOMETRIC = "isometric"  # Изометрическое усилие
    STEP_RESPONSE = "step_response"  # Ступенчатое воздействие
    PERIODIC = "periodic"  # Периодическое движение
    SEQUENTIAL = "sequential"  # Поочередное сгибание пальцев
    BALLISTIC = "ballistic"  # Баллистическое движение
    CUSTOM = "custom"  # Пользовательский сценарий


class ActivationFunctions:
    """Класс-контейнер для функций активации мышц"""

    # ------------------------------------------------------------------------
    # 1. СЦЕНАРИЙ: Сжатие в кулак (по умолчанию)
    # ------------------------------------------------------------------------
    @staticmethod
    def fist_clench_FDP(t: float) -> float:
        if t < 1.0:
            return 0.05
        elif t < 2.5:
            return 0.05 + 0.7 * (t - 1.0) / 1.5
        elif t < 4.0:
            return 0.75
        elif t < 5.5:
            return 0.75 - 0.7 * (t - 4.0) / 1.5
        else:
            return 0.05

    @staticmethod
    def fist_clench_FDS(t: float) -> float:
        if t < 1.0:
            return 0.05
        elif t < 2.5:
            return 0.05 + 0.6 * (t - 1.0) / 1.5
        elif t < 4.0:
            return 0.65
        elif t < 5.5:
            return 0.65 - 0.6 * (t - 4.0) / 1.5
        else:
            return 0.05

    @staticmethod
    def fist_clench_EDC(t: float) -> float:
        if t < 1.0:
            return 0.10
        elif t < 2.5:
            return 0.10 - 0.05 * (t - 1.0) / 1.5
        elif t < 4.0:
            return 0.05
        elif t < 5.5:
            return 0.05 + 0.05 * (t - 4.0) / 1.5
        else:
            return 0.10

    @staticmethod
    def fist_clench_synergist(t: float) -> float:
        """Синергистская активация (для Lumbricales, Interossei)"""
        return 0.2 * ActivationFunctions.fist_clench_FDP(t)

    # ------------------------------------------------------------------------
    # 2. СЦЕНАРИЙ: Разжатие кулака
    # ------------------------------------------------------------------------
    @staticmethod
    def fist_open_FDP(t: float) -> float:
        if t < 1.0:
            return 0.75  # Начинаем сжатым
        elif t < 2.5:
            return 0.75 - 0.7 * (t - 1.0) / 1.5
        elif t < 4.0:
            return 0.05
        elif t < 5.5:
            return 0.05 + 0.7 * (t - 4.0) / 1.5
        else:
            return 0.75

    @staticmethod
    def fist_open_FDS(t: float) -> float:
        if t < 1.0:
            return 0.65
        elif t < 2.5:
            return 0.65 - 0.6 * (t - 1.0) / 1.5
        elif t < 4.0:
            return 0.05
        elif t < 5.5:
            return 0.05 + 0.6 * (t - 4.0) / 1.5
        else:
            return 0.65

    @staticmethod
    def fist_open_EDC(t: float) -> float:
        if t < 1.0:
            return 0.05
        elif t < 2.5:
            return 0.05 + 0.05 * (t - 1.0) / 1.5
        elif t < 4.0:
            return 0.10
        elif t < 5.5:
            return 0.10 - 0.05 * (t - 4.0) / 1.5
        else:
            return 0.05

    # ------------------------------------------------------------------------
    # 3. СЦЕНАРИЙ: Изометрическое усилие (постоянная активация)
    # ------------------------------------------------------------------------
    @staticmethod
    def isometric_FDP(t: float, level: float = 0.5) -> float:
        return level

    @staticmethod
    def isometric_FDS(t: float, level: float = 0.4) -> float:
        return level

    @staticmethod
    def isometric_EDC(t: float, level: float = 0.05) -> float:
        return level

    # ------------------------------------------------------------------------
    # 4. СЦЕНАРИЙ: Ступенчатое воздействие
    # ------------------------------------------------------------------------
    @staticmethod
    def step_response_FDP(t: float, on_time: float = 1.0, off_time: float = 3.0, level: float = 0.7) -> float:
        return level if on_time <= t < off_time else 0.05

    @staticmethod
    def step_response_FDS(t: float, on_time: float = 1.0, off_time: float = 3.0, level: float = 0.6) -> float:
        return level if on_time <= t < off_time else 0.05

    @staticmethod
    def step_response_EDC(t: float, on_time: float = 1.0, off_time: float = 3.0) -> float:
        return 0.05 if on_time <= t < off_time else 0.10

    # ------------------------------------------------------------------------
    # 5. СЦЕНАРИЙ: Периодическое движение
    # ------------------------------------------------------------------------
    @staticmethod
    def periodic_FDP(t: float, period: float = 2.0, amplitude: float = 0.3, offset: float = 0.5) -> float:
        return offset + amplitude * np.sin(2 * np.pi * t / period)

    @staticmethod
    def periodic_FDS(t: float, period: float = 2.0, amplitude: float = 0.3, offset: float = 0.4) -> float:
        return offset + amplitude * np.sin(2 * np.pi * t / period)

    @staticmethod
    def periodic_EDC(t: float, period: float = 2.0, amplitude: float = 0.05, offset: float = 0.10) -> float:
        return offset - amplitude * np.sin(2 * np.pi * t / period)

    # ------------------------------------------------------------------------
    # 6. СЦЕНАРИЙ: Поочередное сгибание пальцев
    # ------------------------------------------------------------------------
    # Для этого сценария нужны отдельные функции на палец
    @staticmethod
    def sequential_FDP_finger(t: float, finger: int, duration: float = 1.0,
                              start_time: float = None, level: float = 0.7) -> float:
        if start_time is None:
            start_times = {2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}
            start_time = start_times.get(finger, 1.0)

        if start_time <= t < start_time + duration:
            return level
        return 0.05

    # ------------------------------------------------------------------------
    # 7. СЦЕНАРИЙ: Баллистическое движение (быстрое)
    # ------------------------------------------------------------------------
    @staticmethod
    def ballistic_FDP(t: float) -> float:
        if t < 0.1:
            return 0.05
        elif t < 0.2:
            return 0.05 + 0.7 * (t - 0.1) / 0.1
        elif t < 0.3:
            return 0.75
        elif t < 0.5:
            return 0.75 - 0.7 * (t - 0.3) / 0.2
        else:
            return 0.05

    @staticmethod
    def ballistic_FDS(t: float) -> float:
        if t < 0.1:
            return 0.05
        elif t < 0.2:
            return 0.05 + 0.6 * (t - 0.1) / 0.1
        elif t < 0.3:
            return 0.65
        elif t < 0.5:
            return 0.65 - 0.6 * (t - 0.3) / 0.2
        else:
            return 0.05

    @staticmethod
    def ballistic_EDC(t: float) -> float:
        if t < 0.1:
            return 0.10
        elif t < 0.2:
            return 0.10 - 0.05 * (t - 0.1) / 0.1
        elif t < 0.3:
            return 0.05
        elif t < 0.5:
            return 0.05 + 0.05 * (t - 0.3) / 0.2
        else:
            return 0.10


class ActivationFactory:
    """Фабрика для создания функций активации по сценарию"""

    @staticmethod
    def create(scenario: MovementScenario, **kwargs) -> Dict[str, Callable]:
        """
        Создает набор функций активации для заданного сценария

        Параметры:
            scenario: сценарий движения
            **kwargs: дополнительные параметры для сценариев
                - level_FDP, level_FDS, level_EDC: для изометрического
                - on_time, off_time: для ступенчатого
                - period, amplitude: для периодического
                - finger_durations, finger_levels: для последовательного
        """

        if scenario == MovementScenario.FIST_CLENCH:
            return {
                'FDP': ActivationFunctions.fist_clench_FDP,
                'FDS': ActivationFunctions.fist_clench_FDS,
                'EDC': ActivationFunctions.fist_clench_EDC,
                'synergist': ActivationFunctions.fist_clench_synergist
            }

        elif scenario == MovementScenario.FIST_OPEN:
            return {
                'FDP': ActivationFunctions.fist_open_FDP,
                'FDS': ActivationFunctions.fist_open_FDS,
                'EDC': ActivationFunctions.fist_open_EDC,
                'synergist': ActivationFunctions.fist_clench_synergist
            }

        elif scenario == MovementScenario.ISOMETRIC:
            level_fdp = kwargs.get('level_FDP', 0.5)
            level_fds = kwargs.get('level_FDS', 0.4)
            level_edc = kwargs.get('level_EDC', 0.05)

            return {
                'FDP': lambda t: ActivationFunctions.isometric_FDP(t, level_fdp),
                'FDS': lambda t: ActivationFunctions.isometric_FDS(t, level_fds),
                'EDC': lambda t: ActivationFunctions.isometric_EDC(t, level_edc),
                'synergist': lambda t: 0.2 * level_fdp
            }

        elif scenario == MovementScenario.STEP_RESPONSE:
            on_time = kwargs.get('on_time', 1.0)
            off_time = kwargs.get('off_time', 3.0)
            level_fdp = kwargs.get('level_FDP', 0.7)
            level_fds = kwargs.get('level_FDS', 0.6)

            return {
                'FDP': lambda t: ActivationFunctions.step_response_FDP(t, on_time, off_time, level_fdp),
                'FDS': lambda t: ActivationFunctions.step_response_FDS(t, on_time, off_time, level_fds),
                'EDC': lambda t: ActivationFunctions.step_response_EDC(t, on_time, off_time),
                'synergist': lambda t: 0.2 * level_fdp if on_time <= t < off_time else 0.01
            }

        elif scenario == MovementScenario.PERIODIC:
            period = kwargs.get('period', 2.0)
            amplitude_fdp = kwargs.get('amplitude_FDP', 0.3)
            amplitude_fds = kwargs.get('amplitude_FDS', 0.3)

            return {
                'FDP': lambda t: ActivationFunctions.periodic_FDP(t, period, amplitude_fdp, 0.5),
                'FDS': lambda t: ActivationFunctions.periodic_FDS(t, period, amplitude_fds, 0.4),
                'EDC': lambda t: ActivationFunctions.periodic_EDC(t, period, 0.05, 0.10),
                'synergist': lambda t: 0.1 + 0.06 * np.sin(2 * np.pi * t / period)
            }

        elif scenario == MovementScenario.SEQUENTIAL:
            durations = kwargs.get('finger_durations', {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
            levels = kwargs.get('finger_levels', {2: 0.7, 3: 0.7, 4: 0.7, 5: 0.7})
            start_times = kwargs.get('start_times', {2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0})

            # Для последовательного сценария возвращаем словарь с функциями для каждого пальца
            def make_finger_func(finger):
                return lambda t: ActivationFunctions.sequential_FDP_finger(
                    t, finger, durations[finger], start_times[finger], levels[finger]
                )

            return {
                'FDP_finger': {f: make_finger_func(f) for f in [2, 3, 4, 5]},
                'FDS': lambda t: 0.05,  # Базовый тонус
                'EDC': lambda t: 0.10,
                'synergist': lambda t: 0.01
            }

        elif scenario == MovementScenario.BALLISTIC:
            return {
                'FDP': ActivationFunctions.ballistic_FDP,
                'FDS': ActivationFunctions.ballistic_FDS,
                'EDC': ActivationFunctions.ballistic_EDC,
                'synergist': lambda t: 0.2 * ActivationFunctions.ballistic_FDP(t)
            }

        else:
            # По умолчанию - сжатие в кулак
            return {
                'FDP': ActivationFunctions.fist_clench_FDP,
                'FDS': ActivationFunctions.fist_clench_FDS,
                'EDC': ActivationFunctions.fist_clench_EDC,
                'synergist': ActivationFunctions.fist_clench_synergist
            }

