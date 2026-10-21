"""Примерные сценарии, сохраненные из исходного прототипа."""
from __future__ import annotations
import numpy as np
from .activation import MovementScenario
from .model import HandModel
from .visualization import HandVisualizer

def print_scenario_help():
    """Вывод справки по доступным сценариям"""
    help_text = """
    ============================================================================
    ДОСТУПНЫЕ СЦЕНАРИИ ДВИЖЕНИЙ
    ============================================================================

    1. FIST_CLENCH (по умолчанию) - Сжатие в кулак
       Параметры: нет

    2. FIST_OPEN - Разжатие кулака
       Параметры: нет

    3. ISOMETRIC - Изометрическое усилие
       Параметры:
           level_FDP (float, по умолч. 0.5) - активация FDP
           level_FDS (float, по умолч. 0.4) - активация FDS
           level_EDC (float, по умолч. 0.05) - активация EDC

    4. STEP_RESPONSE - Ступенчатое воздействие
       Параметры:
           on_time (float, по умолч. 1.0) - время включения
           off_time (float, по умолч. 3.0) - время выключения
           level_FDP (float, по умолч. 0.7) - уровень активации FDP
           level_FDS (float, по умолч. 0.6) - уровень активации FDS

    5. PERIODIC - Периодическое движение
       Параметры:
           period (float, по умолч. 2.0) - период колебаний (с)
           amplitude_FDP (float, по умолч. 0.3) - амплитуда FDP
           amplitude_FDS (float, по умолч. 0.3) - амплитуда FDS

    6. SEQUENTIAL - Поочередное сгибание пальцев
       Параметры:
           finger_durations (dict, по умолч. {2:1,3:1,4:1,5:1}) - длительности
           finger_levels (dict, по умолч. {2:0.7,3:0.7,4:0.7,5:0.7}) - уровни
           start_times (dict, по умолч. {2:1,3:2,4:3,5:4}) - времена начала

    7. BALLISTIC - Баллистическое движение (быстрое)
       Параметры: нет

    8. CUSTOM - Пользовательский сценарий (требуется передать activations)

    ============================================================================
    """
    print(help_text)


def run_example():
    """Пример запуска модели с различными сценариями"""

    print("=" * 60)
    print("ЗАПУСК ООП-МОДЕЛИ КИСТИ")
    print("=" * 60)

    # Создание модели
    model = HandModel()
    visualizer = HandVisualizer(model)

    # Вывод справки
    print_scenario_help()

    # ------------------------------------------------------------------------
    # Пример 1: Сжатие в кулак (по умолчанию)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("ПРИМЕР 1: Сжатие в кулак (сценарий по умолчанию)")
    print("=" * 40)

    t, y = model.simulate(MovementScenario.FIST_CLENCH, t_span=(0, 6))
    visualizer.plot_activation(t, MovementScenario.FIST_CLENCH)
    visualizer.plot_angles(t, y)

    # ------------------------------------------------------------------------
    # Пример 2: Изометрическое усилие
    # ------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("ПРИМЕР 2: Изометрическое усилие (50% FDP, 40% FDS)")
    print("=" * 40)

    t2, y2 = model.simulate(MovementScenario.ISOMETRIC, t_span=(0, 3),
                            level_FDP=0.5, level_FDS=0.4, level_EDC=0.05)
    visualizer.plot_activation(t2, MovementScenario.ISOMETRIC,
                               level_FDP=0.5, level_FDS=0.4, level_EDC=0.05)
    visualizer.plot_angles(t2, y2)

    # ------------------------------------------------------------------------
    # Пример 3: Периодическое движение
    # ------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("ПРИМЕР 3: Периодическое движение (период 2 с)")
    print("=" * 40)

    t3, y3 = model.simulate(MovementScenario.PERIODIC, t_span=(0, 3),
                            period=2.0, amplitude_FDP=0.3, amplitude_FDS=0.3)
    visualizer.plot_activation(t3, MovementScenario.PERIODIC,
                               period=2.0, amplitude_FDP=0.3, amplitude_FDS=0.3)

    # ------------------------------------------------------------------------
    # Пример 4: 3D анимация (сжатие в кулак)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("ПРИМЕР 4: 3D анимация (сжатие в кулак)")
    print("=" * 40)

    t_anim, y_anim = model.simulate(MovementScenario.FIST_CLENCH, t_span=(0, 6))
    visualizer.animate_3d(t_anim, y_anim, interval=50)

    # ------------------------------------------------------------------------
    # Вывод финальных углов
    # ------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("ФИНАЛЬНЫЕ УГЛЫ (сжатие в кулак, t=6 с)")
    print("=" * 40)

    for finger in [2, 3, 4, 5]:
        theta1, theta2, theta3 = model.get_finger_angles(y_anim, finger)
        print(f"Палец {finger}: MCP={theta1[-1] * 180 / np.pi:5.1f}°, "
              f"PIP={theta2[-1] * 180 / np.pi:5.1f}°, "
              f"DIP={theta3[-1] * 180 / np.pi:5.1f}°")


def quick_run(scenario: str = "fist_clench", **kwargs):
    """
    Быстрый запуск модели с указанным сценарием

    Параметры:
        scenario: название сценария ('fist_clench', 'isometric', 'periodic',
                                   'step_response', 'sequential', 'ballistic')
        **kwargs: параметры для сценария
    """
    scenario_map = {
        'fist_clench': MovementScenario.FIST_CLENCH,
        'fist_open': MovementScenario.FIST_OPEN,
        'isometric': MovementScenario.ISOMETRIC,
        'step_response': MovementScenario.STEP_RESPONSE,
        'periodic': MovementScenario.PERIODIC,
        'sequential': MovementScenario.SEQUENTIAL,
        'ballistic': MovementScenario.BALLISTIC,
    }

    selected = scenario_map.get(scenario, MovementScenario.FIST_CLENCH)

    model = HandModel()
    visualizer = HandVisualizer(model)

    print(f"Запуск сценария: {selected.value}")
    t, y = model.simulate(selected, **kwargs)

    visualizer.plot_activation(t, selected, **kwargs)
    visualizer.plot_angles(t, y)

    return model, visualizer, t, y

