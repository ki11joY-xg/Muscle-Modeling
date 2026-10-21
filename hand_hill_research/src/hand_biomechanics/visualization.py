"""Графики и анимация результатов моделирования."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .activation import ActivationFactory, MovementScenario
from .model import HandModel

class HandVisualizer:
    """Класс для визуализации модели кисти"""

    def __init__(self, hand_model: HandModel):
        self.model = hand_model
        self._init_geometry()

    def _init_geometry(self):
        """Инициализация геометрии для визуализации"""
        # Длины пястных костей (м)
        self.mc_lengths = {2: 0.069, 3: 0.072, 4: 0.066, 5: 0.060}

        # Положения MCP суставов
        self.mcp_positions = {}
        for finger in [2, 3, 4, 5]:
            phi = self.model.fingers[finger].phi
            self.mcp_positions[finger] = np.array([
                self.mc_lengths[finger] * np.cos(phi),
                self.mc_lengths[finger] * np.sin(phi),
                0.0
            ])

        # Цвета для пальцев
        self.finger_colors = {2: 'red', 3: 'green', 4: 'blue', 5: 'purple'}

    def get_joint_positions(self, t_idx: int, sol_y: np.ndarray) -> Dict:
        """Получение положений суставов для заданного временного индекса"""
        positions = {}
        wrist = np.array([0, 0, 0])

        for finger in [2, 3, 4, 5]:
            theta1, theta2, theta3 = self.model.get_finger_angles(sol_y, finger)
            theta1_val = theta1[t_idx] if hasattr(theta1, '__len__') else theta1
            theta2_val = theta2[t_idx] if hasattr(theta2, '__len__') else theta2
            theta3_val = theta3[t_idx] if hasattr(theta3, '__len__') else theta3

            l1, l2, l3 = self.model.fingers[finger].lengths
            phi = self.model.fingers[finger].phi

            # Локальные координаты
            pip_local = np.array([
                l1 * np.cos(theta1_val),
                0,
                -l1 * np.sin(theta1_val)
            ])

            dip_local = pip_local + np.array([
                l2 * np.cos(theta1_val + theta2_val),
                0,
                -l2 * np.sin(theta1_val + theta2_val)
            ])

            tip_local = dip_local + np.array([
                l3 * np.cos(theta1_val + theta2_val + theta3_val),
                0,
                -l3 * np.sin(theta1_val + theta2_val + theta3_val)
            ])

            # Матрица поворота для плоскости пальца
            R = np.array([
                [np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)]
            ])

            mcp = self.mcp_positions[finger]
            pip = mcp + R @ pip_local
            dip = mcp + R @ dip_local
            tip = mcp + R @ tip_local

            positions[finger] = {'MCP': mcp, 'PIP': pip, 'DIP': dip, 'TIP': tip}

        return positions

    def plot_angles(self, t: np.ndarray, sol_y: np.ndarray, save_path: Optional[str] = None):
        """Построение графиков углов"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        finger_titles = {2: 'Палец 2 (указательный)', 3: 'Палец 3 (средный)',
                         4: 'Палец 4 (безымянный)', 5: 'Палец 5 (мизинец)'}
        colors = ['red', 'blue', 'green']
        labels = ['MCP', 'PIP', 'DIP']

        for idx, finger in enumerate([2, 3, 4, 5]):
            ax = axes[idx // 2, idx % 2]
            theta1, theta2, theta3 = self.model.get_finger_angles(sol_y, finger)

            ax.plot(t, theta1 * 180 / np.pi, colors[0], linewidth=2, label=labels[0])
            ax.plot(t, theta2 * 180 / np.pi, colors[1], linewidth=2, label=labels[1])
            ax.plot(t, theta3 * 180 / np.pi, colors[2], linewidth=2, label=labels[2])

            ax.set_xlabel('Время (с)')
            ax.set_ylabel('Угол (градусы)')
            ax.set_title(finger_titles[finger])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-10, 90)

        plt.suptitle('Кинематика пальцев', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_activation(self, t: np.ndarray, scenario: MovementScenario = None, **kwargs):
        """Построение графиков активации мышц"""
        if scenario is not None:
            activations = ActivationFactory.create(scenario, **kwargs)
        else:
            activations = ActivationFactory.create(MovementScenario.FIST_CLENCH)

        t_plot = np.linspace(0, max(t) if len(t) > 0 else 6, 500)

        a_fdp = [activations['FDP'](tt) for tt in t_plot]
        a_fds = [activations['FDS'](tt) for tt in t_plot]
        a_edc = [activations['EDC'](tt) for tt in t_plot]

        plt.figure(figsize=(10, 5))
        plt.plot(t_plot, a_fdp, 'r-', label='FDP', linewidth=2)
        plt.plot(t_plot, a_fds, 'b-', label='FDS', linewidth=2)
        plt.plot(t_plot, a_edc, 'g-', label='EDC', linewidth=2)
        plt.xlabel('Время (с)')
        plt.ylabel('Активация')
        plt.title('Активация мышц')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.show()

    def animate_3d(self, t: np.ndarray, sol_y: np.ndarray,
                   interval: int = 50, save_path: Optional[str] = None):
        """
        Создание 3D анимации кисти

        Параметры:
            t: массив времени
            sol_y: решение (углы и скорости)
            interval: интервал между кадрами (мс)
            save_path: путь для сохранения анимации (опционально)
        """
        n_frames = len(t)

        # Предварительный расчет положений
        all_positions = [self.get_joint_positions(i, sol_y) for i in range(n_frames)]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        wrist = np.array([0, 0, 0])

        def update(frame):
            ax.clear()
            positions = all_positions[frame]

            # Пястные кости
            for finger in [2, 3, 4, 5]:
                mcp = positions[finger]['MCP']
                ax.plot([wrist[0], mcp[0]], [wrist[1], mcp[1]], [wrist[2], mcp[2]],
                        color='gray', linewidth=3, alpha=0.7)

            # Фаланги и суставы
            for finger in [2, 3, 4, 5]:
                color = self.finger_colors[finger]
                mcp = positions[finger]['MCP']
                pip = positions[finger]['PIP']
                dip = positions[finger]['DIP']
                tip = positions[finger]['TIP']

                ax.plot([mcp[0], pip[0]], [mcp[1], pip[1]], [mcp[2], pip[2]],
                        color=color, linewidth=3)
                ax.plot([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]],
                        color=color, linewidth=2)
                ax.plot([dip[0], tip[0]], [dip[1], tip[1]], [dip[2], tip[2]],
                        color=color, linewidth=2)

                ax.scatter(*mcp, color='gold', s=50, zorder=5)
                ax.scatter(*pip, color='orange', s=40, zorder=5)
                ax.scatter(*dip, color='tomato', s=30, zorder=5)
                ax.scatter(*tip, color='black', s=20, zorder=5)

            ax.scatter(*wrist, color='gray', s=80, zorder=5)


            ax.set_xlim(-0.05, 0.15)
            ax.set_ylim(-0.08, 0.08)
            ax.set_zlim(-0.08, 0.05)
            ax.view_init(elev=30, azim=60)
            ax.set_xlabel('X (м)')
            ax.set_ylabel('Y (м)')
            ax.set_zlabel('Z (м)')
            ax.set_title(f'3D модель кисти, t = {t[frame]:.2f} с')
            ax.set_box_aspect([1.5, 1.0, 0.8])

            if frame == 0:
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], color='gray', linewidth=3, label='Пястные кости')]
                for finger in [2, 3, 4, 5]:
                    legend_elements.append(Line2D([0], [0], color=self.finger_colors[finger],
                                                  linewidth=3, label=f'Палец {finger}'))
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

        anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
        plt.tight_layout()

        plt.show()

        if save_path:
            anim.save(save_path, writer='pillow', fps=1000 / interval, savefig_kwargs={'facecolor': 'white'}, dpi=80)
            print(f'Анимация сохранена в {save_path}')

        return anim

