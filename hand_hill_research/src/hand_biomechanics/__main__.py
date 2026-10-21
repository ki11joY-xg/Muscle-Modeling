"""Запуск: python -m hand_biomechanics --scenario periodic --output results."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from .activation import MovementScenario
from .model import HandModel
from .visualization import HandVisualizer


def main():
    parser = argparse.ArgumentParser(description="Моделирование движения пальцев кисти")
    parser.add_argument("--scenario", choices=[s.value for s in MovementScenario if s != MovementScenario.CUSTOM], default="fist_clench")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--points", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.duration <= 0 or args.points < 2:
        parser.error("duration must be > 0 and points must be >= 2")
    args.output.mkdir(parents=True, exist_ok=True)
    model = HandModel()
    t, y = model.simulate(MovementScenario(args.scenario), t_span=(0, args.duration), t_eval=np.linspace(0, args.duration, args.points))
    np.savez_compressed(args.output / "trajectory.npz", t=t, y=y)
    viz = HandVisualizer(model)
    viz.plot_angles(t, y, save_path=str(args.output / "angles.png"))
    viz.animate_3d(t, y, interval=50)
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
