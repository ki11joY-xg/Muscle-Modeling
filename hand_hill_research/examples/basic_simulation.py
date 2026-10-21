"""Минимальный воспроизводимый опыт (после pip install -e .)."""
from pathlib import Path
import numpy as np
from hand_biomechanics import HandModel, HandVisualizer, MovementScenario

out = Path("results")
out.mkdir(exist_ok=True)
model = HandModel()
t, y = model.simulate(MovementScenario.FIST_CLENCH, t_eval=np.linspace(0, 6, 300))
np.savez_compressed(out / "fist_clench.npz", t=t, y=y)
HandVisualizer(model).plot_angles(t, y, save_path=str(out / "fist_clench_angles.png"))
