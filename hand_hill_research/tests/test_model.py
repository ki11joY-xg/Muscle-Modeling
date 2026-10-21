import numpy as np
import pytest
from hand_biomechanics import HandModel, Joint, JointType, MovementScenario
from hand_biomechanics.activation import ActivationFactory


def test_joint_passive_moment_neutral():
    j = Joint(JointType.MCP, k=0.8, b=0.07, theta_neutral=0.2)
    assert j.passive_moment(0.2, 0.0) == pytest.approx(0.0)


def test_force_distribution_fdp():
    m = HandModel()
    m.set_activation_functions(ActivationFactory.create(MovementScenario.ISOMETRIC))
    f = m.muscle_groups["FDP"].get_forces(0)
    assert sum(f.values()) == pytest.approx(100.0)


def test_rhs_state_shape():
    m = HandModel()
    m.set_activation_functions(ActivationFactory.create(MovementScenario.FIST_CLENCH))
    dydt = m.dynamics(0.0, np.zeros(24))
    assert dydt.shape == (24,)
    assert np.all(np.isfinite(dydt))


def test_short_simulation():
    m = HandModel()
    t, y = m.simulate(MovementScenario.ISOMETRIC, t_span=(0, 0.02), t_eval=np.linspace(0, 0.02, 5))
    assert t.shape == (5,)
    assert y.shape == (24, 5)
    assert np.all(np.isfinite(y))
