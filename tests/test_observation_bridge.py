import numpy as np

from policies.closed_loop import (
    extract_pose_context,
    get_action_from_scripted,
    make_env,
    observation_to_local_history,
    unwrap_reset,
    unwrap_step,
)


def _step_once(scenario: str):
    env = make_env(seed=1, scenario=scenario)
    obs, _ = unwrap_reset(env.reset())
    action = get_action_from_scripted(env, obs)
    obs, _, _, info = unwrap_step(env.step(action))
    return obs, info, action


def test_local_obs_agents_bridge_shapes():
    for scenario in ["nominal", "hard_comm"]:
        obs, info, action = _step_once(scenario)
        local_obs_agents = np.asarray(info["local_obs_agents"], dtype=np.float32)
        assert local_obs_agents.shape == (2, 17)

        hist = observation_to_local_history(obs, info, history=8, fallback_action=action)
        assert tuple(hist.shape) == (2, 8, 17)

        rel_pose, object_pose = extract_pose_context(info, ego_id=0)
        assert tuple(rel_pose.shape) == (3,)
        assert tuple(object_pose.shape) == (3,)


def test_hard_comm_local_obs_agents_are_agent_specific():
    _, info, _ = _step_once("hard_comm")
    local_obs_agents = np.asarray(info["local_obs_agents"], dtype=np.float32)
    assert not np.allclose(local_obs_agents[0], local_obs_agents[1])
