from before_we_act.benchmark import TASKS as BENCHMARK_TASKS
from before_we_act.r13n import SPLIT_EPISODES, TASKS, TASK_SPECS


def test_r13n_contract_has_exact_no_stack_portfolio():
    assert TASKS == (
        "lift_barrier",
        "camera_alignment",
        "long_pipeline_delivery",
        "take_photo",
        "pass_shoe",
        "place_food",
    )
    assert tuple(BENCHMARK_TASKS) == TASKS
    assert not any("stack" in task for task in TASKS)
    assert set(TASK_SPECS) == set(TASKS)
    assert SPLIT_EPISODES == {"train": 120, "validation": 15, "test": 15}


def test_place_food_is_deliberately_global_only():
    assert TASK_SPECS["place_food"]["camera_order"] == ("global",)
    assert TASK_SPECS["place_food"]["agents"] == 2
