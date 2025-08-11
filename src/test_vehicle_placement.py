import numpy as np
import src.vehicle_placement as VP

def test_lateral_adjustment():
    test_cases = [
        (0, 0, 0), 
        (0, np.pi / 2, -np.pi / 2),  
        (1, 0, 0),  
        (1, -np.pi / 2, np.pi / 2),  
        (0.5, np.pi / 4, np.pi / 4), 
    ]

    for i, (latitude, angle_offset, expected) in enumerate(test_cases):
        result = VP.lateral_adjustment(latitude, angle_offset)
        assert result == expected, f"{i}: Expected {expected}, but got {result}"
    print("Vehicle Placement Test: All lateral adjustment tests PASSED")


def test_open_loop_adjustment():
    test_cases = [
        (0, 0.5, 0, 0),  
        (
            0,
            0.5,
            np.pi / 2,
            np.pi / 2,
        ), 
        (1, 0.5, 0, 0), 
        (
            1,
            0.5,
            -np.pi / 2,
            -np.pi / 2,
        ), 
        (0, 0, 0, 0),  
        (1, 1, 0, 0),  
        (0, 0, np.pi / 2, 0), 
        (1, 1, -np.pi / 2, 0),  
        (
            0,
            0,
            -3 * np.pi / 2,
            -np.pi / 2,
        ), 
        (
            1,
            1,
            3 * np.pi / 2,
            np.pi / 2,
        ), 
        (
            0.5,
            0.5,
            np.pi / 4,
            np.pi / 4,
        ),  
        (
            0.5,
            0.5,
            -np.pi / 4,
            -np.pi / 4,
        ),  
        (
            0,
            0.5,
            -3 * np.pi / 4,
            -np.pi / 2,
        ), 
        (
            1,
            0.5,
            3 * np.pi / 4,
            np.pi / 2,
        ),  
        (
            0,
            0,
            -np.pi / 2,
            -np.pi / 2,
        ), 
        (
            1,
            1,
            np.pi / 2,
            np.pi / 2,
        ),  
    ]

    for i, (longitude, latitude, angle_offset, expected) in enumerate(test_cases):
        result = VP.open_loop_adjustment(longitude, latitude, angle_offset)
        assert result == expected, f"{i}: Expected {expected}, but got {result}"
    print("Vehicle Placement Test: All open loop adjustment tests PASSED")

def run_tests():
    test_lateral_adjustment()
    test_open_loop_adjustment()
    print("Vehicle Placement Test: All tests PASSED.\n")

if __name__ == "__main__":
    run_tests()
